from typing import Dict, List, Tuple
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
import torch
import os
import sys
import math
import numpy as np
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
from inversion import * 
from diffusers.utils import load_image
from torchvision.utils import save_image
import torch.nn as nn
from diffusers.models.attention_processor import (CustomDiffusionAttnProcessor, 
                                                  CustomDiffusionAttnProcessor2_0, 
                                                  CustomDiffusionXFormersAttnProcessor)
from diffusers.loaders import AttnProcsLayers
import itertools
from torch import FloatTensor

def fr_params(params):
    for param in params:
        param.requires_grad = False

class InstaStyle(StableDiffusionXLPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel, 
                 text_encoder_2: CLIPTextModelWithProjection, 
                 tokenizer: CLIPTokenizer, 
                 tokenizer_2: CLIPTokenizer, 
                 unet: UNet2DConditionModel, 
                 scheduler: KarrasDiffusionSchedulers, 
                 image_encoder: CLIPVisionModelWithProjection = None, 
                 feature_extractor: CLIPImageProcessor = None, 
                 force_zeros_for_empty_prompt: bool = True, 
                 add_watermarker: bool | None = None):
        super().__init__(vae,
                         text_encoder, 
                         text_encoder_2, 
                         tokenizer, 
                         tokenizer_2, 
                         unet, 
                         scheduler, 
                         image_encoder, 
                         feature_extractor, 
                         force_zeros_for_empty_prompt, 
                         add_watermarker)
        
    def __create_opt_token__(self, **kwargs):
        modifier_token_id = []
        initializer_token_id = []
    
        kwargs["modifier_token"] = kwargs["modifier_token"].split("+")
        kwargs["initializer_token"] = kwargs["initializer_token"].split("+")
        if len(kwargs["modifier_token"]) > len(kwargs["initializer_token"]):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(
            kwargs["modifier_token"], kwargs["initializer_token"][: len(kwargs["modifier_token"])]
        ):
            # Add the placeholder token in tokenizer
            num_added_tokens = self.tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = self.tokenizer.encode([initializer_token], add_special_tokens=False)
            print(token_ids)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(self.tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        for x, y in zip(modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]

            

    def __freeze_params__(self):
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        params_to_freeze = itertools.chain(
                self.text_encoder.text_model.encoder.parameters(),
                self.text_encoder.text_model.final_layer_norm.parameters(),
                self.text_encoder.text_model.embeddings.position_embedding.parameters(),
            )
        fr_params(params_to_freeze)
        train_kv = True
        train_q_out = False 
        custom_diffusion_attn_procs = {}
        attention_class = CustomDiffusionAttnProcessor

        st = self.unet.state_dict()
        for name, _ in self.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
                "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
            }
            if train_q_out:
                weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
                weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
                weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
            if cross_attention_dim is not None:
                custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=train_kv,
                    train_q_out=train_q_out,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(self.unet.device)
                custom_diffusion_attn_procs[name].load_state_dict(weights)
            else:
                custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=False,
                    train_q_out=False,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
        del st
        self.unet.set_attn_processor(custom_diffusion_attn_procs)

        #custom_diffusion_layers = AttnProcsLayers(self.unet.attn_processors)
        #accelerator.register_for_checkpointing(custom_diffusion_layers)

    def __call__(self, 
                 prompt: str | List[str] = None,
                 prompt_2: str | List[str] | None = None,
                 height: int | None = None, 
                 width: int | None = None, 
                 num_inference_steps: int = 50, 
                 timesteps: List[int] = None, 
                 denoising_end: float | None = None, 
                 guidance_scale: float = 5, 
                 negative_prompt: str | List[str] | None = None, 
                 negative_prompt_2: str | List[str] | None = None, 
                 num_images_per_prompt: int | None = 1, 
                 eta: float = 0, 
                 generator = None, 
                 latents: FloatTensor | None = None, 
                 prompt_embeds: FloatTensor | None = None, 
                 negative_prompt_embeds: FloatTensor | None = None, 
                 pooled_prompt_embeds: FloatTensor | None = None, 
                 negative_pooled_prompt_embeds: FloatTensor | None = None, 
                 ip_adapter_image = None, 
                 output_type: str | None = "pil", 
                 return_dict: bool = True, 
                 cross_attention_kwargs = None, 
                 guidance_rescale: float = 0, 
                 original_size: Tuple[int, int] | None = None, 
                 crops_coords_top_left: Tuple[int, int] = ..., 
                 target_size: Tuple[int, int] | None = None, 
                 negative_original_size: Tuple[int, int] | None = None, 
                 negative_crops_coords_top_left: Tuple[int, int] = ..., 
                 negative_target_size: Tuple[int, int] | None = None, 
                 clip_skip: int | None = None, 
                 callback_on_step_end: Callable[[int, int, Dict], None] | None = None, 
                 callback_on_step_end_tensor_inputs: List[str] = ...,
                 **kwargs):
        return super().__call__(prompt,
                                prompt_2, 
                                height, 
                                width, 
                                num_inference_steps, 
                                timesteps, 
                                denoising_end, 
                                guidance_scale, 
                                negative_prompt, 
                                negative_prompt_2, 
                                num_images_per_prompt, 
                                eta, 
                                generator, 
                                latents, 
                                prompt_embeds, 
                                negative_prompt_embeds, 
                                pooled_prompt_embeds, 
                                negative_pooled_prompt_embeds, 
                                ip_adapter_image, 
                                output_type, 
                                return_dict, 
                                cross_attention_kwargs, 
                                guidance_rescale, 
                                original_size, 
                                crops_coords_top_left, 
                                target_size, 
                                negative_original_size, 
                                negative_crops_coords_top_left, 
                                negative_target_size, 
                                clip_skip, 
                                callback_on_step_end, 
                                callback_on_step_end_tensor_inputs, 
                                **kwargs)
    def forward(self, batch):
        latents = self.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

                # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        return model_pred, target
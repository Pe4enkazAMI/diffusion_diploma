from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch
import os
import sys
import handler as hd
import math
import numpy as np
import utility.inversion as inversion
from diffusers.utils import load_image
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torchvision.utils import save_image

def make_prompts(prompts, src_style):
    for i in range(1, len(prompts)):
        prompts[i] = f'{prompts[i]}, {src_style}.'
    return prompts

@hydra.main(version_base=None, config_path="", config_name="config_sa")
def main(cfg):
    OmegaConf.resolve(cfg)

    print(f'{OmegaConf.to_yaml(cfg)}')
    scheduler = instantiate(cfg["scheduler"])

    # scheduler = DDIMScheduler(
    # beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
    # clip_sample=False, set_alpha_to_one=False)

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True,
        scheduler=scheduler
    ).to("cuda")



    src_style = cfg["desired_style"]
    src_prompt = f'{cfg["desired_prompt"]}, {src_style}.'
    image_path = cfg["ref_image_path"]

    prompts = make_prompts(cfg["prompts"])
    prompts.append(src_prompt)

    num_inference_steps = cfg["num_steps"]
    x0 = np.array(load_image(image_path).resize((1024, 1024)))
    zts = inversion.ddim_inversion(pipeline, x0, src_prompt, num_inference_steps, 2)

    shared_score_shift = cfg["shared_score_shift"]  # higher value induces higher fidelity, set 0 for no shift
    shared_score_scale = cfg["shared_score_scale"]  # higher value induces higher, set 1 for no rescale

    handler = hd.Handler(pipeline)

    sa_args = instantiate(cfg["handler"])
    # sa_args = handler.StyleAlignedArgs(
    # share_group_norm=True, share_layer_norm=True, share_attention=True,
    # adain_queries=True, adain_keys=True, adain_values=False,
    # shared_score_shift=shared_score_shift, shared_score_scale=shared_score_scale,)
    handler.register(sa_args)

    zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)

    g_cpu = torch.Generator(device='cpu')
    g_cpu.manual_seed(10)

    latents = torch.randn(len(prompts), 4, 128, 128, device='cpu', generator=g_cpu,
                      dtype=pipeline.unet.dtype,).to('cuda:0')
    latents[0] = zT

    images_a = pipeline(prompts, latents=latents,
                    callback_on_step_end=inversion_callback,
                    num_inference_steps=num_inference_steps, guidance_scale=10.0).images

    handler.remove()
    save_image(images_a)


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    print("start training")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
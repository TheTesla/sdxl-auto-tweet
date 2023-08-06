#!/usr/bin/env python3

from diffusers import DiffusionPipeline
import torch

repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")

def make_image(prompt, seed=42, h=1024, w=1024, save=True):
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(prompt=prompt,height=h,width=w,generator=generator).images[0]
    #name = prompt.replace(' ', '_').replace(',', '_')
    name = prompt
    if save:
        image.save(f'images/{name}_HEIGHT_{h}_WIDTH_{w}_SEED_{seed:010d}.png')
    return image

prompt = "A photograph of a group of men from Indonesia wearing inflatable shorts."

for i in range(1000):
    make_image(prompt, i, h=1024, w=1024)


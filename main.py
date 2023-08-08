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

#prompt = "A photograph of a group of men from Indonesia wearing inflatable shorts."
#prompt = "Many gay men from India are wearing inflatable shorts."
#prompts = [ "a photograph of the interior of a McDonalds restaurant in the style of Sanifair",
#        "Saniflair",
#        "a McDonalds restaurant in the style of Sanifair",
#        "a Sanifair fastfood restaurant",
#        "a Saniflair fastfood restaurant",
#        "a fastfood restaurant with Sinfair flair",
#        "a Saniflair restaurant",
#        "a Saniflair McDonalds"
#        ]
#prompts = [ "A man from Japan is wearing inflatable shorts.",
#            "A man in Japan is wearing inflatable shorts.",
#            "A man from China is wearing inflatable shorts.",
#            "A man in China is wearing inflatable shorts.",
#            "A man from Nigeria is wearing inflatable shorts.",
#            "A man in Nigeria is wearing inflatable shorts.",
#            "A man from India is wearing inflatable shorts.",
#            "A man in India is wearing inflatable shorts.",
#        ]
#prompts = [ "A young man from Japan is wearing inflatable shorts.",
#            "A young man in Japan is wearing inflatable shorts.",
#            "A young man from China is wearing inflatable shorts.",
#            "A young man in China is wearing inflatable shorts.",
#            "A young man from Nigeria is wearing inflatable shorts.",
#            "A young man in Nigeria is wearing inflatable shorts.",
#            "A young man from India is wearing inflatable shorts.",
#            "A young man in India is wearing inflatable shorts.",
#        ]
#prompts = [ "A capybara is eating apples.",
#            "capybara",
#            "many capybaras",
#            "rainbow capybaras",
#            "Don't worry, be capy.",
#            "Many capybaras are having fun in a forest.",
#            "A capybara is swimming in a see.",
#            "Many capybaras are swimming in river.",
#        ]
prompts = [ "A young man from Japan is wearing an inflatable suit.",
            "A young man in Japan is wearing an inflatable suit.",
            "A young man from China is wearing an inflatable suit.",
            "A young man in China is wearing an inflatable suit.",
            "A young man from Nigeria is wearing an inflatable suit.",
            "A young man in Nigeria is wearing an inflatable suit.",
            "A young man from India is wearing an inflatable suit.",
            "A young man in India is wearing an inflatable suit.",
        ]
for i in range(0,100000):
    prompt = prompts[i%8]
    seed = int(i/8)
    print(f'SEED {seed:010d} - {prompt}')
    make_image(prompt, seed, h=1024, w=1024)


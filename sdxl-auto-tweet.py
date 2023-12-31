#!/usr/bin/env python3

from diffusers import DiffusionPipeline
import torch
import time

import tweepy
from secret import twitter_auth_keys

repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")

def make_image(prompt, seed=42, h=1024, w=1024, save=True):
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(prompt=prompt,height=h,width=w,generator=generator).images[0]
    name = prompt
    filename = f'images/{name}_HEIGHT_{h}_WIDTH_{w}_SEED_{seed:010d}.png'
    if save:
        image.save(filename)
    return image, filename


def post(text, image_filename, keys):
    tweepy_auth = tweepy.OAuth1UserHandler(
        keys['consumer_key'],
        keys['consumer_secret'],
        keys['access_token'],
        keys['access_token_secret']
    )
    
    tweepy_api = tweepy.API(tweepy_auth)
    
    client = tweepy.Client(
        consumer_key=keys['consumer_key'], #API Key
        consumer_secret=keys['consumer_secret'], #API Secret
        access_token=keys['access_token'],
        access_token_secret=keys['access_token_secret']
    )

    post = tweepy_api.simple_upload(image_filename)
    response = client.create_tweet(text=text, media_ids=[post.media_id])
    print(f"https://twitter.com/user/status/{response.data['id']}")



#tags = "#inflatableshorts #inflatablefetish #sdxl #stablediffusionxl #ai #aiart #inflatable #inflatables #shorts"
tags = "#sdxl #stablediffusionxl #ai #aiart #aigenerated"

prompts = [ [
            "a restaurant with all chairs in the shape of a toilet",
            "A gay couple is wearing inflatable pants.",
            "a paradies, where cats can fly",
            "Many capybaras are swimming in a rainbow colored river.",
            "Many men from different cultures are wearing inflatable pants.",
            "a chessboard with wooden chess pieces in the shape of capybaras",
            "Some men are wearing largely inflated inflatable shorts.",
            "Many animals are having fun in a capybaradies. "
], [
            "A capybara is eating apples.",
            "capybara",
            "many capybaras",
            "rainbow capybaras",
            "Don't worry, be capy.",
            "Many capybaras are having fun in a forest.",
            "A capybara is swimming in a sea.",
            "Many capybaras are swimming in river.",
], [
            "A couple of colorful birds be enjoying life.",
            "Many colorful birds are flying around.",
            "A beautiful bird is eating.",
            "Three birds are singing.",
            "A group of beautiful birds is playing.",
            "Two parrots are playing together.",
            "Many beautiful birds are having fun in a paradies.",
            "Bathing birds."
], [
            "A young man from Japan is wearing an inflatable suit.",
            "A young man in Japan is wearing an inflatable suit.",
            "A young man from China is wearing an inflatable suit.",
            "A young man in China is wearing an inflatable suit.",
            "A young man from Nigeria is wearing an inflatable suit.",
            "A young man in Nigeria is wearing an inflatable suit.",
            "A young man from India is wearing an inflatable suit.",
            "A young man in India is wearing an inflatable suit."
]
]

#h = 512
#w = 512
h = 1024
w = 1024

try:
    with open("statefile", "r") as f:
        state_str = f.read()
    i_start = int(state_str.split("=")[1])
except Exception as e:
    print(e)
    i_start = 0

for i in range(i_start,100000):
    with open("statefile", "w") as f:
        f.write(f"i={i}")
    accNr = i%4
    sel = int(i/4) % 8
    seed = int(i/32)
    prompt = prompts[accNr][sel]
    text = f'#prompt: "{prompt}"\n\nheight: {h}\nwidth: {w}\nseed: {seed}\n\n{tags}'
    print(f'i={i}')
    print(text)
    _, filename = make_image(prompt, seed, h=h, w=w)

    while True:
        try:
            post(text, filename, twitter_auth_keys[accNr])
        except Exception as e:
            print(e)
            time.sleep(10)
            continue
        break 



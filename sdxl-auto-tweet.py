#!/usr/bin/env python3

from diffusers import DiffusionPipeline
import torch

import tweepy
import secret

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


tweepy_auth = tweepy.OAuth1UserHandler(
    secret.twitter_auth_keys['consumer_key'],
    secret.twitter_auth_keys['consumer_secret'],
    secret.twitter_auth_keys['access_token'],
    secret.twitter_auth_keys['access_token_secret']
)

tweepy_api = tweepy.API(tweepy_auth)

client = tweepy.Client(
    consumer_key=secret.twitter_auth_keys['consumer_key'], #API Key
    consumer_secret=secret.twitter_auth_keys['consumer_secret'], #API Secret
    access_token=secret.twitter_auth_keys['access_token'],
    access_token_secret=secret.twitter_auth_keys['access_token_secret']
)



prompt = "A high detail photograph of a group of men from China wearing inflatable shorts."
tags = "#inflatableshorts #inflatablefetish #sdxl #stablediffusionxl #ai #aiart #inflatable #inflatables #shorts"

#h = 512
#w = 512
h = 1024
w = 1024

for i in range(1,100000000):
    _, filename = make_image(prompt, i, h=h, w=w)
    text = f'#prompt: "{prompt}"\n\nheight: {h}\nwidth: {w}\nseed: {i}\n\n{tags}'


    tweepy_auth = tweepy.OAuth1UserHandler(
        secret.twitter_auth_keys['consumer_key'],
        secret.twitter_auth_keys['consumer_secret'],
        secret.twitter_auth_keys['access_token'],
        secret.twitter_auth_keys['access_token_secret']
    )
    
    tweepy_api = tweepy.API(tweepy_auth)
    
    client = tweepy.Client(
        consumer_key=secret.twitter_auth_keys['consumer_key'], #API Key
        consumer_secret=secret.twitter_auth_keys['consumer_secret'], #API Secret
        access_token=secret.twitter_auth_keys['access_token'],
        access_token_secret=secret.twitter_auth_keys['access_token_secret']
    )

    post = tweepy_api.simple_upload(filename)
    response = client.create_tweet(text=text, media_ids=[post.media_id])
    print(f"https://twitter.com/user/status/{response.data['id']}")



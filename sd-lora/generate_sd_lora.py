#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate.py
加载本地 SDXL + LoRA 权重，生成单张图片
"""
import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

# ========== 用户只改这里 ==========
model_id   = "./stable-diffusion"     # 原始 SDXL 目录
lora_path  = "./out-lora/checkpoint-800/lora.safetensors"  # 你的 LoRA 路径
prompt     = "golden snub-nosed monkey by the Sea"
negative   = "low quality, blurry"
width      = 1024
height     = 1024
seed       = 42
device     = "cuda"
# ========== 结束 ==========

# 加载管道
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

# 加载 LoRA（WebUI 格式）
pipe.load_lora_weights(lora_path)

# 生成
generator = torch.Generator(device).manual_seed(seed)
image = pipe(
    prompt=prompt,
    negative_prompt=negative,
    width=width,
    height=height,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator
).images[0]

# 保存
image.save("output.png")
print("已保存为 output.png")
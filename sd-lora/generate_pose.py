#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_pose.py
SDXL + LoRA + ControlNet(OpenPose) 姿势控制生成

https://ai.gitcode.com/hf_mirrors/thibaud/controlnet-openpose-sdxl-1.0/tree/main
https://www.modelscope.cn/models/lllyasviel/Annotators

https://www.modelscope.cn/models/stabilityai/stable-diffusion-xl-base-1.0
"""
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
from PIL import Image

# ========== 用户参数 ==========
base_model   = "./stable-diffusion"            # 原始 SDXL
lora_path    = "./out-lora/checkpoint-800/lora.safetensors"  # 你的 LoRA
pose_path    = "pose.png"                      # 参考姿势图（骨架）
prompt       = "a girl in cyberpunk style"  # S* 是你学的文本反演/token
negative     = "low quality, blurry"
width        = 1024
height       = 1024
num_steps    = 30
guidance     = 7.5
seed         = 42
device       = "cuda"
# ========== 结束 ==========

# 1. 加载 ControlNet
controlnet = ControlNetModel.from_pretrained(
    "./models/controlnet-openpose-sdxl",
    torch_dtype=torch.float16
)

# 2. 加载 SDXL + ControlNet 管道
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

# 3. 加载 LoRA（WebUI 格式）
#pipe.load_lora_weights(lora_path)

# 4. 加载姿势图
pose_image = Image.open(pose_path).convert("RGB").resize((width, height))

# 5. 生成
generator = torch.Generator(device).manual_seed(seed)
image = pipe(
    prompt=prompt,
    negative_prompt=negative,
    image=pose_image,
    width=width,
    height=height,
    num_inference_steps=num_steps,
    guidance_scale=guidance,
    generator=generator,
    controlnet_conditioning_scale=1.0,  # 姿势强度 0-1
).images[0]

# 6. 保存
image.save("output_pose.png")
print("已保存为 output_pose.png")
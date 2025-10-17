#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_sdxl_lora.py
本地 ImageFolder + SDXL + LoRA（无 datasets mock）
用法：
CUDA_VISIBLE_DEVICES=1 accelerate launch \
  --mixed_precision fp16 --num_processes 1 \
  train_sdxl_lora.py \
  --pretrained_model_name_or_path ./stable-diffusion \
  --train_data_dir ./data \
  --output_dir ./out-lora \
  --resolution 1024 --train_batch_size 1 \
  --gradient_checkpointing \
  --max_train_steps 1000 --checkpointing_steps 200 \
  --use_spacetime --train_text_encoder
"""
import argparse, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
from transformers import CLIPTokenizer

logger = get_logger(__name__)


# ---------- 本地 ImageFolder（无 datasets mock） ----------
class LocalImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.captions = []
        for img_path, _ in self.samples:
            txt_path = Path(img_path).with_suffix('.txt')
            self.captions.append(
                txt_path.read_text().strip() if txt_path.exists() else ""
            )

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return {"pixel_values": img, "text": self.captions[index]}


# ----------- 主函数 -----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True, help="本地图片文件夹（含 txt）")
    parser.add_argument("--output_dir", type=str, default="out-lora")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_spacetime", action="store_true", help="是否插入时空分离插件")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        gradient_accumulation_plugin=None,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs")),
    )

    # 1. tokenizers & noise scheduler
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # 2. load SDXL
    pipe = StableDiffusionXLPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float32)
    vae, unet, text_encoder_one, text_encoder_two = pipe.vae, pipe.unet, pipe.text_encoder, pipe.text_encoder_2
    vae.requires_grad_(False)

    # 3. LoRA
    # lora_conf = LoraConfig(
    #     r=args.rank,
    #     lora_alpha=args.lora_alpha,
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    #     lora_dropout=0.0,
    #     bias="none",
    # )
    lora_conf = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",  # SDXL Attention
            "k_proj", "q_proj", "v_proj", "out_proj",  # 兼容写法
        ],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_conf)
    if args.train_text_encoder:
        text_encoder_one = get_peft_model(text_encoder_one, lora_conf)
        text_encoder_two = get_peft_model(text_encoder_two, lora_conf)

    # 4. 梯度检查点
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # 5. 本地 ImageFolder（无 datasets mock）
    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    class LocalImageFolder(ImageFolder):
        def __init__(self, root, transform=None):
            super().__init__(root, transform=transform)
            self.captions = []
            for img_path, _ in self.samples:
                txt_path = Path(img_path).with_suffix('.txt')
                self.captions.append(
                    txt_path.read_text().strip() if txt_path.exists() else ""
                )

        def __getitem__(self, index):
            img, _ = super().__getitem__(index)
            return {"pixel_values": img, "text": self.captions[index]}

    dataset = LocalImageFolder(root=args.train_data_dir, transform=train_transforms)

    # 6. 当场 tokenize 的 collate_fn
    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        texts = [item["text"] for item in batch]
        tok1 = tokenizer_one(texts, max_length=tokenizer_one.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        tok2 = tokenizer_two(texts, max_length=tokenizer_two.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {"pixel_values": pixel_values, "input_ids": tok1.input_ids, "input_ids_1": tok2.input_ids}

    train_dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
        num_workers=0, collate_fn=collate_fn
    )

    # 7. 优化器 & 调度器
    params = list(unet.parameters())
    if args.train_text_encoder:
        params += list(text_encoder_one.parameters()) + list(text_encoder_two.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-2)
    max_train_steps = args.max_train_steps or args.num_train_epochs * math.ceil(len(train_dataloader))
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=max_train_steps)

    # 8. accelerate 准备
    unet, text_encoder_one, text_encoder_two, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder_one, text_encoder_two, vae, optimizer, train_dataloader, lr_scheduler
    )

    # 9. 训练循环
    global_step = 0
    for epoch in range(math.ceil(max_train_steps // len(train_dataloader))):
        print(1)
        # print(2)
        for step, batch in enumerate(train_dataloader):
            # print(3)
            pixel_values = batch["pixel_values"].to(dtype=accelerator.unwrap_model(unet).dtype)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # SDXL 条件生成
            with torch.no_grad():
                encoder_hidden_states1 = accelerator.unwrap_model(text_encoder_one)(batch["input_ids"])
                encoder_hidden_states2 = accelerator.unwrap_model(text_encoder_two)(batch["input_ids_1"])
            encoder_hidden_states = torch.cat([encoder_hidden_states1.last_hidden_state, encoder_hidden_states2.last_hidden_state], dim=-1)
            pooled_embeds = encoder_hidden_states2.last_hidden_state[:, 0]  # pooled
            height, width = args.resolution, args.resolution
            time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=latents.device).repeat(bsz, 1)
            added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": time_ids}

            model_pred = accelerator.unwrap_model(unet)(
                noisy_latents, timesteps, encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs
            ).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            # print(4)
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % args.checkpointing_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    save_path.mkdir(exist_ok=True, parents=True)
                    unet_save = accelerator.unwrap_model(unet)
                    unet_save.save_pretrained(save_path / "unet_lora")
                    if args.train_text_encoder:
                        accelerator.unwrap_model(text_encoder_one).save_pretrained(save_path / "text_encoder_lora")
                        accelerator.unwrap_model(text_encoder_two).save_pretrained(save_path / "text_encoder_2_lora")
                    state_dict = get_peft_model_state_dict(unet_save)
                    save_file(state_dict, save_path / "lora.safetensors")
                    logger.info(f"Saved checkpoint to {save_path}")
                if global_step >= max_train_steps:
                    break

    accelerator.end_training()


if __name__ == '__main__':
    main()
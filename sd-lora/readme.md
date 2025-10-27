# 🐒 金丝猴 LoRA + 姿势控制 全流程

> 从「拍照片 → LoRA 训练 → 姿势控制生成」一步不少，全部可跑。

---

## 🎯 项目目标
1. 用 **30 张金丝猴照片** 训练 **SDXL LoRA**
2. 用 **OpenPose 控制姿势**，生成「**姿势锁死 + 概念保留**」的金丝猴图
3. **CLIP score 当在线 Gate**，>0.30 自动放行

---

## 📦 硬件 & 环境
- **GPU**：≥12 GB（SDXL fp16）
- **系统**：Linux / WSL2 / macOS
- **环境**：
  ```bash
  pip install torch torchvision transformers diffusers accelerate controlnet-aux xformers
  ```

---

## 📁 项目结构
```
golden-monkey-lora/
├── data/                     # 原始照片 + txt
│   ├── golden_monkey_001.png
│   └── golden_monkey_001.txt
├── models/                   # 下载的权重
│   ├── stable-diffusion/     # SDXL 基础
│   ├── sd-controlnet-openpose-sdxl/
│   └── annotator/
├── out-lora/                 # 训练输出
├── generate_pose.py          # 姿势控制生成
├── clip_score.py             # CLIP score + 残差图
├── generate_sd_lora.py       # 微调脚本
└── README.md                 # 本文件
```

---

## 🐒 1. 数据准备（30 张图即可）
### ① 拍照/找图
- **统一 1024×1024**，正面/侧面/跳跃姿态
- 命名：`golden_monkey_001.png … 030.png`

### ② 打标签
- 每图同内容 txt：**`golden monkey`**
- 放同一目录：`data/`

---

## 🧪 2. 训练 LoRA（SDXL）
```bash
accelerate launch train_sdxl_lora.py \
  --pretrained_model_name_or_path ./models/stable-diffusion \
  --train_data_dir ./data \
  --output_dir ./out-lora \
  --resolution 1024 --train_batch_size 1 --gradient_checkpointing \
  --max_train_steps 500 --checkpointing_steps 100 \
  --rank 32 --lora_alpha 16 --train_text_encoder \
  --use_spacetime --seed 42
```
**产出**：`out-lora/checkpoint-500/lora.safetensors`

---

## 🕺 3. 姿势控制生成（OpenPose 锁姿势）
### ① 下载 ControlNet（SDXL 版）以及姿势标注
```bash
modelscope download --model lllyasviel/Annotators
mkdir -p models/controlnet/sd-controlnet-openpose-sdxl
wget https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors -O models/controlnet/sd-controlnet-openpose-sdxl/diffusion_pytorch_model.safetensors
wget https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/raw/main/config.json -O models/controlnet/sd-controlnet-openpose-sdxl/config.json
```

### ② 生成脚本
```python
# generate_pose.py
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
import torch

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
```

---

## 📊 4. CLIP Score 在线 Gate（>0.30 放行）
```python
# clip_score.py
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("output_pose.png")
text = "a golden monkey, S* outfit"
inputs = processor(images=image, text=text, return_tensors=True)

with torch.no_grad():
    image_feat = model.get_image_features(**inputs)
    text_feat  = model.get_text_features(**inputs)
    score = (image_feat @ text_feat.T).squeeze() / (image_feat.norm() * text_feat.norm())

print("CLIP score:", score.item())   # >0.30 放行
```

---

## 🧪 5. 零卷积残差图（眼见为实）
```python
# 在 generate_pose.py 里插 hook
residual_log = []
def hook(m, inp, out):
    down, mid = out
    down_vis = [torch.abs(d[0, 0:1]).clamp(0, 1) for d in down]
    mid_vis  = torch.abs(mid[0, 0:1]).clamp(0, 1)
    residual_log.append((down_vis, mid_vis, torch.mean(torch.abs(mid)).item()))

for blk in controlnet.down_blocks:
    blk.register_forward_hook(hook)
controlnet.mid_block.register_forward_hook(hook)

# 生成后
for step, (down, mid, mean) in enumerate(residual_log):
    grid = torchvision.utils.make_grid(down + [mid], nrow=4, padding=2)
    torchvision.utils.save_image(grid, f"residual_step_{step}.png")
    print(f"step={step}  mid_block_mean={mean:.4f}")   # 眼见为实
```



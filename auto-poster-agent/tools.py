"""
tools.py  ——  本地部署版
所有函数签名保持与之前一致，返回类型不变。
"""
import torch
import os
import json
import base64
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List



from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
from diffusers import StableDiffusionXLPipeline

# ----------- 1. 文本模型加载（全局单例） -----------
# LLM = Llama(
#     model_path="qwen2-7b-instruct-q4_0.gguf",
#     n_ctx=2048,
#     n_gpu_layers=35 if torch.cuda.is_available() else 0  # 有卡就用
# )


from openai import OpenAI

LLM = OpenAI(base_url="http://127.0.0.1:8000/v1",
             api_key="dummy")

#  函数同步/异步任选，这里写同步版
def openai_chat(prompt: str) -> str:
    response = LLM.chat.completions.create(
        model="qwen2_5_coder_7b",          # 本地后端一般忽略，占位即可
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=64
    )
    return response.choices[0].message.content.strip()
# ----------- 2. 文生图模型加载（全局单例） -----------
SD_PIPE = StableDiffusionXLPipeline.from_pretrained(
    "/home/610-wzy/hz_video/sd_lora/stable-diffusion",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

if torch.cuda.is_available():
    SD_PIPE = SD_PIPE.to("cuda")

def call_stable_diffusion(prompt: str) -> str:
    """
    使用 Stable Diffusion XL 生成图像并返回 base64 编码的字符串。
    :param prompt: 输入的文本提示
    :return: base64 编码的图像字符串
    """
    try:
        # 设置原始大小和目标大小
        original_size = target_size = (1024, 1024)
        time_ids = list(original_size) + [0, 0] + list(target_size)

        # 直接使用原始 prompt 进行推理
        image = SD_PIPE(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.0,
            added_cond_kwargs={
                "time_ids": torch.tensor([time_ids], device=SD_PIPE.device, dtype=torch.float32)
            }
        ).images[0]

        # 将生成的图像保存到 BytesIO 缓冲区
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        # 将图像转换为 base64 编码的字符串
        image_base64 = base64.b64encode(buf.getvalue()).decode().replace("\n", "")
        return image_base64
    except Exception as e:
        return "生成图像失败，请稍后再试。"
# ----------- 3. 具体 Agent 函数（签名保持不动） -----------
def llm_agent(product: str, style: str = "") -> str:
    """生成一句 20 字以内的校园社团招新文案固定可指定风格国风"""
    prompt = f"给'{product}'写一句20字以内、带梗的招新文案，风格={style}"
    return openai_chat(prompt)

# def sd_agent(product: str, copy: str, style: str = "国风") -> str:
#     """根据社团名称+文案+风格，返回 base64 PNG 海报（留白二维码区域）"""
#     sd_prompt = f"{product}，{copy}，{style}，社团招新海报，留白二维码，4K"
#     return call_stable_diffusion(sd_prompt)
def sd_agent(product: str, copy: str, style: str = "简约") -> str:
    """根据社团名称+文案+风格，返回本地海报图片路径"""
    import tempfile, os
    sd_prompt = f"{product}用一下内容{copy}生成{style}风格的社团招新海报"
    print(sd_prompt)
    image = call_stable_diffusion(sd_prompt)          # 这一步仍返回 base64

    # 落盘
    path = tempfile.mktemp(suffix=".png")
    print(path)
    with open(path, "wb") as f:
        f.write(base64.b64decode(image))
    return path          # 只给路径
def rag_agent(product: str) -> List[dict]:
    """返回该社团近 3 年招新历史记录（日期、报名人数、男女比例）"""
    csv = "club_history.csv"
    print(f">>>【rag_agent】查找文件：{os.path.abspath(csv)} 存在={os.path.exists(csv)}")
    if not os.path.exists(csv):
        return []

    df = pd.read_csv(csv)
    print(f">>>【rag_agent】CSV 总行数={len(df)}")
    print(f">>>【rag_agent】CSV 列名={list(df.columns)}")
    print(f">>>【rag_agent】筛选 product='{product}' 结果数={len(df[df.club == product])}")
    rows = df[df.club == product].to_dict(orient="records")
    print(f">>>【rag_agent】返回内容：{rows}")
    return rows

def chart_agent(data: List[dict] = None) -> str:
    """把历史报名数据画成折线+饼图，返回本地 png 路径"""
    if isinstance(data, str):
        data = json.loads(data)

    if not data:

        path = tempfile.mktemp(suffix=".png")
        plt.text(0.5, 0.5, "首届招新\n无历史数据", ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    path = tempfile.mktemp(suffix=".png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(df['date'], df['num_apply'], marker='o')
    ax1.set_title("历年报名人数")
    ax1.set_ylabel("人数")
    total_m = df['male'].sum()
    total_f = df['female'].sum()
    ax2.pie([total_m, total_f], labels=['Male', 'Female'], autopct='%1.1f%%')
    ax2.set_title("男女比例")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path

# ---------- 以下同步修改：按路径读图 ----------
def build_pdf(copy: str, img_path: str, chart_path: str, output_path: str):
    """把文案+海报+图表压成 A4 PDF"""
    c = canvas.Canvas(output_path, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, h - 100, copy)
    # 海报（路径读取）
    c.drawImage(ImageReader(img_path), 50, h - 550, width=400, height=600)
    c.showPage()
    # 图表
    c.drawImage(ImageReader(chart_path), 50, h - 450, width=500, height=350)
    c.setFont("Helvetica", 10)
    c.drawString(50, 50, f"生成时间：{datetime.now():%F %T}")
    c.save()

def pdf_agent(copy: str, img_path: str, chart_path: str) -> str:
    """将文案+海报+图表合并成可打印的 A4 PDF，返回文件路径"""
    path = tempfile.mktemp(suffix=".pdf")
    build_pdf(copy, img_path, chart_path, path)
    return path

# def pdf_agent(copy: str, img_path: str, chart_path: str) -> str:
#     """将文案+海报+图表合并成可打印的 A4 PDF，返回文件路径"""
#     path = tempfile.mktemp(suffix=".pdf")
#     build_pdf(copy, img_path, chart_path, path)
#     return path
import os
import base64
from datetime import datetime


def sd_agent1(sd_prompt: str) -> str:
    """
    根据社团名称+文案+风格，返回本地海报图片路径
    :param sd_prompt: 输入的文本提示
    :return: 本地海报图片路径
    """
    try:
        # 调用 call_stable_diffusion 函数生成图像并获取 base64 编码的字符串
        image_base64 = call_stable_diffusion(sd_prompt)

        # 生成文件名，包含时间戳以避免文件名冲突
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"poster_{timestamp}.png"

        # 保存到当前文件夹
        path = os.path.join(os.getcwd(), filename)
        print(f"保存路径: {path}")

        # 将 base64 编码的图像解码并保存到文件
        with open(path, "wb") as f:
            f.write(base64.b64decode(image_base64))

        return path
    except Exception as e:
        print(f"生成海报时出错: {e}")
        return "生成海报失败，请稍后再试。"

if __name__ == '__main__':
    prompt = "Join us with some content from the photography club and capture the most beautiful moments of life! Generate a minimalist style club recruitment poster"
    aaa = sd_agent1(prompt)
    print(f"生成的图像 base64: {aaa}")
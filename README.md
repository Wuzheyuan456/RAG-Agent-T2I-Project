
# AI 应用合集

这里汇总了 4 个即开即用、全部开源的 AI 项目，本地就能跑，按「一句话 → 一张图/一份 PDF/一个答案」思路设计，适合二次开发。

---

## 🌦️ Rainy-Day-Agent
纯本地 200 行 Python 实现「ReAct 搜索 → 计算 → 生图」全流程。  
- vLLM + ChatGLM3-6B / Llama-2-7B，零 OpenAI 依赖  
- 天气查询｜计算器｜Stable Diffusion 三件套  
- 一键启动，5 分钟移植到任意 7B 模型  

[📂 进入目录](./agent)


---

## 📑 RAG 合规问答助手
Streamlit + Milvus + vLLM 构建的「制度文档秒回」项目。  
- 解决底部固定输入、流式不串字、多轮不爆炸  
- 向量+BM25 混合检索，24 G 卡安全运行  
- 单文件上传即可问答，1 分钟完成部署  

rag-milvus:<img width="1920" height="1020" alt="849dadaecbee335926158593102fcfaf" src="https://github.com/user-attachments/assets/a9ea0f8e-4201-4926-b734-741a0d976ada" />
下载：bge-base-zh-v1.5放到model里。
链接：https://www.modelscope.cn/models/BAAI/bge-base-zh-v1.5/files
[📂 进入目录](./rag-milvus)

---

## 🐒 金丝猴 LoRA + 姿势控制
6 张照片训练 SDXL LoRA，再用 OpenPose 锁姿势生成同款金丝猴。  
- 完整数据→训练→生成→CLIP Gate 链路  
- 提供 accelerate、diffusers 脚本与可视化残差图  
- 12 G 显存可跑，一键复现  

[📂 进入目录](./sd-lora)

---



## 🎨 AutoPoster Agent
多智能体协作的「一句话生成中文 PDF 海报」系统。  
- 固定工作流：文案 → 数据 → 图表 → 主视觉 → PDF  
- SD 只画背景，文字 Pillow 叠加，告别伪汉字  
- 无需 LangChain，<30 行核心占位符引擎，极易魔改  

[📂 进入目录](./auto-poster-agent)

---
> 如果对你有帮助，记得给仓库点 ⭐ 鼓励一下 ~
```



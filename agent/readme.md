# 🌦️ Rainy-Day-Agent  
> 200 行 Python 跑通「ReAct 搜索 → 计算 → 生图」全流程，本地 7B 模型即可调用工具，5 分钟移植到任意开源 LLM。

---

## ✨ 特性一览
- **纯本地**：vLLM + ChatGLM3-6B / Llama-2-7B，零 OpenAI 依赖  
- **三工具链**：天气查询 | 计算器 | 图片生成 
- **ReAct 范式**：Thought-Action-Observation 完整可见，可解释性强  
- **一键脚本**：Docker-Compose 编排，CPU 调试 & GPU 生产双模式  
- **生产扩展**：已预留 LangGraph、Redis 缓存、API Gateway 接入点

---

## 🚀 30 秒快速启动

1. 启动本地模型（以 vLLM 为例）  
```bash
vllm serve ./Qwen2-5-Coder-7B-Instruct \
    --dtype float16 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.5 \
    --served-model-name qwen2_5_coder_7b \
    --enable-auto-tool-choice \
--tool-call-parser qwen3_coder   # ← 用官方白名单里的名字
注意解析器一定是 qwen/qwen3_coder 而非 hermes，否则格式对不上
```

2. 拉起 Agent  
```bash
conda create -n react-agent python=3.10
conda activate  react-agent
pip install -r requirements.txt
python react_weather_poster.py
```

3. 提问  
```text
帮我查今日 Beijing 天气并生成一张雨天海报
```

4. 得到  
```
今天 Beijing 小雨，25°C，已为你生成雨天海报：
https://images.unsplash.com/photo-15...
```

---

## 🏗️ 系统架构

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  用户提问    │────▶│  ReAct Agent │────▶│ vLLM 推理    │
└─────────────┘     └──────────────┘     └──────────────┘
                            │调用
               ┌────────────┼────────────┐
               ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │天气API   │ │计算器    │ │图片生成  │
        └──────────┘ └──────────┘ └──────────┘
```

---

## 🔧 核心踩坑 & 解决方案
| 坑点 | 现象 | 根因 | 本项目对策 |
|---|---|---|---|
| 本地模型不调用工具 | 无限 `Thought/Action` 死循环 | 没开 `--enable-auto-tool-choice` | vLLM 启动参数已写明 |
| KeyError `\n "action"` | Agent 初始化崩溃 | 反引号里的 `{}` 被 `.format()` 误解析 | 全部 `{{` `}}` 转义 |
| 中文幻觉 | 模型写小说不返回 JSON | Llama2 中文语料少 | 英文 Prompt + 结构化 JSON |

---

---

## 📚 下一步 Roadmap
- [ ] 迁移 **LangGraph** 支持长时间异步任务  
- [ ] 增加 **Functionary** 微调版 7B，原生 tool-call  
- [ ] 支持 **多模态输入**（图片→BLIP→生图）  
- [ ] **K8s + Helm** 图表，自动扩缩容  
- [ ] 在线体验 Demo（Hugging Face Spaces）

---



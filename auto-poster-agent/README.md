

# 🎉 项目名称：`auto-poster-agent`

> 一个基于多智能体协作的自动化海报生成系统，输入一句话，输出一张 PDF 海报。

---

## 📁 项目结构

```bash
auto-poster-agent/
├── main.py                 # 主入口：运行整个 workflow
├── tools.py/                 # 各个 Agent 模块
│   ├── llm_agent        # 文案生成
│   ├── rag_agent        # 历史数据检索
│   ├── chart_agent      # 图表生成
│   ├── sd_agent         # 文生图（Stable Diffusion）
│   └── pdf_agent        # PDF 合成
├── data/                   # 数据文件
│   └── club_history.csv    # 历史招新数据示例
├── output/                 # 生成的 PDF 输出目录
├── orchestrator.py/                 # 执行器
├── README.md               # 项目说明文档（英文/中文）
├── requirements.txt        # 依赖库
└── reshaper.py              # 请求转换工具
```

---

# auto-poster-agent

一个基于多智能体协作的自动化海报生成系统。输入一句话，自动生成包含文案、图表和图像的 PDF 海报。

## 🚀 功能特点

- ✅ 多 Agent 协作：LLM + RAG + Chart + SD + PDF
- ✅ 动态占位符替换，支持数据传递
- ✅ 支持中文渲染与字体处理
- ✅ 可扩展：支持不同社团、风格、输出格式

## 📦 安装

```bash
pip install -r requirements.txt
```


## 🤝 贡献

欢迎提交 PR 或提出建议！
```

---

## 📦 下一步建议

1. **克隆到本地**：
   ```bash
   git clone https://github.com/yourname/auto-poster-agent.git
   ```

2. **填充真实模型调用**（如 OpenAI API、HuggingFace Token）

3. **添加 `.gitignore`**：
   ```
   __pycache__/
   *.pyc
   .env
   output/
   ```

4. **推送到 GitHub**，开启你的开源之旅！

---

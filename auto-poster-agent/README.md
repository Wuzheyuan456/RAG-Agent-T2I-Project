```
🎨 auto-poster-agent：基于多智能体协作的自动化海报生成系统
输入一句话，输出一张 PDF 海报。一个融合 LLM、Stable Diffusion、数据可视化与流程编排的 AI 应用项目。


一个轻量级但功能完整的 AI Agent 系统，能够根据用户一句话指令（如“为汉服社生成一张国风风格的招新海报”），自动完成：
✅ 文案生成
✅ 历史数据检索与图表绘制
✅ 主视觉图像生成（Stable Diffusion）
✅ 中文 PDF 海报合成

🚀 核心特性

特性 说明
------ ------
多 Agent 协作 模块化设计，5 个 Agent 各司其职
动态数据传递 自研占位符替换机制，实现 Agent 间通信
中文友好 支持中文字体渲染，解决 matplotlib 乱码问题
图像文字清晰 SD 仅生成背景，文字由代码叠加，避免伪汉字
轻量无依赖 无需 LangChain 等重型框架，易于理解和部署

🧩 系统架构

text
用户输入
↓
Workflow 引擎
↓
┌─────────────┐
│ llm_agent │ → 生成招新文案
└─────────────┘
↓
┌─────────────┐
│ rag_agent │ → 检索历史报名数据
└─────────────┘
↓
┌─────────────┐
│ chart_agent │ → 生成趋势图（自动解决中文字体问题）
└─────────────┘
↓
┌─────────────┐
│ sd_agent │ → 文生图生成主视觉（SD v1.5）
└─────────────┘
↓
┌─────────────┐
│ pdf_agent │ → 合成最终 PDF（含清晰中文字体）
└─────────────┘
↓
📄 输出：output/flyer.pdf

🛠️ 技术栈
Python 3.8+
openai / transformers：LLM 文案生成
pandas：历史数据处理
matplotlib：图表生成（已解决中文显示）
diffusers + torch：Stable Diffusion 图像生成
Pillow：图像处理与文字叠加
reportlab：PDF 生成
json + re：自研占位符替换引擎（替代 LangChain）

📦 快速开始
1. 克隆项目

bash
git clone https://github.com/yourname/auto-poster-agent.git 
cd auto-poster-agent

2. 安装依赖

bash
pip install -r requirements.txt

3. 运行项目

bash
python main.py
4. 查看输出

生成的海报将保存在：

output/flyer.pdf
output/poster.png # 主视觉图
output/chart.png # 趋势图

🧪 示例输入

修改 main.py 中的调用：

python
run("为汉服社生成一张国风风格的招新海报")

或尝试：

python
run("为摄影社生成一张小清新风格的招新海报")

⚠️ 已知问题与解决方案

问题 解决方案
------ ----------
Glyph missing from DejaVu Sans 已在 chart_agent.py 中设置 SimHei 字体
SD 生成文字模糊/错乱 已改为“SD 生成背景 + Pillow 叠加文字”
Agent 间数据传递 使用 <your_xxx> 占位符 + replace_placeholder 函数

🧠 设计哲学

本项目最初尝试使用 LLM 作为调度员（LLM as Planner）动态决定流程，但在实践中发现：
决策不稳定
难以调试
参数格式错误频发

因此演进为 固定 workflow + 显式数据传递 的设计，提升了系统的可预测性、稳定性和可维护性。
这是一个从“灵活但不可控”到“简单但可靠”的典型工程演进案例。

📁 项目结构

bash
auto-poster-agent/
├── main.py # 主入口
├── agents/ # 各 Agent 模块
│ ├── llm_agent.py # 文案生成
│ ├── rag_agent.py # 历史数据检索
│ ├── chart_agent.py # 图表生成（解决中文字体）
│ ├── sd_agent.py # 图像生成
│ └── pdf_agent.py # PDF 合成
├── tools.py # 工具函数（如 replace_placeholder）
├── data/ # 历史数据 CSV
├── output/ # 输出目录
├── assets/ # 静态资源（SimHei.ttf）
├── config/ # 配置文件
├── requirements.txt # 依赖库
└── README.md # 本文件

🤝 贡献

欢迎提交 PR 或提出建议！
特别欢迎：
增加新风格模板
添加 Web UI（如 Streamlit）
支持更多输出格式（PPT、PNG）

```





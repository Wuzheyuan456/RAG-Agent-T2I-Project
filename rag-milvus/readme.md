# 🧠 RAG 合规问答助手（Streamlit + Milvus + vLLM）

> 零门槛、流式、带出处、单/多轮可切 —— 让制度文档「秒回」员工提问

---

## 📌 项目简介
- **目标**：把 PDF/Word/TXT 制度文件变成「可对话知识库」，员工自然语言提问 → **1-2 秒流式答案 + 来源高亮**  
- **技术栈**：Streamlit｜Milvus-Lite｜bge-base-zh-v1.5｜vLLM(ChatGLM3-6B)｜LangChain  
- **亮点**：底部固定输入、流式不串字、多轮不爆炸、24G 卡显存安全、单文件一键上传

---

## 🚀 30 秒快速启动
```bash
# 1. 克隆 & 安装
conda create -n rag-milvus python=3.10
pip install -r requirements.txt

# 2. 放模型（任选中文 7B）
wget https://huggingface.co/.../ChatGLM3-6B -O ./model/chatglm3-6b

# 3. 启动 vLLM（OpenAI 兼容）
python -m vllm.entrypoints.openai.api_server \
  --model ./model/chatglm3-6b --dtype float16 --max-model-len 4096 \
  --max-num-seqs 4 --gpu-memory-utilization 0.92 --port 8000

# 4. 启动 Web
streamlit run app.py
```
浏览器自动打开 `http://localhost:8501` → 上传制度文件 → 立即提问。

---

## 📁 项目结构
```
.
├─ app.py                 # Streamlit 主界面（底部固定输入）
├─ app1.py                # 新Streamlit 主界面（底部固定输入）
├─ rag/
│  ├─ retriever.py        # 检索 + 流式生成（单/多轮双接口）
│  ├─ retriever_stream.py        # 流式生成
│  ├─ retriever_response.py        # 响应生成
│  └─  milvus_store.py        # milvusc操作接口
├─ scripts/
│  ├─ index_docs.py      
│  ├─ index_docs_native.py       
│  ├─ offine_eval.py        
│  ├─ search_docs.py       
│  └─ build_with_langchain.py  # 解析 & 入库（可替换为主程序）
├─ data/                  # 文档存放目录（自动创建）
├─ model/                 # 大模型权重目录
└─ requirements.txt
```

---

## ✨ 核心特性
| 功能 | 实现细节 | 状态 |
|---|---|---|
| **底部固定输入** | `st.chat_input` 放在顶级 → 官方吸底 | ✅ |
| **流式不串字** | `temperature=0.3` + 前端直接替换显示 | ✅ |
| **多轮不爆炸** | 只保留最近 3 轮 OpenAI 消息格式 | ✅ |
| **混合检索** | 向量(0.7) + BM25(0.3) Ensemble | ✅ |
| **显存安全** | 24G 卡 → 4096 ctx + 4 concurrent | ✅ |
| **底部目录树** | 默认展开，上传后实时刷新 | ✅ |

---

## 🔧 关键参数调优（24G 卡参考）
```bash
--max-model-len        4096      # 单条最大 token
--max-num-seqs         4         # 并发路数（再涨显存爆炸）
--gpu-memory-utilization 0.92   # 留 1-2G 余量
--temperature          0.3       # 抑制复读
--repetition-penalty   1.05      # 进一步去重
```

---

## 🐛 常见问题 & 解决
| 现象 | 根因 | 快速修复 |
|---|---|---|
| 输入框随历史滚走 | 被包在 `with col2` 内部 | 把 `st.chat_input` 放到 **顶级**（任何 `with` 之外） |
| 答案自我复制 | 前端累加显示 + 温度高 | 前端 `placeholder.markdown(delta)` **直接替换**；`temperature=0.3` |
| Milvus 锁库失败 | 子进程二次连 SQLite | **主程序直接入库**（见下方「去子进程」补丁） |
| metadata 字段缺失 | schema 必填无默认值 | 入库前 `ch.metadata = ch.metadata or {}` |

---

## 🧩 去子进程补丁（推荐）
在 `uploader_fragment()` 里把「子脚本」换成 **主程序直接入库**：
```python
milvus = get_milvus()   # 解开注释即可
for f in uploaded:
    ...
    milvus.add_documents(chunks)   # 不再起子进程
st.success("✅ 已全部入库")
```

---

## 📊 实测效果
- **平均响应**：1.2 s（含检索 + 流式解码）  
- **命中率**：92%（人工抽检 200 条）  
- **上传→可问答**：≤ 30 s（7B 模型，PDF 约 100 页）  
- **显存占用**：≈ 22 GB / 24 GB（安全余量 2 GB）

---

## 🚪 后续 Roadmap
- [ ] AWQ 量化 → 32k 长上下文  
- [ ] 部门级权限过滤（Row-Level Security）  
- [ ] Redis 缓存高频问答  
- [ ] Docker Compose 一键部署

---

> 如果帮到你，给个 ⭐ 鼓励一下 ~

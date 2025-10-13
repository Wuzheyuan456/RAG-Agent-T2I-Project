# ---------- 0. 依赖 ----------
# pip install -U langchain-milvus langchain-community langchain-openai langchain-huggingface

# ---------- 1. 统一用新 import ----------
from langchain_milvus import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI                      # ✅ 0.2+ 无警告
from langchain_huggingface import HuggingFaceEmbeddings     # ✅ 0.2+ 无警告
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from pathlib import Path
import datetime

# ---------- 2. 嵌入模型 ----------
embed_model = HuggingFaceEmbeddings(
    model_name="/home/wzy/pyfile/rag-milvus/model/bge-base-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

# ---------- 3. 连接已有向量库 ----------
vector_store = Milvus(
    embedding_function=embed_model,
    collection_name="policy_chunks",
    connection_args={"uri": "milvus_demo.db"},
    index_params={"index_type": "FLAT", "metric_type": "IP"}
)

# ---------- 4. 内存 BM25 ----------
cur = vector_store.col.query(expr="", output_fields=["text"], limit=16_380)
docs = [Document(page_content=x["text"], metadata={}) for x in cur]
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 1

# ---------- 5. 混合检索 ----------
from langchain.retrievers import EnsembleRetriever
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 1})
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# ---------- 6. 本地 LLM ----------
llm = ChatOpenAI(
    openai_api_base="http://127.0.0.1:8000/v1",
    model_name="chatglm3-6b",
    temperature=0.5,        # ↑ 原来 0.1 太保守
    max_tokens=512,
    openai_api_key="dummy",
    streaming=True       # 调试用同步
)

# ---------- 7. Prompt ----------
system_prompt = (
    "你是公司合规助手，只能使用【已知信息】回答，禁止推测。\n"
    "【已知信息】{context}\n【问题】{question}\n"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# ---------- 8. 格式化文档 + 截断 ----------
def format_docs(docs: list[Document], max_tokens: int = 1500) -> str:
    # 简易字符截断，1 汉字≈1 token
    text = "\n\n".join(d.page_content for d in docs)
    print("【实际传入上下文字符数】", len(text))
    return text[: int(max_tokens * 1.2)]

# ---------- 9. 构建 Runnable ----------
# ---------- 打印节点 ----------
def print_inputs(data: dict) -> dict:
    """只打印，不改变数据流"""
    print("【实际上传 prompt 字典】", data)
    return data

def print_prompt(msg):
    print("【最终 prompt】", msg[0].content)  # system 消息
    return msg

chain = (
    RunnableParallel(          # 第一层：检索
        context=lambda x: format_docs(ensemble_retriever.invoke(x["question"])),
        sources=lambda x: ensemble_retriever.invoke(x["question"]),
        question=lambda x: x["question"]
    )
    | print_inputs             # <- 在这里插入打印
    | RunnableParallel(
        answer=prompt | llm,
        sources=lambda x: x["sources"]
    )
)

# chain = (
#     RunnableParallel(  # 1. 先拿到 docs + 问题
#         context=lambda x: format_docs(ensemble_retriever.invoke(x["question"])),
#         sources=lambda x: ensemble_retriever.invoke(x["question"]),
#         question=lambda x: x["question"]
#     )
#     | RunnableParallel(  # 2. 再同时生成答案 & 保留来源
#         answer=prompt | llm,
#         sources=lambda x: x["sources"]  # 原样带下去
#     )
# )

import asyncio

async def ask_stream(question: str):
    # 1. 同步检索（快）
    docs   = ensemble_retriever.invoke(question)
    context= format_docs(docs)
    sources= {Path(d.metadata.get("metadata", {}).get("source", "")).name
              for d in docs if d.metadata.get("metadata", {}).get("source")}

    # 2. 只流生成部分
    gen_chain = prompt | llm

    # 3. 流式打印
    print("【答案】", end="", flush=True)
    async for chunk in gen_chain.astream({"context": context, "question": question}):
        print(chunk.content, end="", flush=True)
    print()  # 换行

    # 4. 一次性给出来源
    print("【来源】", sorted(sources) if sources else "无文件来源")



# ---------- 流式问答接口 ----------
import asyncio, time
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path

# 复用前面已有的组件（ensemble_retriever / llm / prompt 等）
_stream_chain = (
    RunnablePassthrough()
    | (lambda x: format_docs(ensemble_retriever.invoke(x["question"])))
    | prompt
    | llm
)

async def astream_answer(question: str):
    """逐 token 生成 + 返回来源"""
    docs   = ensemble_retriever.invoke(question)
    context = format_docs(docs)
    sources = {Path(d.metadata.get("metadata", {}).get("source", "")).name
               for d in docs if d.metadata.get("metadata", {}).get("source")}
    # 流式生成
    async for chunk in _stream_chain.astream({"context": context, "question": question}):
        yield chunk.content, sources      # 逐个 token + 来源集合

# ---------- 10. 调用 + 计时 ----------
if __name__ == "__main__":


    q = "一般事故隐患治理怎么做？"
    print("① 开始检索", datetime.datetime.now())
    asyncio.run(ask_stream(q))
    print("② 生成完成", datetime.datetime.now())
    # q = "一般事故隐患治理怎么做？"
    # print("① 开始检索", datetime.datetime.now())
    # out = chain.invoke({"question": q})
    # print("② 生成完成", datetime.datetime.now())
    #
    # # 字典取值
    # print("【答案】", out)
    # print("【答案】", out["answer"].content)
    # print("【来源】")
    # for doc in out["sources"]:
    #     # 先取内层 metadata 字典
    #     inner_meta = doc.metadata.get("metadata", {})
    #     file_path = inner_meta.get("source")
    #     if file_path:
    #         print(Path(file_path).name)
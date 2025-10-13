# retriever.py
from pathlib import Path
from langchain_milvus import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
import asyncio

# ==== 组件工厂（延迟初始化，无异步）====
def build_components():
    embed_model = HuggingFaceEmbeddings(
        model_name="/home/wzy/pyfile/rag-milvus/model/bge-base-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = Milvus(
        embedding_function=embed_model,
        collection_name="policy_chunks",
        connection_args={"uri": "milvus_demo.db"},
        index_params={"index_type": "FLAT", "metric_type": "IP"}
    )
    cur = vector_store.col.query(expr="", output_fields=["text"], limit=16_380)
    docs = [Document(page_content=x["text"], metadata={}) for x in cur]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 1
    from langchain.retrievers import EnsembleRetriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.7, 0.3]
    )

    llm = ChatOpenAI(
        openai_api_base="http://127.0.0.1:8000/v1",
        model_name="chatglm3-6b",
        temperature=0.5,
        max_tokens=512,
        openai_api_key="dummy"
    )
    system_prompt = (
        "你是公司合规助手，只能使用【已知信息】回答，禁止推测。\n"
        "【已知信息】{context}\n"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    def format_docs(docs, max_tokens=1500):
        text = "\n\n".join(d.page_content for d in docs)
        return text[: int(max_tokens * 1.2)]

    return ensemble_retriever, llm, prompt, format_docs


# ==== 流式接口 ====
async def astream_answer(question: str):
    ensemble_retriever, llm, prompt, format_docs = build_components()
    docs = ensemble_retriever.invoke(question)
    context = format_docs(docs)
    sources = {Path(d.metadata.get("metadata", {}).get("source", "")).name
               for d in docs if d.metadata.get("metadata", {}).get("source")}
    chain = prompt | llm
    async for chunk in chain.astream({"context": context, "question": question}):
        yield chunk.content, sources


# 新增：单轮问答，不带任何历史
async def astream_simple(question: str):
    """只根据当前 question 做检索 + 生成，不拼接历史"""
    from langchain_core.documents import Document

    # 1. 同样初始化组件（可缓存加速）
    ensemble_retriever, llm, prompt, format_docs = build_components()

    # 2. 检索
    docs = ensemble_retriever.invoke(question)
    context = format_docs(docs)

    # 3. 构造一次性 prompt（无历史）
    system_prompt = (
        "你是公司合规助手，只能使用【已知信息】回答，禁止推测。\n"
        "【已知信息】{context}\n【问题】{question}\n"
    )
    prompt_val = system_prompt.format(context=context, question=question)

    # 4. 流式生成
    sources = {Path(d.metadata.get("metadata", {}).get("source", "")).name
               for d in docs if d.metadata.get("metadata", {}).get("source")}
    async for chunk in llm.astream(prompt_val):
        yield chunk.content, sources

# 只保留最近 3 轮，可调
KEEP_ROUNDS = 3

async def astream_answer_his(history: list[tuple[str, str]], question: str):
    """ history: [(user1, assistant1), (user2, assistant2), ...] """


    # --- 1. 检索 ---
    ensemble_retriever, llm, prompt_tmpl, format_docs = build_components()
    docs = ensemble_retriever.invoke(question)
    context = format_docs(docs)
    sources = {Path(d.metadata.get("metadata", {}).get("source", "")).name
               for d in docs if d.metadata.get("metadata", {}).get("source")}

    # --- 2. 构造多轮消息 ---
    messages = [("system", prompt_tmpl.messages[0].prompt.format(context=context))]
    # 只取最近 KEEP_ROUNDS 轮
    for q, a in history[-KEEP_ROUNDS:]:
        messages.extend([("human", q), ("assistant", a)])
    messages.append(("human", question))

    chain = ChatPromptTemplate.from_messages(messages) | llm

    # --- 3. 流式返回 ---
    async for chunk in chain.astream({}):
        yield chunk.content, sources
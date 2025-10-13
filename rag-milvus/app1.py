import streamlit as st
# ---------- 页面配置 ----------
st.set_page_config(page_title="RAG 小助手", layout="wide")
from pathlib import Path
import asyncio
from rag.retriever import astream_simple,astream_answer_his
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus



# ---------- 路径 ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# # ---------- 只建一次 embedding & milvus ----------
# @st.cache_resource
# def get_milvus():
#     embed = HuggingFaceEmbeddings(
#         model_name="/home/wzy/pyfile/rag-milvus/model/bge-base-zh-v1.5",
#         encode_kwargs={"normalize_embeddings": True}
#     )
#     return Milvus(
#         embedding_function=embed,
#         collection_name="policy_chunks",
#         connection_args={"uri": "milvus_demo.db"},
#         index_params={"index_type": "FLAT", "metric_type": "IP"},
#
#     )
#
# milvus = get_milvus()


# ---------- 左侧：目录树 ----------


# ---------- 文件上传 ----------
def load_single_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in (".docx", ".doc"):
        return Docx2txtLoader(str(path)).load()
    return TextLoader(str(path), encoding="utf-8").load()

def uploader_fragment():
    st.markdown("### 📤 文件上传")
    uploaded = st.file_uploader(
        "支持 pdf/docx/txt/md（可多选）",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True
    )
    if st.button("加载到知识库", type="primary"):
        if not uploaded:
            st.warning("请先上传文件")
        else:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            for f in uploaded:
                save_path = data_dir / f.name
                with open(save_path, "wb") as tmp:
                    tmp.write(f.getbuffer())
            with st.spinner("解析+向量化中，请稍候..."):
                import subprocess
                result = subprocess.run(
                    ["python", "scripts/build_with_langchain.py"],
                    capture_output=True,
                    text=True
                )
            if result.returncode != 0:
                st.error("脚本运行失败，详情如下：")
                st.code(result.stdout + result.stderr, language="bash")
            else:
                st.success("✅ 已全部入库，可以提问了！")
    # if st.button("加载到知识库", type="primary"):
    #     if not uploaded:
    #         st.error("请先上传文件"); return
    #     prog = st.progress(0)
    #     for idx, file in enumerate(uploaded, 1):
    #         save_path = DATA_DIR / file.name
    #         save_path.write_bytes(file.read())
    #         docs = load_single_file(save_path)
    #         chunks = CharacterTextSplitter(chunk_size=512, chunk_overlap=64).split_documents(docs)
    #         # ✅ 修复：强制补 metadata 字段
    #         for ch in chunks:
    #             ch.metadata = ch.metadata or {}
    #             ch.metadata.setdefault("source", "")
    #
    #         milvus.add_documents(chunks)
    #         prog.progress(idx / len(uploaded))
    #         st.text(f"✅ {file.name}  共 {len(chunks)} 段")
    #     st.success("全部加载完成！")
    #     st.rerun()   # 立即刷新左侧目录树

# ---------- 连续对话 ----------
async def generate_answer(question: str):
    sources = set()
    async for delta, src in astream_simple(question):
        sources |= src
        yield delta, sources     # 直接给最新完整字符串

def main_chat():
    st.markdown("### 💬 连续对话")
    if "history" not in st.session_state:
        st.session_state.history = []
        # --- 展示历史（仅前端显示，不再拼到 prompt） ---
    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    # 2. 底部固定输入框
    with st.container():
        # 让输入框始终贴在底部
        question = st.chat_input("请输入问题")
    if question :
        # 1. 前端立即显示用户消息
        st.session_state.history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        # 2. ***只把当前问题送给模型***（无历史拼接）
        full_prompt = question

        # 3. 流式生成回答
        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer, sources = "", set()

            def sync_gen():
                loop = asyncio.new_event_loop()
                gen = astream_answer_his(st.session_state.history, question)
                #gen = generate_answer(full_prompt)

                while True:
                    try:
                        delta, src = loop.run_until_complete(gen.__anext__())
                        yield delta, src
                    except StopAsyncIteration:
                        break

            for delta, src in sync_gen():
                #answer = delta
                answer += delta
                sources |= src
                placeholder.markdown(answer)

            if sources:
                answer += f"\n\n📎 来源：{', '.join(sources)}"
                placeholder.markdown(answer)

        # 4. 把回答记入前端历史（仅显示，不用于下一轮 prompt）
        st.session_state.history.append(("assistant", answer))


# ---------- 布局 ----------
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("### 📁 data 目录")
    if DATA_DIR.exists():
        # 根目录文件默认展开
        root_files = [f for f in DATA_DIR.iterdir() if f.is_file()]
        if root_files:
            with st.expander("📄 根目录文件", expanded=True):
                for f in sorted(root_files):
                    st.text(f.name)

        # 子文件夹默认展开
        for d in sorted(DATA_DIR.iterdir()):
            if d.is_dir():
                with st.expander(f"📂 {d.name}", expanded=True):
                    for f in sorted(d.iterdir()):
                        if f.is_file():
                            st.text(f.name)
    else:
        st.warning("目录 ./data 不存在")

    st.divider()
    uploader_fragment()
with col2:
    main_chat()
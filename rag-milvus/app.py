import streamlit as st, asyncio, time
from pathlib import Path
from rag.retriever import astream_answer   # ① 导入流式接口

st.set_page_config(page_title="公司内规问答", layout="wide")
st.title("📚 公司内规问答（RAG + Milvus）")

# -------------- 文件上传 --------------
uploaded = st.file_uploader(
    "上传 PDF / DOCX / 网页 HTML / Markdown",
    type=["pdf", "docx", "html", "md"],
    accept_multiple_files=True
)
if st.button("📥 开始解析并入库"):
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
            subprocess.run(["python", "scripts/build_with_langchain.py"], check=True)
        st.success("✅ 已全部入库，可以提问了！")

# -------------- 提问 & 流式回答 --------------
question = st.text_input("请输入问题：")
if st.button("🔍 流式查询"):
    if not question:
        st.stop()
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        msg = st.empty()
        full = ""
        sources = set()

        # 把异步流转成同步生成器
        def sync_gen():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            gen = astream_answer(question)
            try:
                while True:
                    token, src = loop.run_until_complete(gen.__anext__())
                    yield token, src
            except StopAsyncIteration:
                pass

        for tok, src in sync_gen():
            full += tok
            sources.update(src)
            msg.markdown(full + "▌")
        msg.markdown(full)

    if sources:
        with st.expander("📎 查看引用文件"):
            st.write(sorted(sources))
    else:
        st.info("未定位到文件来源")
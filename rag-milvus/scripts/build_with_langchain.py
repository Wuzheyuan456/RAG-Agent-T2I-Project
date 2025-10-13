
from langchain_milvus import Milvus as MilvusVectorStore


import hashlib
from pathlib import Path
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent.parent / "data"
from pathlib import Path
URI = f"{Path(__file__).resolve().parent.parent / 'milvus_demo.db'}"
# URI      = "milvus_demo.db"
COLLECTION = "policy_chunks"


# ----------- 1. 按扩展名选 Loader -----------
def loader_factory(file_path: str):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(file_path)
    # 需要更多格式继续 elif 即可
    else:  # txt、md、py、json...
        return TextLoader(file_path, autodetect_encoding=True)
loader = DirectoryLoader(
    str(DATA_DIR),
    glob="**/*",
    loader_cls=loader_factory,  # 关键：用自定义 loader
    use_multithreading=True,
)

docs_raw = loader.load()

# ----------- 2. 分割 -----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=50
)
docs = text_splitter.split_documents(docs_raw)

# ----------- 3. 文件 sha 去重 metadata -----------
for doc in docs:
    path = doc.metadata.get("source")
    if path:
        doc.metadata["file_sha"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()

# ----------- 4. 嵌入 & 建库 -----------
embed_model = HuggingFaceEmbeddings(
    model_name="/home/wzy/pyfile/rag-milvus/model/bge-base-zh-v1.5"
)

vector_store = MilvusVectorStore.from_documents(
    documents=docs,
    embedding=embed_model,
    collection_name=COLLECTION,
    connection_args={"uri": URI},
    vector_field="vector",
    text_field="text",
    metadata_field="metadata",
    index_params={"index_type": "FLAT", "metric_type": "IP"},
    auto_id=True,
)

print("✅ LangChain 建库完成，chunk 数：", len(docs))
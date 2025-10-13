import os, sys, hashlib, json
from pathlib import Path
from llama_index.core import (
    Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from pymilvus import connections, utility
connections.connect(uri="milvus_demo.db")
if utility.has_collection("policy_chunks"):
    utility.drop_collection("policy_chunks")
    print("🗑️  已删除旧集合（metric=IP），准备重建 COSINE 版")

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "data"

Settings.embed_model = HuggingFaceEmbedding(r"/home/wzy/pyfile/rag-milvus/model/bge-base-zh-v1.5")
Settings.chunk_size, Settings.chunk_overlap = 512, 50

vector_store = MilvusVectorStore(
    uri="milvus_demo.db",
    collection_name="policy_chunks",
    dim=768,
    index_type="FLAT",
    metric_type="IP",         # 与搜索一致
    enable_dynamic=False,     # ✅ 关键：不要动态字段
    text_key="text",
    metadata_key="metadata",
    vector_field="vector"  # ✅ 关键：统一字段名

)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def file_sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load_or_build():
    all_files = list(DATA_DIR.rglob("*"))
    if not all_files:
        print("❌ data/ 目录为空"); return
    docs = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
    # 把文件 sha256 写进 metadata，用于去重
    for doc in docs:
        f_path = doc.metadata.get("file_path")
        if f_path: doc.metadata["file_sha"] = file_sha(Path(f_path))
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    print("✅ 索引完成，chunk 数：", len(docs))

if __name__ == "__main__":
    load_or_build()
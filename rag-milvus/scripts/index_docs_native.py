import json, uuid, sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

# ---------- 基本配置 ----------
DB_FILE      = Path("milvus_demo.db").resolve()
SCRIPT_DIR   = Path(__file__).parent
DATA_DIR     = SCRIPT_DIR.parent / "data"
COLLECTION   = "policy_chunks"
DIM          = 768
METRIC       = "IP"                    # 与搜索侧保持一致
INDEX_TYPE   = "FLAT"                  # Lite 只支持 FLAT / IVF_FLAT / AUTOINDEX
MODEL_NAME   = r"/home/wzy/pyfile/rag-milvus/model/bge-base-zh-v1.5"

# ---------- 连接 ----------
connections.connect(alias="default", uri=str(DB_FILE))

# ---------- 1. 显式建表（全部带 max_length） ----------
def create_collection():
    if utility.has_collection(COLLECTION):
        utility.drop_collection(COLLECTION)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65_535),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65_535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, description="explicit schema demo")
    collection = Collection(COLLECTION, schema)

    # 建 FLAT 索引
    index_params = {
        "metric_type": METRIC,
        "index_type": INDEX_TYPE,
    }
    collection.create_index("embedding", index_params)
    collection.load()
    return collection

# ---------- 2. 文档 → 向量 → 写入 ----------
def build_index(collection: Collection):
    if not DATA_DIR.exists() or not any(DATA_DIR.rglob("*")):
        print("❌ data/ 为空"); sys.exit()

    # 1. 读取 + 分块（chunk 变小，保证字符 ≤ 65535）
    from llama_index.core import SimpleDirectoryReader, Settings
    Settings.chunk_size, Settings.chunk_overlap = 200, 30   # token≈200，字符≪65535
    docs = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()

    # 2. 向量化
    encoder = SentenceTransformer(MODEL_NAME)
    ids, texts, metas, embs = [], [], [], []
    for doc in docs:
        ids.append(str(uuid.uuid4()))
        texts.append(doc.text or "")
        metas.append(json.dumps(doc.metadata, ensure_ascii=False))
        embs.append(encoder.encode(doc.text, normalize_embeddings=True).tolist())

    # 3. 批量插入
    collection.insert([ids, texts, metas, embs])
    collection.flush()
    print(f"✅ 索引完成，chunk 数：{len(docs)}")

# ---------- 3. 原生查询 ----------
def search_demo(collection: Collection, query: str, top_k: int = 5):
    encoder = SentenceTransformer(MODEL_NAME)
    emb = encoder.encode(query, normalize_embeddings=True).tolist()

    res = collection.search(
        data=[emb],
        anns_field="embedding",
        param={"metric_type": METRIC},
        limit=top_k,
        output_fields=["text", "metadata"]
    )

    print(f"【问题】{query}\n【Top-{top_k} 结果】")
    for rank, h in enumerate(res[0], 1):
        print(f"{rank}. score={h.score:.4f}")
        print(f"   text={h.entity.get('text')[:200]}...")
        print(f"   meta={json.loads(h.entity.get('metadata') or '{}')}\n")

# ---------- 4. 一键执行 ----------
if __name__ == "__main__":
    col = create_collection()
    build_index(col)
    search_demo(col, "常用危险源辨识及风险评估方法")
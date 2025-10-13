import sys
from pathlib import Path
import sentence_transformers
from pymilvus import connections, Collection
from pymilvus import utility
SCRIPT_DIR = Path(__file__).parent

# ---------------- 配置 ----------------
DB_FILE      = SCRIPT_DIR / "milvus_demo.db"   #  Llama-index 建好的 Lite 文件
COLLECTION   = "policy_chunks"                #  建索引时起的名字
TOP_K        = 5
MODEL_NAME   = r"/home/wzy/pyfile/rag-milvus/model/bge-base-zh-v1.5"  # 同一套模型

# ---------------- 连接 ----------------
# Milvus-Lite 用文件协议
connections.connect(alias="default", uri=f"{DB_FILE.resolve()}")

# ---------------- 加载集合 ----------------
if not utility.has_collection(COLLECTION):
    print("❌ 集合不存在，先跑 index_docs.py 建索引"); sys.exit()
col = Collection(COLLECTION)
print(col.schema)
col.load()

# ---------------- 向量化 & 检索 ----------------
def search(query: str):
    encoder = sentence_transformers.SentenceTransformer(MODEL_NAME)
    emb = encoder.encode(query, normalize_embeddings=True).tolist()

    res = col.search(
        data=[emb],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=TOP_K,
        output_fields=["text", "metadata"]
    )

    for i, hits in enumerate(res):
        print(f"【问题】{query}\n【Top-{TOP_K} 结果】")
        for rank, h in enumerate(hits, 1):
            meta = h.entity.get("metadata") or "{}"
            print(f"{rank}. score={h.score:.4f}")
            print(f"   text={h.entity.get('text')}")
            print(f"   meta={meta}\n")

# ---------------- CLI ----------------
if __name__ == "__main__":
    q = input("请输入问题：").strip()
    if q:
        search(q)
    else:
        # 给个默认
        search("公司员工请假流程")
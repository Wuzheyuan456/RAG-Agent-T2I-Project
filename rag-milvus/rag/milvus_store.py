"""
轻量级 Milvus 封装
依赖：pymilvus>=2.4
"""
import uuid
from typing import List, Dict, Optional
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility

class MilvusStore:
    """只管「增 / 删 / 查」向量，其余不管"""

    def __init__(self,
                 uri: str = "http://localhost:19530",
                 collection: str = "policy_chunks",
                 dim: int = 768,
                 metric: str = "COSINE",
                 index_type: str = "HNSW",
                 M: int = 32,
                 efConstruction: int = 256):
        connections.connect(alias="default", uri=uri)
        self.collection_name = collection
        self.dim = dim
        self._ensure_collection(metric, index_type, M, efConstruction)

    # ---------- 内部 ----------
    def _ensure_collection(self, metric: str, index_type: str, M: int, efConstruction: int):
        if utility.has_collection(self.collection_name):
            self.col = Collection(self.collection_name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, description="公司内规 chunks")
        self.col = Collection(self.collection_name, schema)

        # 建 HNSW 索引
        index_params = {
            "metric_type": metric,
            "index_type": index_type,
            "params": {"M": M, "efConstruction": efConstruction}
        }
        self.col.create_index("embedding", index_params)
        self.col.load()

    # ---------- 增 ----------
    def add(self,
            texts: List[str],
            embeddings: List[List[float]],
            metadatas: Optional[List[Dict]] = None) -> List[str]:
        """返回写入的 id 列表"""
        ids = [str(uuid.uuid4()) for _ in texts]
        meta_strs = [""] * len(texts) if metadatas is None else [json.dumps(m, ensure_ascii=False) for m in metadatas]
        self.col.insert([ids, texts, meta_strs, embeddings])
        self.col.flush()          # 实时可见
        return ids

    # ---------- 删 ----------
    def delete(self, ids: List[str]):
        expr = f"id in [{','.join(repr(i) for i in ids)}]"
        self.col.delete(expr)

    # ---------- 查 ----------
    def search(self,
               embedding: List[float],
               top_k: int = 5,
               ef: int = 128,
               output_fields: Optional[List[str]] = None) -> List[Dict]:
        """返回 List[Dict] 含 text / metadata / distance"""
        if output_fields is None:
            output_fields = ["text", "metadata"]
        search_params = {"metric_type": "COSINE", "params": {"ef": ef}}
        res = self.col.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )
        hits = []
        for hits_i in res:          # 一次查询可能多组，这里只有 1 组
            for h in hits_i:
                hits.append({
                    "id": h.id,
                    "text": h.entity.get("text"),
                    "metadata": json.loads(h.entity.get("metadata") or "{}"),
                    "distance": h.score
                })
        return hits

    # ---------- 统计 ----------
    def count(self) -> int:
        return self.col.num_entities

if __name__ == "__main__":
    import random, json
    store = MilvusStore()
    texts = ["内规第1条", "内规第2条"]
    embs = [[random.random() for _ in range(768)] for _ in texts]
    ids = store.add(texts, embs)
    print("写入 id:", ids)
    print("查询结果:", store.search(embs[0], top_k=2))
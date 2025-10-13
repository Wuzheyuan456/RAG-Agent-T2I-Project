from datasets import Dataset
from rag.retriever import ensemble_retriever

golden = Dataset.from_json("eval/golden.jsonl")  # 字段：question, golden_chunk_id
k = 5
hits = 0
for row in golden:
    docs = ensemble_retriever.get_relevant_documents(row["question"])
    if row["golden_chunk_id"] in {d.metadata.get("chunk_id") for d in docs[:k]}:
        hits += 1
recall = hits / len(golden)
print(f"Recall@{k} = {recall:.3f}")
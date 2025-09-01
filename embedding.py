import json
import faiss
from sentence_transformers import SentenceTransformer
with open("cleaned_rag_dataset.json","r",encoding="utf-8")as f:
    data=json.load(f)
texts=[item["text"] for item in data]
metadatas=[item["metadata"] for item in data]
ids=[item["id"] for item in data]
model=SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings=model.encode(texts,show_progress_bar=True,convert_to_numpy=True)
dimension=embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index,"dmex_index.faiss")
metadata_index = [{"id": i, "metadata": meta, "text": text} for i, meta, text in zip(ids, metadatas, texts)]
with open("dmex_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata_index, f, indent=2, ensure_ascii=False)
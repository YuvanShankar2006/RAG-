import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_resources():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    index = faiss.read_index("dmex_index.faiss")
    with open("dmex_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, index, metadata
def query_rag_system(user_query, model, index, metadata, top_k=3):
    try:
        query_vector = model.encode([user_query], normalize_embeddings=True, convert_to_numpy=True)
        distances, indices = index.search(query_vector, top_k)
        if len(indices) == 0 or indices[0][0] == -1:
            return [{"error": "No results found", "details": "Try different keywords"}]
        results = []
        for idx, score in zip(indices[0], distances[0]):
            result = metadata[idx].copy()
            result["score"] = float(1 - score) if index.metric_type == faiss.METRIC_INNER_PRODUCT else float(score)
            results.append(result)
        return results
    except Exception as e:
        return [{"error": "Query failed", "details": str(e)}]
st.set_page_config(page_title="DMEX RAG Assistant", layout="wide")
st.title("📚 DMEX RAG Assistant")
st.write("Ask anything about DMEX system documentation")
model, index, metadata = load_resources()
query = st.text_input("🔍 Enter your question:")

if query:
    results = query_rag_system(query, model, index, metadata)
    
    if "error" in results[0]:
        st.error(f"{results[0]['error']}: {results[0]['details']}")
    else:
        for i, res in enumerate(results, 1):
            with st.expander(f"Result {i} (Score: {res['score']:.3f})"):
                st.markdown(f"**Section:** {res['metadata'].get('section', 'N/A')}")
                st.markdown(f"**Source:** {res['metadata'].get('source', 'N/A')}")
                st.markdown(f"**Text:** {res['text']}")


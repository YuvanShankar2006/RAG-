import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from typing import Tuple, List, Dict, Optional

# === Constants ===
MIN_QUERY_LENGTH = 3
COMMON_WORDS = {
    "a", "an", "the",
    "in", "on", "at", "by", "to", "of", "for", "with", "from", "about", "over", "under", "through", "between", "into",
    "and", "but", "or", "yet", "so", "if", "while", "although", "because", "since", "unless", "though",
    "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has", "had",
    "can", "could", "shall", "should", "will", "would", "may", "might", "must", "ought",
    "some", "any", "each", "every", "no", "none", "much", "many", "few", "little", "several", "both", "all",
    "who", "whom", "whose", "which", "what", "where", "when", "why", "how"
}
GIBBERISH_PATTERNS = [
    r"(.)\1{2,}",  
    r"[^a-zA-Z0-9\s]", 
    r"\b\d+\b",
]
@st.cache_resource(show_spinner=False)
def load_resources() -> Tuple:
    with st.spinner("Loading models..."):
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        index = faiss.read_index("dmex_index.faiss")
        with open("dmex_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return model, index, metadata, tokenizer, llm
def validate_query(query: str) -> Tuple[bool, Optional[str]]:
    query = query.strip().lower()
    if len(query) < MIN_QUERY_LENGTH:
        return False, "Query too short. Please ask a complete question."
    if query in COMMON_WORDS:
        return False, f"'{query}' is too vague. Try something like 'How to add a GRN?'"
    if not re.search(r"[a-zA-Z]{3,}", query): 
        return False, "Please use proper words in your question."
    
    for pattern in GIBBERISH_PATTERNS:
        if re.fullmatch(pattern, query):
            return False, "Your query appears to contain random characters."
    
    return True, None
def query_rag_system(
    user_query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: List[Dict],
    top_k: int = 3
) -> Tuple[List[Dict], Optional[str]]:
    """Search with error handling and score normalization"""
    try:
        query_vector = model.encode(
            [user_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        distances, indices = index.search(query_vector, top_k)
        
        if len(indices) == 0 or indices[0][0] == -1:
            return [], "No relevant results found"
        
        return [
            {
                **metadata[idx],
                "score": float(1 - score) if index.metric_type == faiss.METRIC_INNER_PRODUCT 
                       else float(score)
            }
            for idx, score in zip(indices[0], distances[0])
        ], None
    
    except Exception as e:
        return [], f"Search error: {str(e)}"
def generate_answer(
    context: str,
    question: str,
    tokenizer: AutoTokenizer,
    llm: AutoModelForSeq2SeqLM,
    max_length: int = 512
) -> str:
    prompt = (
    "You are a technical support assistant for DMeX Pro, an industrial equipment management system. "
    "Provide a concise, step-by-step answer using ONLY the provided context. "
    "If the question is unclear or missing information, ask for clarification.\n\n"
    "**Rules:**\n"
    "1. Be specific to DMeX Pro features\n"
    "2. Use bullet points for procedures\n"
    "3. Include menu paths when available (e.g., 'Go to Inventory > GRN > Add New')\n"
    "4. Reference error codes if mentioned\n"
    "5. Never invent features - say 'I don't know' if unsure\n\n"
    "**Context:**\n"
    f"{context[:3000]}\n\n"
    "**Question:**\n"
    f"{question}\n\n"
    "**Answer Format:**\n"
    "[Summary] 1-2 sentence overview\n"
    "[Steps] Numbered list if procedural\n"
    "[Location] Where to find in DMeX Pro\n"
    "[Notes] Any warnings or requirements"
)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    
    outputs = llm.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=False   
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
def get_response(query: str, resources: Tuple) -> Dict:
    """
    Main processing pipeline that handles all cases:
    - Invalid queries
    - No results
    - Successful answers
    """
    is_valid, error_msg = validate_query(query)
    if not is_valid:
        return {"type": "error", "content": error_msg}
    
    model, index, metadata, tokenizer, llm = resources
    results, search_error = query_rag_system(query, model, index, metadata)
    
    if search_error:
        return {"type": "error", "content": search_error}
    elif not results:
        return {
            "type": "no_results",
            "content": "No matching documentation found. Try different keywords."
        }
    
    context = " ".join([res["text"] for res in results[:2]])
    answer = generate_answer(context, query, tokenizer, llm)
    
    return {
        "type": "success",
        "answer": answer,
        "results": results
    }
def main():
    st.set_page_config(page_title="DMeX Pro Chat Assistant", layout="wide")
    st.title("🤖 DMeX Pro Smart Assistant")
    st.markdown("Get instant help with DMeX Pro documentation")
    resources = load_resources()
    query = st.chat_input("Ask a question about DMeX Pro...")
    if query:
        with st.spinner("Analyzing your question..."):
            response = get_response(query, resources)
        if response["type"] == "error":
            st.error(response["content"])
        elif response["type"] == "no_results":
            st.warning(response["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            with st.expander("📚 Source References"):
                for i, res in enumerate(response["results"], 1):
                    st.markdown(f"**{res['metadata'].get('section', 'General')}**")
                    st.caption(f"Relevance: {res['score']:.2f}")
                    st.markdown(res["text"][:500] + ("..." if len(res["text"]) > 500 else ""))

if __name__ == "__main__":
    main()
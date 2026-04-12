import os, re
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_model = None
_index = None

def _get_resources():
    global _model, _index
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = pc.Index("mit-bengaluru")
    return _model, _index

def detect_category(query: str) -> str | None:
    """Detect query intent to filter by category."""
    q = query.lower()
    
    if any(w in q for w in ["course", "program", "btech", "mtech", "degree",
                              "admission", "eligibility", "curriculum", "specialization"]):
        return "programs"
    
    if any(w in q for w in ["hostel", "accommodation", "room", "mess", "dining",
                              "stay", "dormitory", "warden"]):
        return "hostel"
    
    if any(w in q for w in ["fee", "scholarship", "loan", "tuition", "cost",
                              "payment", "financial"]):
        return "admissions"
    
    if any(w in q for w in ["faculty", "professor", "teacher", "who teaches",
                              "hod", "head of department", "staff"]):
        return "faculty"
    
    if any(w in q for w in ["event", "news", "conference", "workshop",
                              "seminar", "calendar"]):
        return "news"
    
    return None  # No filter — broad semantic search

def search_campus_data(user_query: str, top_k: int = 3) -> str:
    model, index = _get_resources()
    
    query_embedding = model.encode([user_query]).tolist()[0]
    category = detect_category(user_query)
    
    # Build filter if category detected
    filter_dict = {"category": {"$eq": category}} if category else None
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )
    
    # Fallback: if filtered search returns nothing, retry without filter
    if not results["matches"] and filter_dict:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
    
    if not results["matches"]:
        return "No relevant information found in the MIT Bengaluru knowledge base."
    
    context_parts = []
    for match in results["matches"]:
        relevance = round(match["score"] * 100, 1)
        text = match["metadata"].get("text", "")
        source = match["metadata"].get("source_url", "")
        cat = match["metadata"].get("category", "")
        context_parts.append(
            f"[Category: {cat} | Relevance: {relevance}%]\n{text}"
        )
    
    return "\n\n---\n\n".join(context_parts)

if __name__ == "__main__":
    test_queries = [
        "What courses does MIT Bengaluru offer?",
        "Who are the faculty members in the CS department?",
        "What are the hostel facilities?",
        "What is the fee structure?",
        "Tell me about the BTech Computer Science program",
    ]
    for q in test_queries:
        cat = detect_category(q)
        print(f"\nQuery: {q}")
        print(f"Detected category filter: {cat}")
        print(search_campus_data(q))
        print("=" * 60)
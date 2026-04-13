
"""
search.py — MIT Bengaluru RAG retrieval
========================================
CRITICAL FIXES vs old version:
  1. OLD used SentenceTransformer all-MiniLM-L6-v2 (384d) + index "mit-bengaluru"
     NEW uses Pinecone hosted llama-text-embed-v2 (1024d) + index "mit-bengaluru-v2"
     → MUST match embed_and_store.py exactly or every query returns wrong results
  2. detect_query_intent now returns 3 values: (category, subtype, dept_hint)
  3. HOD / "head of department" queries now handled (were returning None before)
  4. "who is X" name queries use token-matching fallback, not cosine similarity
  5. Reranking: table +0.30, list +0.25, profile +0.10, overview -0.20
  6. Faculty list: pre-built chunk → synthesise from profiles → fallback to profiles
  7. Fee queries force table subtype retrieval with number-pattern boost
  8. Fetch top_k=10, rerank, return best 3

Run: python search.py
"""

import os, re
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

DEBUG = True   # set False in production

# ── Index + embedding config (MUST match embed_and_store.py) ─────────────────
INDEX_NAME   = "mit-bengaluru-v2"
EMBED_MODEL  = "llama-text-embed-v2"
NAMESPACE    = "default"

_pc    = None
_index = None

def _get_resources():
    global _pc, _index
    if _pc is None:
        _pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = _pc.Index(INDEX_NAME)
        print(f"  [INIT] Connected to Pinecone index '{INDEX_NAME}'")
    return _pc, _index


# ── Embedding (Pinecone hosted inference — matches embed_and_store.py) ────────
EMBED_PREFIXES = {
    ("hostel",     "table"):    "Hostel fee structure room type charges: ",
    ("hostel",     "list"):     "Hostel facilities amenities rules: ",
    ("faculty",    "list"):     "Faculty members list names department: ",
    ("faculty",    "profile"):  "Faculty profile professor details: ",
    ("programs",   "overview"): "Academic program curriculum courses: ",
    ("placements", "table"):    "Placement statistics package salary recruiters: ",
    ("placements", "list"):     "Major recruiters companies MIT Bengaluru placements: ",
    ("placements", "overview"): "MIT Bengaluru placement information: ",
}

def embed_query(text: str, category: str = "", subtype: str = "") -> list:
    """
    Embed using Pinecone hosted llama-text-embed-v2.
    input_type='query' (vs 'passage' used during indexing) — correct for retrieval.
    """
    pc, _ = _get_resources()
    prefix = EMBED_PREFIXES.get((category, subtype), "")
    result = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[prefix + text],
        parameters={"input_type": "query", "truncate": "END"}
    )
    return result[0].values



# ── Department detection ──────────────────────────────────────────────────────
DEPT_KEYWORDS = {
    "computer science": ["computer science", "cse", " cs "],
    "electronics":      ["electronics", "ece", "communication engineering"],
    "mechanical":       ["mechanical", " me "],
    "civil":            ["civil", " ce "],
    "chemical":         ["chemical", "chem"],
    "electrical":       ["electrical", "eee"],
    "biotechnology":    ["biotech", "biotechnology"],
    "mathematics":      ["mathematics", "maths"],
    "artificial intelligence": ["artificial intelligence", "aiml"],
    "data science":     ["data science"],
    "aeronautical":     ["aeronautical"],
}

def detect_department(query: str) -> str:
    q = " " + query.lower() + " "
    for dept, aliases in DEPT_KEYWORDS.items():
        if any(a in q for a in aliases):
            return dept
    return ""


# ── Query intent detection ────────────────────────────────────────────────────
def detect_query_intent(query: str) -> tuple:
    """
    Returns (category, preferred_subtype, department_hint).
    All 3 values always returned — unpack as: cat, sub, dept = detect_query_intent(q)
    """
    q    = query.lower()
    dept = detect_department(query)

    is_list = any(w in q for w in ["list", "all", "who are", "members", "names"])
    is_fee  = any(w in q for w in ["fee", "fees", "cost", "charges", "how much",
                                    "price", "room type", "single ac", "double ac"])

    # HOD / Head of department (was completely missing before)
    if any(w in q for w in ["hod", "head of", "head of department",
                              "department head", "incharge", "chair of"]):
        return "department", "overview", dept

    # Faculty
    if any(w in q for w in ["faculty", "professor", "teacher", "staff",
                              "who teaches", "lecturer", "dr.", "dr "]):
        if is_list:
            return "faculty", "list", dept
        return "faculty", "profile", dept

    # Hostel fees (before generic hostel check)
    if is_fee and any(w in q for w in ["hostel", "room", "accommodation", "mess"]):
        return "hostel", "table", ""

    # Hostel general
    if any(w in q for w in ["hostel", "accommodation", "mess", "dining",
                              "stay", "dormitory", "warden", "facilities"]):
        return "hostel", None, ""

    # Programs
    if any(w in q for w in ["course", "program", "btech", "mtech", "degree",
                              "curriculum", "specialization", "admission", "eligibility"]):
        return "programs", None, dept

    # Non-hostel fees
    if is_fee:
        return "admissions", "table", ""

    # News / events
    if any(w in q for w in ["event", "news", "conference", "workshop", "seminar"]):
        return "news", None, ""

    return None, None, dept


# ── Query expansion ───────────────────────────────────────────────────────────
EXPANSIONS = {
    "cs faculty":          "Computer Science faculty members list MIT Bengaluru names professors",
    "cse faculty":         "Computer Science Engineering faculty members MIT Bengaluru",
    "ece faculty":         "Electronics Communication Engineering faculty MIT Bengaluru",
    "hostel fee":          "hostel room fee structure charges 2025 MIT Bengaluru annual AC non-AC",
    "hostel room":         "hostel room types single double AC non-AC fee MIT Bengaluru",
    "cs department":       "Computer Science department MIT Bengaluru faculty programs",
    "list all faculty":    "faculty members list names MIT Bengaluru professors department",
    "who are the faculty": "faculty members list names department MIT Bengaluru professors",
    "head of":             "head of department HOD designation MIT Bengaluru",
    "placement":           "placement statistics companies recruiters package MIT Bengaluru",
}

def expand_query(query: str) -> str:
    q_lower = query.lower()
    for shorthand, expansion in EXPANSIONS.items():
        if shorthand in q_lower:
            return expansion
    return query


# ── Reranking ─────────────────────────────────────────────────────────────────
SUBTYPE_BOOST = {
    "table":    0.30,
    "list":     0.25,
    "profile":  0.10,
    "overview": -0.20,
    "handbook": -0.10,
}

NOISE_PHRASES = [
    "home away from home", "institution of eminence",
    "holistic student development", "shaping future leaders",
    "committed to making your stay", "dynamic learning environment",
]
def rerank(matches: list, preferred_subtype: str, dept_hint: str) -> list:
    scored = []
    
    # Use the absolute built-in reference directly to avoid any local shadowing
    import builtins
    
    # Ensure matches is a valid list
    if not matches:
        return []
    
    for m in matches:
        # 1. CRITICAL: Skip if the match object itself is None
        if m is None:
            continue
            
        try:
            # 2. Use builtins.dict to bypass any variable named 'dict'
            # Pinecone 'ScannableObject' must be converted to a standard dictionary
            if hasattr(m, "to_dict"):
                match_dict = m.to_dict()
            else:
                match_dict = builtins.dict(m) 
            
            meta = match_dict.get("metadata", {})
            
            # 3. Ensure score is a float and handle potential NoneType scores
            score_val = match_dict.get("score")
            cosine = float(score_val) if score_val is not None else 0.0

            subtype = meta.get("subtype", "overview")
            importance = meta.get("importance", "normal")
            dept = meta.get("department", "").lower()
            text = meta.get("text", "").lower()

            s_boost = SUBTYPE_BOOST.get(subtype, 0.0)
            i_boost = 0.15 if importance == "high" else 0.0
            intent_boost = 0.20 if (preferred_subtype and subtype == preferred_subtype) else 0.0
            dept_boost = 0.15 if (dept_hint and (dept_hint in dept or dept_hint in text[:300])) else 0.0
            noise_pen = -0.25 if any(p in text for p in NOISE_PHRASES) else 0.0

            final = cosine + s_boost + i_boost + intent_boost + dept_boost + noise_pen
            
            # 4. Use builtins.round for safety
            match_dict["final_score"] = builtins.round(final, 4)
            scored.append(match_dict)
            
        except Exception as e:
            if DEBUG:
                print(f"  [DEBUG] Skipping result due to error: {e}")
            continue

    scored.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return scored

# ── Name-exact search fallback ────────────────────────────────────────────────
def name_search_fallback(index, query_vec: list, query: str) -> str | None:
    """
    For queries like "who is Iven Jose":
    Cosine similarity is unreliable for uncommon proper nouns.
    Pull top-60 faculty profiles, do token-level string matching instead.
    """
    name_match = re.search(
        r"(?:who is|about|tell me about|details of)\s+"
        r"((?:dr\.?\s+|prof\.?\s+|mr\.?\s+|ms\.?\s+)?[a-z]+(?: [a-z]+){0,3})",
        query, re.I
    )
    if not name_match:
        return None

    tokens = [t for t in name_match.group(1).lower().split() if len(t) > 2]
    if not tokens:
        return None

    if DEBUG:
        print(f"  [DEBUG] Name lookup tokens: {tokens}")

    results = index.query(
        vector=query_vec,
        top_k=60,
        include_metadata=True,
        namespace=NAMESPACE,
        filter={"category": {"$eq": "faculty"}, "subtype": {"$eq": "profile"}},
    )

    for match in results.get("matches", []):
        meta        = match.get("metadata", {})
        stored_name = meta.get("name", "").lower()
        stored_text = meta.get("text", "").lower()[:400]
        if all(t in stored_name or t in stored_text for t in tokens):
            if DEBUG:
                print(f"  [DEBUG] Name match found: {meta.get('name', '')}")
            return meta.get("text", "")

    return None


# ── Core search ───────────────────────────────────────────────────────────────
def search_campus_data(user_query: str, top_k: int = 3) -> str:
    pc, index = _get_resources()

    expanded = expand_query(user_query)
    if DEBUG and expanded != user_query:
        print(f"  [DEBUG] Expanded: {expanded}")

    category, preferred_subtype, dept_hint = detect_query_intent(user_query)

    is_name_lookup = bool(re.search(
        r"who is|tell me about|details of|about (dr|prof|mr|ms)\b",
        user_query, re.I
    ))

    if DEBUG:
        print(f"  [DEBUG] Intent → category={category} subtype={preferred_subtype} "
              f"dept='{dept_hint}' name_lookup={is_name_lookup}")

    # Embed query using the same model as embed_and_store.py
    try:
        query_vec = embed_query(expanded, category or "", preferred_subtype or "")
    except Exception as e:
        print(f"  [ERROR] Embedding failed: {e}")
        return "No relevant information found in the MIT Bengaluru knowledge base."

    def run_query(filter_dict=None, k=10):
        kwargs = {
            "vector":           query_vec,
            "top_k":            k,
            "include_metadata": True,
            "namespace":        NAMESPACE,
        }
        if filter_dict:
            kwargs["filter"] = filter_dict
        resp = index.query(**kwargs)
        if DEBUG:
            n = len(resp.get("matches", []))
            f = str(filter_dict) if filter_dict else "none"
            print(f"  [DEBUG] filter={f} → {n} hits")
            for m in resp.get("matches", []):
                meta = m.get("metadata", {})
                print(f"    cosine={m['score']:.3f} "
                      f"cat={meta.get('category','')} "
                      f"sub={meta.get('subtype','')} "
                      f"imp={meta.get('importance','')} "
                      f"dept={meta.get('department','')} "
                      f"id={m['id']}")
        return resp

    # ── 1. Name-specific lookup ───────────────────────────────────────────────
    if is_name_lookup:
        exact = name_search_fallback(index, query_vec, user_query)
        if exact:
            return _format([{"metadata": {"text": exact, "category": "faculty",
                                          "subtype": "profile"},
                             "score": 1.0, "final_score": 1.0, "id": "name_match"}])

    # ── 2. Faculty list ───────────────────────────────────────────────────────
    if category == "faculty" and preferred_subtype == "list":
        # Try pre-built list chunk first
        r = run_query({"category": {"$eq": "faculty"}, "subtype": {"$eq": "list"}})
        if r.get("matches"):
            reranked = rerank(r["matches"], "list", dept_hint)
            text = reranked[0]["metadata"].get("text", "")
            # Filter to the right department if query was dept-specific
            if dept_hint:
                lines = text.splitlines()
                filtered = [l for l in lines if ":" in l or dept_hint in l.lower()]
                if len(filtered) > 2:
                    text = "\n".join(filtered)
            return text

        # Synthesise list from profiles at runtime
        if DEBUG:
            print("  [DEBUG] No pre-built list chunk — synthesising from profiles")
        r = run_query({"category": {"$eq": "faculty"}, "subtype": {"$eq": "profile"}}, k=60)
        entries = []
        for p in r.get("matches", []):
            meta  = p.get("metadata", {})
            pdept = meta.get("department", "").lower()
            ptext = meta.get("text", "")
            if dept_hint and dept_hint not in pdept and dept_hint not in ptext.lower()[:300]:
                continue
            name  = meta.get("name", "")
            desig = ""
            dm = re.search(r"Designation:\s*(.+)", ptext)
            if dm:
                desig = dm.group(1).strip()
            if name:
                entries.append(f"• {name}" + (f" ({desig})" if desig else ""))
        if entries:
            label = dept_hint.title() if dept_hint else "MIT Bengaluru"
            return f"Faculty Members – {label}:\n" + "\n".join(entries[:40])

        # Last resort: return individual profiles
        if r.get("matches"):
            return _format(rerank(r["matches"], "profile", dept_hint)[:top_k])

    # ── 3. Fee / table queries ────────────────────────────────────────────────
    if preferred_subtype == "table":
        r = run_query({"subtype": {"$eq": "table"}})
        if r.get("matches"):
            reranked = rerank(r["matches"], "table", "")
            # Extra boost for chunks with actual fee numbers
            for item in reranked:
                if re.search(r"[\d,]{4,}\s*(\||INR|USD|₹)", item["metadata"].get("text", "")):
                    item["final_score"] += 0.30
            reranked.sort(key=lambda x: x["final_score"], reverse=True)
            return _format(reranked[:top_k])
        # Fall back to hostel category (PDF handbook has fee text)
        r = run_query({"category": {"$eq": "hostel"}})
        if r.get("matches"):
            return _format(rerank(r["matches"], "table", "")[:top_k])

    # ── 4. Standard cascade ───────────────────────────────────────────────────
    matches = []

    # Tier 1: category + subtype
    if category and preferred_subtype:
        matches = run_query({"category": {"$eq": category},
                             "subtype":  {"$eq": preferred_subtype}}).get("matches", [])

    # Tier 2: category only
    if not matches and category:
        matches = run_query({"category": {"$eq": category}}).get("matches", [])

    # Tier 3: unfiltered (broad semantic)
    if not matches:
        matches = run_query().get("matches", [])

    if not matches:
        return "No relevant information found in the MIT Bengaluru knowledge base."

    reranked = rerank(matches, preferred_subtype or "", dept_hint)

    # Deduplicate by first 200 chars
    seen, deduped = set(), []
    for item in reranked:
        key = item["metadata"].get("text", "")[:200]
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return _format(deduped[:top_k])


# ── Result formatter ──────────────────────────────────────────────────────────
def _format(matches: list) -> str:
    if not matches:
        return "No relevant information found in the MIT Bengaluru knowledge base."

    parts = []
    for m in matches:
        meta       = m.get("metadata", {})
        text       = meta.get("text", "")
        cat        = meta.get("category", "")
        subtype    = meta.get("subtype", "")
        dept       = meta.get("department", "")
        name       = meta.get("name", "")
        importance = meta.get("importance", "")
        cosine     = round(float(m.get("score", 0)) * 100, 1)
        final      = round(float(m.get("final_score", m.get("score", 0))) * 100, 1)

        tags = []
        if cat:        tags.append(f"Category: {cat}")
        if subtype:    tags.append(f"Type: {subtype}")
        if dept:       tags.append(f"Dept: {dept}")
        if name:       tags.append(f"Faculty: {name}")
        if importance == "high": tags.append("⭐ Structured")

        label = " | ".join(tags) if tags else "General"
        parts.append(f"[{label} | Cosine: {cosine}% → Reranked: {final}%]\n{text}")

    return "\n\n---\n\n".join(parts)


# ── CLI test harness ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What courses does MIT Bengaluru offer?",
        "Who are the faculty members in the CS department?",
        "List all faculty in Computer Science",
        "What are the hostel facilities?",
        "What is the hostel fee structure?",
        "What are the hostel room types and fees?",
        "Tell me about the BTech Computer Science program",
        "Who is head of electronics department?",
        "who is iven jose",
        "Who are the faculty in ECE?",
        "What is the fee for AC single room?",
    ]
    for q in test_queries:
        cat, sub, dept = detect_query_intent(q)
        print(f"\n{'='*60}")
        print(f"Query  : {q}")
        print(f"Intent : category={cat} | subtype={sub} | dept='{dept}'")
        print(search_campus_data(q))

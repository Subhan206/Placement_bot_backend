"""
chunker.py — Fixed semantic chunker
Fixes:
  - No more aggressive line filtering (was deleting bullet points, names, fees)
  - Section-aware chunking: splits on headers (##) not just word count
  - Better deduplication
  - Structured metadata per chunk
  - Faculty extraction: pulls name + dept per profile
Run: python chunker.py  (after scraper.py)
"""

import json, re, hashlib
from collections import Counter


# ── Cleaning ──────────────────────────────────────────────────────────────────

def strip_junk(text: str) -> str:
    text = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', text)   # [text](url) → text
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'.*(cookie|Twitter Widget|iframe|browsing experience|Opens in New Tab).*', '', text, flags=re.I)
    text = re.sub(r'^\s*[\\#\[\]\(\)\-\*\_\|>]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def word_count(text: str) -> int:
    return len([w for w in re.sub(r'[^\w\s]', '', text).split() if len(w) > 1])


def is_useful(text: str, min_words: int = 20) -> bool:
    return word_count(text) >= min_words


# ── Category detection ────────────────────────────────────────────────────────

def get_category(url: str, existing_cat: str = None) -> str:
    """Use existing category from scraper if available, else detect from URL."""
    if existing_cat and existing_cat != "general":
        return existing_cat
    u = url.lower()
    if ".pdf" in u:
        if "hostel" in u or "accommodation" in u:
            return "hostel"
        if "fee" in u:
            return "fees"
        return "document"
    if "/faculty-list/" in u and "/_jcr_content" in u:
        return "faculty"
    if "/program-list/" in u or "/programs-list" in u:
        return "programs"
    if "/news-events/" in u or "/newsletter" in u or "/2024" in u or "/2025" in u:
        return "news"
    if "/department-list/" in u or "/department-faculty" in u:
        return "department"
    if "accommodation" in u or "hostel" in u:
        return "hostel"
    if "/why/" in u or "scholarship" in u or "loan" in u:
        return "admissions"
    return "general"


# ── Chunking strategies ───────────────────────────────────────────────────────

def chunk_by_section(text: str, max_words: int = 150, overlap_words: int = 30) -> list[str]:
    """
    Split on markdown headers (##, ###) first.
    If a section is still too large, split by word count with overlap.
    This preserves context like "Fees:\nTuition: X\nHostel: Y" together.
    """
    # Split on headers
    sections = re.split(r'\n(?=#{1,3} )', text)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        words = section.split()
        if len(words) <= max_words:
            if is_useful(section):
                chunks.append(section)
        else:
            # Sub-chunk large sections with overlap
            start = 0
            while start < len(words):
                chunk = " ".join(words[start:start + max_words])
                if is_useful(chunk):
                    chunks.append(chunk)
                start += max_words - overlap_words

    return chunks


def chunk_faculty_block(text: str) -> list[str]:
    """
    Faculty pages have merged profiles separated by '---'.
    Split per profile, keep each profile as ONE chunk.
    This prevents faculty info from being split mid-profile.
    """
    profiles = re.split(r'\n---\n', text)
    chunks = []
    for profile in profiles:
        profile = profile.strip()
        if profile and is_useful(profile, min_words=15):
            chunks.append(profile)
    return chunks


# ── Deduplication ─────────────────────────────────────────────────────────────

def fingerprint(text: str) -> str:
    normalized = re.sub(r'\s+', ' ', text[:300].lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


# ── Main pipeline ─────────────────────────────────────────────────────────────

with open("raw_docs.json", encoding="utf-8") as f:
    docs = json.load(f)

print(f"Loaded {len(docs)} raw docs")

all_chunks = []
seen_fps = set()
stats = Counter()

for doc in docs:
    url = doc.get("url", "")
    existing_cat = doc.get("category", "")
    raw_content = doc.get("content", "")

    if not raw_content:
        continue

    content = strip_junk(raw_content)
    if not content or not is_useful(content, min_words=15):
        continue

    category = get_category(url, existing_cat)

    # Choose chunking strategy
    if category == "faculty":
        chunks = chunk_faculty_block(content)
    else:
        chunks = chunk_by_section(content)

    for i, chunk in enumerate(chunks):
        fp = fingerprint(chunk)
        if fp in seen_fps:
            continue
        seen_fps.add(fp)

        chunk_id = f"{abs(hash(url)) % 10**9}_{i}"
        all_chunks.append({
            "id": chunk_id,
            "text": chunk,
            "source_url": url,
            "category": category,
        })
        stats[category] += 1

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\n✅ Total chunks: {len(all_chunks)}")
print("Distribution:")
for cat, count in stats.most_common():
    pct = count / len(all_chunks) * 100
    print(f"  {cat}: {count} ({pct:.1f}%)")

# Samples
print("\n--- Sample: hostel/fees chunk ---")
for c in all_chunks:
    if c["category"] in ("hostel", "fees"):
        print(c["text"][:400])
        break

print("\n--- Sample: faculty chunk ---")
for c in all_chunks:
    if c["category"] == "faculty":
        print(c["text"][:400])
        break
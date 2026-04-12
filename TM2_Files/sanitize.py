import json, re

# ─── BLOCK: other campuses and institutions ───
BLOCKLIST = [
    "mit manipal", "manipal institute of technology, manipal",
    "udupi", "mangalore", "576104", "manipal - 576",
    "kasturba",
    "manipal college of nursing", "manipal college of dental sciences",
    "manipal college of pharmaceutical sciences",
    "welcomgroup graduate school",
    "manipal institute of communication",
    "manipal institute of management",
    "american university of antigua",
    "melaka manipal",
    "manipal school of life sciences",
    "manipal centre for natural sciences",
]

# ─── ALLOW: MIT Bengaluru identifiers (expanded to match real site structure) ───
MIT_BENGALURU_IDENTIFIERS = [
    # URL-based identifiers (these will appear in content too)
    "mit-blr", "mit blr",

    # How the site actually refers to it
    "mahe bengaluru", "mahe-bengaluru",
    "manipal institute of technology bengaluru",
    "manipal institute of technology, bengaluru",
    "manipal institute of technology bangalore",
    "mit bengaluru", "mit bangalore", "mit-b",
    "yelahanka",

    # Departments (MIT Bengaluru specific)
    "computer science", "cse",
    "information science", "ise",
    "electronics", "ece",
    "electrical", "eee",
    "mechanical engineering",
    "civil engineering",
    "chemical engineering",
    "biotechnology",
    "artificial intelligence", "ai & ml", "aiml",
    "data science",
    "aeronautical",
    "industrial engineering",

    "hostel", "accommodation", "residence hall", "residential facility",
    "boys hostel", "girls hostel", "ladies hostel", "gents hostel",
    "on-campus stay", "on campus stay", "student housing",
    "mess", "dining hall", "canteen", "food court",
    "warden", "hostel warden", "hostel block", "hostel room",
    "hostel fee", "hostel charges", "hostel rules", "hostel facilities",
    "laundry", "common room", "recreation room",
    "24/7 security", "cctv", "wifi hostel",
]

def is_mit_bengaluru_content(text: str, url: str = "") -> bool:
    text_lower = text.lower()
    url_lower = url.lower()

    # ✅ STRONG ACCEPT: URL clearly belongs to Bengaluru section
    if "mahe-bengaluru" in url_lower or "mit-blr" in url_lower:
        return True

    # ❌ HARD BLOCK: clear Manipal campus indicators
    HARD_BLOCK = [
        "udupi",
        "576104",
        "manipal campus, manipal",
        "manipal institute of technology, manipal"
    ]

    if any(term in text_lower for term in HARD_BLOCK):
        return False

    # ⚠️ SOFT ACCEPT: Bengaluru-related keywords
    SOFT_ALLOW = ["bengaluru", "bangalore", "yelahanka"]

    if any(term in text_lower for term in SOFT_ALLOW):
        return True

    # ❌ Otherwise reject (this is the balancing part)
    return False

def clean_markdown(text: str) -> str:
    text = re.sub(r'\[.*?cookie.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[Skip to.*?\]', '', text, flags=re.IGNORECASE)
    lines = text.splitlines()
    lines = [l for l in lines if len(l.strip()) > 20 or l.strip() == ""]
    return "\n".join(lines).strip()

def is_quality_content(text: str) -> bool:
    """Rejects chunks that are pure navigation noise or too link-heavy."""
    # If more than 40% of lines are markdown links, it's a nav dump
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    link_lines = sum(1 for l in lines if l.startswith("[") and "](http" in l)
    if len(lines) > 0 and link_lines / len(lines) > 0.4:
        return False
    # Must have some real words, not just links
    plain_text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    plain_text = re.sub(r'\s+', ' ', plain_text).strip()
    if len(plain_text) < 150:
        return False
    return True
# ─── Main ───
with open("raw_docs.json", encoding="utf-8") as f:
    docs = json.load(f)

clean_docs = []
rejected_count = 0
rejection_log = []

for doc in docs:
    content = doc.get("content", "")
    url = doc.get("url", "")

    if not content or len(content) < 100:
        rejected_count += 1
        continue

    # URL-level filter
    if not is_mit_bengaluru_content(content, url):
        rejected_count += 1
        rejection_log.append(url)
        continue

    # Content quality filter
    cleaned = clean_markdown(content)
    if not is_quality_content(cleaned):
        rejected_count += 1
        rejection_log.append(f"[LOW QUALITY] {url}")
        continue

    doc["content"] = cleaned
    clean_docs.append(doc)
print("\n" + "="*50)
print(f"✅ Total input docs: {len(docs)}")
print(f"🧹 Clean docs kept: {len(clean_docs)}")
print(f"❌ Rejected docs: {rejected_count}")
print(f"📊 Retention rate: {len(clean_docs)/len(docs)*100:.2f}%")
print("="*50)
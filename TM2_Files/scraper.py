
"""
scraper.py — Fully parallel scraper
  PDFs  → Firecrawl (async, all 4 at once)
  HTML  → 16 persistent worker threads sharing one queue
          New links are fed back into the queue in real time
          No batching — workers never sit idle waiting for a batch to finish

Run: python scraper.py
"""

import requests, re, json, time, os, asyncio
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from queue import Queue, Empty
from threading import Lock, Thread
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DOMAIN       = "https://www.manipal.edu"
VALID_PATH_PREFIX = "/mu/campuses/mahe-bengaluru"
MAX_PAGES         = 300
NUM_WORKERS       = 16       # simultaneous HTML threads
DELAY             = 0.2      # polite delay per thread (all 16 run in parallel)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}

HTML_SEEDS = [
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr/department-faculty.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr/program-list.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/campus-life/accommodation.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr/department-faculty/department-list/school-of-computer-science.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr/department-faculty/department-list/electronics-communication-engineering.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr/department-faculty/department-list/mechanical-engineering.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr/department-faculty/department-list/civil-engineering.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr/department-faculty/department-list/chemical-engineering.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr/why/education-loan--scholarships-portal.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr/why/study-abroad-program.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr/news-events.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/institution-list/mit-blr/newsletter.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/academics/programs-list.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/about-us/leadership.html",
    "https://www.manipal.edu/mu/campuses/mahe-bengaluru/contact-us.html",
]

PDF_SEEDS = [
    "https://www.manipal.edu/content/dam/manipal/mu/hostel/MAHE%20B%27LRU%20new%20Hostel%20Handbook_2024%20for%20website%20PDF.pdf",
    "https://www.manipal.edu/content/dam/manipal/mu/maheblr/MAHE%20Bangalore%20Hostels%20Annexure%201.pdf",
    "https://www.manipal.edu/content/dam/manipal/mu/hostel/Revised-guidelines-on-hostel-refund-rules-2025-26-Blr-campus.pdf",
    "https://www.manipal.edu/content/dam/manipal/mu/documents/mahe/fee-notifications/ay-2025-2026/MAHE%20BANGALORE%20HOSTEL%20FEE%2025-26.pdf",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_category(url: str) -> str:
    u = url.lower()
    if ".pdf" in u:
        if any(k in u for k in ["hostel", "handbook", "annexure", "refund"]):
            return "hostel"
        if "fee" in u:
            return "fees"
        return "document"
    if "/faculty-list/" in u:                          return "faculty"
    if "/program-list/" in u or "/programs-list" in u: return "programs"
    if "/news-events/" in u or "/newsletter" in u:     return "news"
    if "/department-list/" in u or "/department-faculty" in u: return "department"
    if "accommodation" in u or "hostel" in u:          return "hostel"
    if "/why/" in u or "scholarship" in u:             return "admissions"
    return "general"


def is_valid_url(url: str) -> bool:
    if not url or not url.startswith("http"):           return False
    if "chrome-extension" in url:                       return False
    if url.count(".html") > 1:                          return False
    if VALID_PATH_PREFIX not in url:                    return False
    if any(b in url for b in [
        "mahe-mlr", "mangalore", "melaka", "antigua",
        "sitemap", ".xml", ".jpg", ".png", ".gif",
        "mailto:", "javascript:"
    ]):
        return False
    return True


def is_useful(text: str, min_words: int = 15) -> bool:
    return len([w for w in re.sub(r'[^\w\s]', '', text).split() if len(w) > 1]) >= min_words


def extract_links(soup: BeautifulSoup) -> list:
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if href.startswith("/"):
            href = BASE_DOMAIN + href
        elif not href.startswith("http"):
            continue
        href = href.split("#")[0].strip()
        if is_valid_url(href):
            links.append(href)
    return list(set(links))


def clean_html(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(["nav", "footer", "header", "script",
                               "style", "iframe", "noscript", "aside"]):
        tag.decompose()
    for tag in soup.find_all(class_=re.compile(
            r"cookie|social|share|nav|menu|sidebar|breadcrumb|pagination", re.I)):
        tag.decompose()

    main = (
        soup.find("main") or
        soup.find(id=re.compile(r"main|content|body", re.I)) or
        soup.find(class_=re.compile(r"main|content|article", re.I)) or
        soup.find("body") or soup
    )

    raw = md(str(main), heading_style="ATX", strip=["a"])
    raw = re.sub(r'!\[.*?\]\(.*?\)', '', raw)
    raw = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', raw)
    raw = re.sub(r'https?://\S+', '', raw)
    raw = re.sub(r'.*(cookie|Twitter|browsing experience|Skip to).*', '', raw, flags=re.I)
    raw = re.sub(r'^\s*[\\#\[\]\(\)\-\*\_\|>]+\s*$', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    raw = re.sub(r'[ \t]+', ' ', raw)
    return "\n".join(l for l in raw.splitlines() if l.strip()).strip()


# ── PDF scraping (Firecrawl, async, all 4 simultaneously) ────────────────────
async def scrape_pdfs() -> list:
    key = os.getenv("FIRECRAWL_API_KEY")
    if not key:
        print("  ⚠️  No FIRECRAWL_API_KEY — skipping PDFs")
        return []
    try:
        from firecrawl.v1 import AsyncV1FirecrawlApp
        app = AsyncV1FirecrawlApp(api_key=key)
    except ImportError:
        print("  ⚠️  firecrawl-py not installed — skipping PDFs")
        return []

    # Fire all 4 PDF requests simultaneously
    async def fetch_one(url):
        try:
            result = await app.scrape_url(url, formats=["markdown"])
            content = ""
            if hasattr(result, "markdown") and result.markdown:
                content = result.markdown.strip()
            elif isinstance(result, dict):
                content = (result.get("markdown") or result.get("content") or "").strip()
            return url, content
        except Exception as e:
            print(f"  ❌ {url.split('/')[-1][:40]}: {e}")
            return url, ""

    tasks = [fetch_one(url) for url in PDF_SEEDS]
    results = await asyncio.gather(*tasks)   # all 4 in parallel

    docs = []
    for url, content in results:
        if content and is_useful(content):
            cat = get_category(url)
            docs.append({"url": url, "content": content, "category": cat})
            print(f"  ✅ {url.split('/')[-1][:50]}  [{cat}]  {len(content):,} chars")
        else:
            print(f"  ⚠️  No content: {url.split('/')[-1][:50]}")
    return docs


# ── HTML scraping (persistent thread pool with shared queue) ──────────────────
def scrape_html_parallel(max_pages: int = MAX_PAGES) -> list:
    """
    16 worker threads share one Queue.
    Each thread picks a URL, scrapes it, then pushes newly discovered
    links back into the queue — no batching, no idle waiting.
    """
    url_queue  = Queue()
    seen_lock  = Lock()
    docs_lock  = Lock()
    seen_urls  = set(HTML_SEEDS)
    all_docs   = []
    done_count = [0]   # mutable int shared across threads

    for url in HTML_SEEDS:
        url_queue.put(url)

    def worker():
        while True:
            # Stop if we've hit the page limit
            with docs_lock:
                if done_count[0] >= max_pages:
                    break

            try:
                url = url_queue.get(timeout=5)   # wait up to 5s for new URLs
            except Empty:
                break   # queue empty and no new links coming → we're done

            try:
                time.sleep(DELAY)
                resp = requests.get(url, headers=HEADERS, timeout=12)
                if resp.status_code != 200:
                    url_queue.task_done()
                    continue

                soup    = BeautifulSoup(resp.text, "html.parser")
                content = clean_html(soup)
                links   = extract_links(soup)

                # Save doc if useful
                if content and is_useful(content):
                    with docs_lock:
                        if done_count[0] < max_pages:
                            cat = get_category(url)
                            all_docs.append({"url": url, "content": content, "category": cat})
                            done_count[0] += 1
                            short = url.split("mahe-bengaluru/")[-1][:55]
                            print(f"  ✅ [{done_count[0]:03d}] [{cat:10s}] {short}")
                else:
                    short = url.split("mahe-bengaluru/")[-1][:55]
                    print(f"  ⚠️  skip [{short}]")

                # Enqueue new links
                with seen_lock:
                    for link in links:
                        if link not in seen_urls:
                            seen_urls.add(link)
                            url_queue.put(link)

            except Exception as e:
                pass   # silently skip failures
            finally:
                url_queue.task_done()

    # Start all workers simultaneously
    threads = [Thread(target=worker, daemon=True) for _ in range(NUM_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return all_docs


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    start = time.time()
    print("=" * 60)
    print(f"🚀 MIT Bengaluru Scraper  ({NUM_WORKERS} HTML threads)")
    print("=" * 60)

    all_docs = []

    # PDFs — all 4 simultaneously via Firecrawl async
    print("\n📄 Phase 1: PDFs (all 4 simultaneously)")
    pdf_docs = await scrape_pdfs()
    all_docs.extend(pdf_docs)
    print(f"  → {len(pdf_docs)} PDF docs\n")

    # HTML — 16 threads, shared queue, continuous
    print(f"🌐 Phase 2: HTML ({NUM_WORKERS} threads, continuous queue)")
    html_docs = scrape_html_parallel()
    all_docs.extend(html_docs)

    with open("raw_docs.json", "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    from collections import Counter
    cats = Counter(d["category"] for d in all_docs)
    elapsed = round(time.time() - start, 1)

    print(f"\n{'='*60}")
    print(f"✅ Done in {elapsed}s")
    print(f"   PDFs : {len(pdf_docs)}")
    print(f"   HTML : {len(html_docs)}")
    print(f"   Total: {len(all_docs)}")
    print("\nCategory breakdown:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    print("\n✅ Saved → raw_docs.json")
    print("Next: python chunker.py")


if __name__ == "__main__":
    asyncio.run(main())

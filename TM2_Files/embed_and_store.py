"""
embed_and_store.py — Fast version
Speed fixes:
  1. Encode ALL chunks in one shot (batch=256, multi-process) instead of 100 at a time
  2. Upsert batches to Pinecone in parallel threads (8 threads simultaneously)
  3. No sleep between batches

Run: python embed_and_store.py  (after chunker.py)
"""

import json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "mit-bengaluru"
NAMESPACE        = "default"
BATCH_SIZE       = 100     # Pinecone upsert limit per call
UPSERT_THREADS   = 8       # parallel upsert threads
ENCODE_BATCH     = 256     # sentences per encoding batch (GPU/CPU optimized)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model ready")

# ── Pinecone setup ────────────────────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    print("✅ Index created")
else:
    print(f"✅ Index exists")

index = pc.Index(INDEX_NAME)

try:
    index.delete(delete_all=True, namespace=NAMESPACE)
    print("✅ Cleared existing vectors")
    time.sleep(2)
except Exception:
    print("ℹ️  Index was empty, nothing to clear")

# ── Load chunks ───────────────────────────────────────────────────────────────
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"\n📦 {len(chunks)} chunks to embed and store")

# ── Step 1: Encode ALL at once ────────────────────────────────────────────────
# encode() with batch_size=256 and show_progress_bar handles everything internally
# This is 2-3x faster than encoding 100 at a time in a loop
print("\n⚡ Encoding all chunks...")
t0 = time.time()

texts = [c["text"] for c in chunks]
embeddings = model.encode(
    texts,
    batch_size=ENCODE_BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # cosine similarity works best with normalized vectors
).tolist()

encode_time = round(time.time() - t0, 1)
print(f"✅ Encoded {len(embeddings)} chunks in {encode_time}s")

# ── Step 2: Build all vector dicts ────────────────────────────────────────────
vectors = [
    {
        "id": str(chunks[i]["id"]),
        "values": embeddings[i],
        "metadata": {
            "text":       chunks[i]["text"],
            "source_url": chunks[i]["source_url"],
            "category":   chunks[i].get("category", "general"),
        }
    }
    for i in range(len(chunks))
]

# ── Step 3: Upsert in parallel threads ────────────────────────────────────────
# Split into batches of 100 (Pinecone limit)
batches = [vectors[i:i + BATCH_SIZE] for i in range(0, len(vectors), BATCH_SIZE)]
print(f"\n🚀 Upserting {len(batches)} batches with {UPSERT_THREADS} parallel threads...")

t1 = time.time()
success, failed = 0, 0

def upsert_batch(args):
    batch_num, batch = args
    index.upsert(vectors=batch, namespace=NAMESPACE)
    return batch_num, len(batch)

with ThreadPoolExecutor(max_workers=UPSERT_THREADS) as executor:
    futures = {
        executor.submit(upsert_batch, (i + 1, batch)): i
        for i, batch in enumerate(batches)
    }
    for future in as_completed(futures):
        try:
            batch_num, count = future.result()
            success += count
            print(f"  ✅ Batch {batch_num}/{len(batches)} — {count} vectors")
        except Exception as e:
            failed += BATCH_SIZE
            print(f"  ❌ Batch failed: {e}")

upsert_time = round(time.time() - t1, 1)

# ── Done ──────────────────────────────────────────────────────────────────────
total_time = round(time.time() - t0, 1)

print(f"\n{'='*50}")
print(f"✅ Done!")
print(f"   Encode time : {encode_time}s")
print(f"   Upsert time : {upsert_time}s")
print(f"   Total time  : {total_time}s")
print(f"   Stored      : {success}")
print(f"   Failed      : {failed}")
print(f"{'='*50}")
print("Next: python search.py")
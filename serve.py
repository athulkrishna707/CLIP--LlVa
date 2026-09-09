import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel

# Database initialization
DB_PATH = Path("telemetry.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            query_text TEXT,
            top1_score REAL,
            latency_ms REAL,
            results_count INTEGER
        )
    """)
    conn.commit()
    conn.close()


init_db()

app = FastAPI(title="Product Multi-Modal Retrieval Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and index
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_NAME = "openai/clip-vit-base-patch32"
INDEX_PATH = Path("index/product_faiss.index")
META_PATH = Path("index/index_metadata.json")
CHECKPOINT_DIR = Path("checkpoints/best_model")

processor = None
model = None
faiss_index = None
index_metadata = []


@app.on_event("startup")
def load_resources():
    global processor, model, faiss_index, index_metadata

    print(f"Loading CLIP text encoder on {DEVICE}...")
    processor = CLIPProcessor.from_pretrained(BASE_MODEL_NAME)
    base_model = CLIPModel.from_pretrained(BASE_MODEL_NAME).to(DEVICE)

    if CHECKPOINT_DIR.exists():
        try:
            model = PeftModel.from_pretrained(base_model, str(CHECKPOINT_DIR)).to(DEVICE)
            print(f"Loaded fine-tuned model checkpoint from {CHECKPOINT_DIR}")
        except Exception as e:
            print(f"Notice: Could not load PeftModel ({e}). Using base model.")
            model = base_model
    else:
        print(f"Notice: Checkpoint {CHECKPOINT_DIR} not found. Using base CLIP model.")
        model = base_model

    model.eval()

    if INDEX_PATH.exists() and META_PATH.exists():
        faiss_index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            index_metadata = json.load(f)
        print(f"Loaded FAISS index with {faiss_index.ntotal} product vectors.")
    else:
        print("Warning: FAISS index or metadata not found. Run build_index.py first.")


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10


class SearchResultItem(BaseModel):
    id: str
    image_id: str
    image_path: str
    title: str
    description: str
    llava_caption: str
    category: str
    score: float


class SearchResponse(BaseModel):
    query: str
    top_k: int
    latency_ms: float
    results: List[SearchResultItem]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "index_vector_count": faiss_index.ntotal if faiss_index is not None else 0,
        "model_loaded": model is not None
    }


def extract_tensor(feat):
    """Unwraps tensor from CLIP output objects if necessary."""
    if isinstance(feat, torch.Tensor):
        return feat
    if hasattr(feat, "image_embeds") and feat.image_embeds is not None:
        return feat.image_embeds
    if hasattr(feat, "text_embeds") and feat.text_embeds is not None:
        return feat.text_embeds
    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
        return feat.pooler_output
    if hasattr(feat, "last_hidden_state") and feat.last_hidden_state is not None:
        return feat.last_hidden_state[:, 0, :]
    if isinstance(feat, (list, tuple)):
        return feat[0]
    return feat


@app.post("/search", response_model=SearchResponse)
def search_products(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    if faiss_index is None or not index_metadata:
        raise HTTPException(status_code=503, detail="Search index is not built yet. Run build_index.py.")

    start_time = time.time()

    # Encode query text
    inputs = processor(text=[request.query], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        text_features = extract_tensor(model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")))
        text_features = F.normalize(text_features, dim=-1)

    query_vec = text_features.cpu().numpy().astype(np.float32)

    # Search FAISS index
    k = min(request.top_k, faiss_index.ntotal)
    scores, indices = faiss_index.search(query_vec, k)

    scores = scores[0]
    indices = indices[0]

    results = []
    seen_ids = set()
    top1_score = float(scores[0]) if len(scores) > 0 else 0.0

    for score, idx in zip(scores, indices):
        if idx >= 0 and idx < len(index_metadata):
            meta = index_metadata[idx]
            img_id = meta.get("id", meta["image_id"])

            # Guarantee unique result items
            if img_id in seen_ids:
                continue
            seen_ids.add(img_id)

            results.append(SearchResultItem(
                id=img_id,
                image_id=img_id,
                image_path=meta["image_path"],
                title=meta["title"],
                description=meta.get("description", meta.get("llava_caption", "")),
                llava_caption=meta["llava_caption"],
                category=meta.get("category", "product"),
                score=round(float(score), 4)
            ))

    latency_ms = round((time.time() - start_time) * 1000.0, 2)

    # Log query to telemetry database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO search_telemetry (timestamp, query_text, top1_score, latency_ms, results_count) VALUES (?, ?, ?, ?, ?)",
            (time.time(), request.query, top1_score, latency_ms, len(results))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Telemetry logging error: {e}")

    return SearchResponse(
        query=request.query,
        top_k=k,
        latency_ms=latency_ms,
        results=results
    )


@app.get("/telemetry")
def get_telemetry():
    """Returns real telemetry data: average latency, p95 latency, average top-1 score, recent queries."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT timestamp, query_text, top1_score, latency_ms FROM search_telemetry ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_top1_score": 0.0,
            "recent_queries": []
        }

    latencies = [r[3] for r in rows]
    top1_scores = [r[2] for r in rows]

    avg_latency = round(float(np.mean(latencies)), 2)
    p95_latency = round(float(np.percentile(latencies, 95)), 2)
    avg_top1 = round(float(np.mean(top1_scores)), 4)

    recent_queries = [
        {
            "timestamp": r[0],
            "query": r[1],
            "top1_score": round(r[2], 4),
            "latency_ms": round(r[3], 2)
        }
        for r in rows[:15]
    ]

    return {
        "total_queries": len(rows),
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "avg_top1_score": avg_top1,
        "recent_queries": recent_queries
    }


# Mount data directory for image serving
data_dir = Path("data")
if data_dir.exists():
    app.mount("/data", StaticFiles(directory="data"), name="data")

# Mount static directory for search web interface
static_dir = Path("static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)

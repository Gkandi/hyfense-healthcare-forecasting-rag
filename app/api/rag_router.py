# app/api/rag_router.py
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List, Dict, Any
from pathlib import Path
import os, re, time
import pandas as pd
import numpy as np

# Retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter(prefix="/rag", tags=["rag"])

# ------------------------------
# Config
# ------------------------------
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", "docs")).resolve()
ALLOWED_EXT = {".md", ".txt"}  # keep it simple & robust
MAX_FEATURES = 50000
CHUNK_SIZE = 900          # ~900 chars per chunk
CHUNK_OVERLAP = 150       # overlap between chunks

# ------------------------------
# Global state (simple singleton)
# ------------------------------
class RagState:
    def __init__(self):
        self.files: List[Path] = []
        self.file_mtimes: Dict[str, float] = {}
        self.chunks: pd.DataFrame | None = None  # columns: ['doc_id','chunk_id','title','path','text']
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: Any = None  # sparse TF-IDF matrix
        self.last_built: float | None = None

RAG = RagState()

# ------------------------------
# Utilities
# ------------------------------
def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return p.read_text(errors="ignore")

_para_split = re.compile(r"\n\s*\n", flags=re.MULTILINE)
def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # split by paragraphs, accumulate to approx chunk_size with overlap
    paras = [s.strip() for s in _para_split.split(text) if s.strip()]
    if not paras:
        paras = [text.strip()]
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= chunk_size:
            buf = (buf + "\n" + p) if buf else p
        else:
            if buf:
                chunks.append(buf.strip())
            # start next buffer, include overlap tail from previous buffer
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = (tail + "\n" + p).strip()
    if buf:
        chunks.append(buf.strip())
    return chunks

def _scan_files() -> List[Path]:
    if not DOCS_DIR.exists():
        return []
    out = []
    for ext in ALLOWED_EXT:
        out.extend(sorted(DOCS_DIR.rglob(f"*{ext}")))
    return out

def _build_index() -> Dict[str, Any]:
    files = _scan_files()
    if not files:
        RAG.files = []
        RAG.file_mtimes = {}
        RAG.chunks = pd.DataFrame(columns=["doc_id","chunk_id","title","path","text"])
        RAG.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), max_features=MAX_FEATURES)
        RAG.matrix = RAG.vectorizer.fit_transform(pd.Series([""]))  # harmless empty fit
        RAG.last_built = time.time()
        return {"files": 0, "chunks": 0, "status": "no_files"}

    rows = []
    file_mtimes = {}
    for i, p in enumerate(files):
        text = _read_text(p)
        title = p.name
        file_mtimes[str(p)] = p.stat().st_mtime
        chs = _chunk_text(text)
        for j, ch in enumerate(chs):
            rows.append({"doc_id": i, "chunk_id": j, "title": title, "path": str(p), "text": ch})

    df = pd.DataFrame(rows)
    if df.empty:
        # ensure non-empty fit
        df = pd.DataFrame([{"doc_id": -1, "chunk_id": 0, "title": "(empty)", "path": "", "text": ""}])

    vect = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), max_features=MAX_FEATURES)
    mat = vect.fit_transform(df["text"])

    # commit to state
    RAG.files = files
    RAG.file_mtimes = file_mtimes
    RAG.chunks = df
    RAG.vectorizer = vect
    RAG.matrix = mat
    RAG.last_built = time.time()

    return {"files": len(files), "chunks": int(len(df)), "status": "ok"}

def _ensure_index():
    if RAG.chunks is None or RAG.vectorizer is None or RAG.matrix is None:
        _build_index()

def _is_stale() -> bool:
    if not RAG.files:
        return True
    for p in RAG.files:
        try:
            if RAG.file_mtimes.get(str(p)) != p.stat().st_mtime:
                return True
        except FileNotFoundError:
            return True
    return False

def _summarize_answer(question: str, hits: pd.DataFrame) -> str:
    """
    Very simple extractive summary: pick a few sentences from the
    top chunks that overlap most with the question terms.
    """
    if hits.empty:
        return "I couldn't find anything relevant in the documents."

    q_terms = set(re.findall(r"[a-zA-Z0-9_]+", question.lower()))
    def score_sentence(s: str) -> int:
        toks = set(re.findall(r"[a-zA-Z0-9_]+", s.lower()))
        return len(q_terms & toks)

    best = []
    for _, r in hits.head(3).iterrows():
        # split by sentence-ish
        sentences = re.split(r"(?<=[\.\!\?])\s+", r["text"])
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences.sort(key=score_sentence, reverse=True)
        best.extend(sentences[:2])

    if not best:
        # fallback to first ~2 lines of the best chunk
        first = hits.iloc[0]["text"].splitlines()[:2]
        best = [b.strip() for b in first if b.strip()]

    # Short stitched answer
    ans = " ".join(best[:4])
    return ans[:1200]  # be safe

# ------------------------------
# Routes
# ------------------------------
@router.post("/refresh")
def refresh_index():
    info = _build_index()
    return JSONResponse(content=jsonable_encoder({"docs_dir": str(DOCS_DIR), **info}))

@router.get("/sources")
def sources():
    _ensure_index()
    if RAG.chunks is None:
        raise HTTPException(500, "Index not built yet.")
    by_file = (RAG.chunks.groupby(["doc_id","title","path"]).size()
               .reset_index(name="chunks"))
    out = by_file.to_dict(orient="records")
    return JSONResponse(content=jsonable_encoder({
        "docs_dir": str(DOCS_DIR),
        "files_indexed": len(out),
        "chunks": int(len(RAG.chunks)),
        "last_built": RAG.last_built,
        "sources": out
    }))

@router.post("/ask")
def ask(payload: Dict[str, Any] = Body(...)):
    question: str = (payload.get("question") or "").strip()
    k: int = int(payload.get("k", 4))
    if not question:
        raise HTTPException(400, "Missing 'question'.")

    _ensure_index()
    if _is_stale():
        _build_index()

    if RAG.chunks is None or RAG.vectorizer is None or RAG.matrix is None:
        raise HTTPException(500, "RAG index not available.")

    # Vectorize question and score
    qv = RAG.vectorizer.transform([question])
    sims = cosine_similarity(qv, RAG.matrix).ravel()
    idx = np.argsort(-sims)[: max(1, k)]

    hits = RAG.chunks.iloc[idx].copy()
    hits["score"] = sims[idx]
    hits = hits.sort_values("score", ascending=False)

    # Prepare answer + citations
    answer = _summarize_answer(question, hits)

    citations = [{
        "rank": int(i+1),
        "title": r["title"],
        "path": r["path"],
        "score": float(r["score"]),
        "chunk_id": int(r["chunk_id"])
    } for i, (_, r) in enumerate(hits.iterrows())]

    # Return both citations and the actual chunk texts (handy for UI expanders)
    chunks = [{
        "title": r["title"],
        "path": r["path"],
        "chunk_id": int(r["chunk_id"]),
        "score": float(r["score"]),
        "text": r["text"]
    } for _, r in hits.iterrows()]

    return JSONResponse(content=jsonable_encoder({
        "question": question,
        "retriever": "tfidf_cosine",
        "top_k": int(k),
        "answer": answer,
        "citations": citations,
        "chunks": chunks
    }))

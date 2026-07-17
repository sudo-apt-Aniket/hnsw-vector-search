"""
VectorDB — a from-scratch vector database (brute force, KD-tree, HNSW)
with a small RAG pipeline on top, served over a Flask API + static UI.

Originally a Python port of a C++ prototype; since then: split into
modules, added disk persistence, added a cloud fallback for the RAG
pipeline (sentence-transformers + Groq) so it runs without a local
Ollama install, and added a pytest suite covering the algorithms.

Run:
    pip install -r requirements.txt
    python main.py
"""

import webbrowser
import threading as _t
from pathlib import Path

from flask import Flask, request, jsonify, send_file

from core import VectorDB, DocumentDB, get_dist_fn, chunk_text, cosine
from ai_providers import AIProvider
from demo_data import load_demo
import persistence

DIMS = 16  # demo vector dimensions

# =====================================================================
#  APP STATE
# =====================================================================

app = Flask(__name__)
db = VectorDB(DIMS)
doc_db = DocumentDB()
ai = AIProvider()

# Restore persisted state if it exists; otherwise seed with demo data
# and write it out so the first run creates the persisted file too.
_restored = persistence.load_vector_store(db, get_dist_fn("cosine"))
if _restored == 0:
    load_demo(db)
    persistence.save_vector_store(db)

persistence.load_doc_store(doc_db)


def _persist_vectors():
    try:
        persistence.save_vector_store(db)
    except Exception as e:
        app.logger.warning(f"vector store save failed: {e}")


def _persist_docs():
    try:
        persistence.save_doc_store(doc_db)
    except Exception as e:
        app.logger.warning(f"doc store save failed: {e}")


# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

@app.route("/search")
def search():
    v_str = request.args.get("v", "")
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    algo = request.args.get("algo", "hnsw")

    try:
        q = [float(x) for x in v_str.split(",") if x]
    except ValueError:
        q = []

    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400

    return jsonify(db.search(q, k, metric, algo))


@app.route("/insert", methods=["POST"])
def insert():
    data = request.get_json(force=True, silent=True) or {}
    meta = data.get("metadata", "")
    cat = data.get("category", "")
    emb = data.get("embedding", [])
    if not meta or len(emb) != DIMS:
        return jsonify({"error": "invalid body"}), 400
    id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    _persist_vectors()
    return jsonify({"id": id_})


@app.route("/delete/<int:id_>", methods=["DELETE"])
def delete(id_):
    ok = db.remove(id_)
    if ok:
        _persist_vectors()
    return jsonify({"ok": ok})


@app.route("/items")
def items():
    all_items = db.all()
    return jsonify([
        {"id": v.id, "metadata": v.metadata,
         "category": v.category, "embedding": v.emb}
        for v in all_items
    ])


@app.route("/benchmark")
def benchmark():
    v_str = request.args.get("v", "")
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    try:
        q = [float(x) for x in v_str.split(",") if x]
    except ValueError:
        q = []
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    return jsonify(db.benchmark(q, k, metric))


@app.route("/hnsw-info")
def hnsw_info():
    return jsonify(db.hnsw_info())


@app.route("/stats")
def stats():
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"]
    })


# ── DOCUMENT + RAG ENDPOINTS ─────────────────────────────────────

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    data = request.get_json(force=True, silent=True) or {}
    title = data.get("title", "")
    text = data.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400

    chunks = chunk_text(text, 250, 30)
    ids = []
    backend_used = None
    for i, chunk in enumerate(chunks):
        emb, backend = ai.embed(chunk)
        if not emb:
            return jsonify({"error": (
                "No embedding backend available. Either run Ollama locally "
                "(ollama pull nomic-embed-text) or install sentence-transformers "
                "(pip install sentence-transformers)."
            )}), 503
        backend_used = backend
        chunk_title = (f"{title} [{i + 1}/{len(chunks)}]"
                       if len(chunks) > 1 else title)
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    _persist_docs()
    return jsonify({
        "ids": ids, "chunks": len(chunks),
        "dims": doc_db.get_dims(), "embedBackend": backend_used
    })


@app.route("/doc/list")
def doc_list():
    docs = doc_db.all()
    result = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        result.append({
            "id": d.id,
            "title": d.title,
            "preview": preview,
            "words": len(d.text.split())
        })
    return jsonify(result)


@app.route("/doc/delete/<int:id_>", methods=["DELETE"])
def doc_delete(id_):
    ok = doc_db.remove(id_)
    if ok:
        _persist_docs()
    return jsonify({"ok": ok})


@app.route("/doc/search", methods=["POST"])
def doc_search():
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    k = int(data.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400
    q_emb, _backend = ai.embed(question)
    if not q_emb:
        return jsonify({"error": "No embedding backend available"}), 503
    hits = doc_db.search(q_emb, k)
    return jsonify({"contexts": [
        {"id": doc.id, "title": doc.title, "distance": round(d, 4)}
        for d, doc in hits
    ]})


@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    k = int(data.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400

    # Step 1: embed the question
    q_emb, embed_backend = ai.embed(question)
    if not q_emb:
        return jsonify({"error": "No embedding backend available"}), 503

    # Step 2: retrieve top-k relevant chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt
    ctx_parts = []
    for i, (d, doc) in enumerate(hits):
        ctx_parts.append(f"[{i + 1}] {doc.title}:\n{doc.text}\n")
    context = "\n".join(ctx_parts)

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{context}\n"
        f"Question: {question}\n\nAnswer:"
    )

    # Step 4: generate answer
    answer, gen_backend = ai.generate(prompt)
    if not answer:
        return jsonify({"error": (
            "No generation backend available. Either run Ollama locally "
            "or set a GROQ_API_KEY environment variable."
        )}), 503

    return jsonify({
        "answer": answer,
        "embedBackend": embed_backend,
        "genBackend": gen_backend,
        "contexts": [
            {"id": doc.id, "title": doc.title,
             "text": doc.text, "distance": round(d, 4)}
            for d, doc in hits
        ],
        "docCount": doc_db.size()
    })


@app.route("/status")
def status():
    ai_status = ai.status()
    return jsonify({
        **ai_status,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size()
    })


# ── SERVE FRONTEND ────────────────────────────────────────────────

@app.route("/")
def index():
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        return "index.html not found — place it in the same folder as main.py", 404
    return send_file(str(html_path))


# ── CORS (allow all origins) ──────────────────────────────────────

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    # Note: we deliberately do NOT eagerly check sentence-transformers here.
    # Loading it imports torch, which can take 10-20s on a cold start; it's
    # lazy-loaded on first actual /doc/insert or /doc/ask call instead, and
    # reported by /status once it's been probed.
    ollama_up = ai.ollama.is_available()
    groq_ready = ai.groq.is_available()
    print("=== VectorDB Engine (Python) ===", flush=True)
    print("http://localhost:8080", flush=True)
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce", flush=True)
    print(f"Ollama:              {'ONLINE' if ollama_up else 'offline'}", flush=True)
    print(f"Groq (GROQ_API_KEY): {'configured' if groq_ready else 'not set'}", flush=True)
    print("sentence-transformers: checked lazily on first /doc/insert or /doc/ask call", flush=True)
    _t.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8080")).start()
    app.run(host="0.0.0.0", port=8080, threaded=True)

# VectorDB

A vector database built from scratch in Python — brute force, KD-tree,
and HNSW (Hierarchical Navigable Small World) implemented without any
ANN library — with a small RAG (retrieval-augmented generation) pipeline
on top, served over a Flask API with a single-page UI.

Live demo: _add your hosted link here_

## Why three search algorithms

The project keeps all three implementations side by side on purpose,
so you can compare exact vs. approximate search directly:

| Algorithm    | Exact? | Complexity (query) | Notes |
|--------------|--------|---------------------|-------|
| Brute force  | Yes    | O(n)                | Ground truth. Checks every vector. |
| KD-tree      | Yes* for Euclidean/Manhattan | O(log n) avg, degrades in high dims | See known limitation below |
| HNSW         | No (approximate) | O(log n) | Graph-based, the same family of algorithm used by production vector DBs |

`/benchmark` runs the same query against all three and returns their
timings side by side. `/hnsw-info` exposes the graph structure (layers,
edges, entry point) so the UI can visualize how HNSW's "highway" layers
work.

### HNSW, briefly

Every inserted vector is randomly assigned a maximum layer using an
exponentially decaying distribution (`level = floor(-ln(random) * 1/ln(M))`),
so most vectors only exist at layer 0 and a shrinking fraction climb
higher. Search starts at the top layer's entry point and greedily walks
toward the query at each layer before dropping down, which is what
gives HNSW its logarithmic-ish search time instead of a linear scan.
Implementation follows Malkov & Yashunin's 2016 paper.

### Known limitation: KD-tree + cosine distance is not exact

While writing the test suite I found and confirmed this: the KD-tree's
branch-pruning check (`abs(diff) < -heap[0][0]`) assumes that a single
axis difference lower-bounds the true distance between two points. That
holds for Euclidean and Manhattan distance, but **not** for cosine
distance, which depends on the angle between vectors, not a per-axis
difference. So `algo=kdtree&metric=cosine` can silently prune away a
branch that actually contains a closer neighbor.

`tests/test_core.py::test_kdtree_cosine_is_only_approximate_known_limitation`
measures this directly: KD-tree + cosine recalls roughly 70-90% of true
top-10 neighbors on a random 200-point dataset, not 100%. KD-tree +
Euclidean/Manhattan match brute force exactly, every time (also tested).

**Practical takeaway:** if you need guaranteed-correct cosine search,
use `algo=bruteforce` (exact, fine for small datasets) or accept HNSW's
benchmarked approximate recall — don't rely on KD-tree for cosine.

## Architecture

```
main.py            Flask routes only — no algorithm logic lives here
core.py             BruteForce, KDTree, HNSW, VectorDB, DocumentDB, distance metrics
ai_providers.py      Ollama -> sentence-transformers -> Groq cascade (see below)
persistence.py       JSON save/load so data survives a restart
demo_data.py         the 20 seed vectors (cs/math/food/sports clusters)
index.html           single-page UI, talks to the Flask API
tests/               pytest suite (51 tests) — see "Testing" below
```

This started as a Python port of a C++ prototype. Since then it's been
split into modules, given disk persistence, given a cloud-fallback path
for the RAG pipeline, and covered with tests — the algorithm code itself
(HNSW/KD-tree/brute-force) is a faithful reimplementation of the
original design.

## The RAG pipeline, and why it doesn't require Ollama

The original design used [Ollama](https://ollama.com) running locally
for both embeddings (`nomic-embed-text`) and generation (`llama3.2`) —
great for local/private use, but it means a hosted public demo would
require every visitor to have Ollama installed, which defeats the point
of a live link.

`ai_providers.py` cascades through three backends:

1. **Ollama** (if reachable) — used first if you have it running locally.
2. **sentence-transformers** (`all-MiniLM-L6-v2`, runs in-process, no
   API key) — used for embeddings if Ollama isn't available.
3. **Groq** (hosted, OpenAI-compatible API, free tier) — used for
   generation if Ollama isn't available.

Once an embedding backend is used successfully, it's **pinned** for the
rest of the process's life — different backends produce different
dimensional embeddings (384 for MiniLM vs. 768 for nomic-embed-text),
and silently mixing them would corrupt the vector index. This is
covered by `test_embed_backend_pins_after_first_success`.

`/status` reports which backends are currently available/configured.

### Setting up each backend

**Ollama (local, full privacy):**
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
# then just run the app — it auto-detects Ollama on localhost:11434
```

**sentence-transformers (no API key, runs locally):**
```bash
pip install -r requirements-optional.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Groq (hosted generation, free tier):**
```bash
export GROQ_API_KEY=your-key-here   # from console.groq.com
```

You only need Ollama, OR sentence-transformers + Groq — not both.

## Running locally

```bash
pip install -r requirements.txt
python main.py
# → http://localhost:8080
```

Data (the demo vectors, anything you insert, and any documents you add
for RAG) is saved to `data/*.json` and reloaded automatically on the
next start.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

51 tests across four files:
- `test_core.py` — distance metrics, brute force correctness, KD-tree
  exactness (and its cosine limitation, above), **HNSW recall@k against
  brute-force ground truth** (not just "does it run" — actual measured
  recall), text chunking.
- `test_persistence.py` — save/load round-trips, id preservation across
  a restart, corrupt-file handling.
- `test_ai_providers.py` — the Ollama/sentence-transformers/Groq
  cascade logic, fully mocked (no real network calls).
- `test_api.py` — the Flask routes end to end via the test client.

## Running with Docker

```bash
# lean build (Ollama/Groq only, no local-embedding fallback)
docker build -t vectordb .
docker run -p 8080:8080 -e GROQ_API_KEY=... -v $(pwd)/data:/app/data vectordb

# with the sentence-transformers fallback baked in
docker build --build-arg INSTALL_ST=true -t vectordb .
```

Or with docker-compose:
```bash
docker compose up app                        # cloud-fallback mode
docker compose --profile local-llm up         # + a local Ollama container
```

**Note on scaling:** the app's state (the vector index and documents)
lives in process memory, not a shared database, so it has to run as a
single process — `Dockerfile` deliberately pins gunicorn to
`--workers 1` (with threads for concurrency). I found this the hard
way: with 2 worker processes, each held its own copy of the index, and
a request round-robined to the "other" worker wouldn't see an item that
was just inserted. Scaling this beyond one process would mean moving
state to something like Redis or Postgres first — a reasonable next
step, not implemented here to keep scope focused.

## Deploying a live demo

The core vector-search demo has zero external dependencies, so it'll
run on any free-tier host (Render, Railway, Fly.io). For the RAG
endpoints to work for visitors (who obviously don't have Ollama
running), set a `GROQ_API_KEY` and optionally build with
`INSTALL_ST=true` so embeddings work without any key at all.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/search?v=&k=&metric=&algo=` | GET | k-NN search over the demo vectors |
| `/insert` | POST | add a vector |
| `/delete/<id>` | DELETE | remove a vector |
| `/items` | GET | list all vectors |
| `/benchmark?v=&k=&metric=` | GET | time all three algorithms on the same query |
| `/hnsw-info` | GET | HNSW graph structure, for visualization |
| `/stats` | GET | counts, available algorithms/metrics |
| `/doc/insert` | POST | chunk + embed + store a document for RAG |
| `/doc/list` | GET | list stored documents |
| `/doc/delete/<id>` | DELETE | remove a document |
| `/doc/search` | POST | retrieve top-k relevant chunks for a question |
| `/doc/ask` | POST | full RAG: retrieve + generate an answer |
| `/status` | GET | which AI backends are currently available |

## License

MIT.

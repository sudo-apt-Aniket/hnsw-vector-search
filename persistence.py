"""
persistence.py — flat-file JSON persistence for the two in-memory
databases. Deliberately simple (no SQLite/Postgres) since the point of
this project is the index algorithms, not a storage engine — but data
now survives a restart, which the original in-memory-only design didn't.

Auto-saves after every mutation (insert/delete). For a demo-sized
dataset (tens to low thousands of items) writing the whole JSON file
back out is fast enough not to matter; if this were scaled up, this
would be the first thing to replace with incremental writes or a real
embedded DB.
"""

import json
import os
from pathlib import Path
from typing import Callable

DATA_DIR = Path(os.environ.get("VECTORDB_DATA_DIR", Path(__file__).parent / "data"))
VECTOR_STORE_FILE = DATA_DIR / "vectordb_store.json"
DOC_STORE_FILE = DATA_DIR / "docdb_store.json"


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_vector_store(db) -> None:
    _ensure_data_dir()
    items = db.all()
    payload = [
        {"id": v.id, "metadata": v.metadata, "category": v.category, "emb": v.emb}
        for v in items
    ]
    VECTOR_STORE_FILE.write_text(json.dumps(payload))


def load_vector_store(db, dist_fn: Callable) -> int:
    """Returns number of items restored. 0 if no saved file exists yet."""
    if not VECTOR_STORE_FILE.exists():
        return 0
    try:
        payload = json.loads(VECTOR_STORE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return 0
    count = 0
    for item in payload:
        db.insert(item["metadata"], item["category"], item["emb"], dist_fn, id_=item["id"])
        count += 1
    return count


def save_doc_store(doc_db) -> None:
    _ensure_data_dir()
    items = doc_db.all()
    payload = [
        {"id": d.id, "title": d.title, "text": d.text, "emb": d.emb}
        for d in items
    ]
    DOC_STORE_FILE.write_text(json.dumps(payload))


def load_doc_store(doc_db) -> int:
    if not DOC_STORE_FILE.exists():
        return 0
    try:
        payload = json.loads(DOC_STORE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return 0
    count = 0
    for item in payload:
        doc_db.insert(item["title"], item["text"], item["emb"], id_=item["id"])
        count += 1
    return count

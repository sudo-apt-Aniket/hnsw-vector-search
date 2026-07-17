"""
test_persistence.py — round-trip tests for the JSON persistence layer.
Uses a temp directory (monkeypatched onto persistence's module-level
paths) so tests never touch the project's real data/ folder.
"""

import json
import pytest

import persistence
from core import VectorDB, DocumentDB, get_dist_fn


@pytest.fixture
def tmp_persistence(tmp_path, monkeypatch):
    """Point persistence.py's file constants at a scratch directory."""
    vec_file = tmp_path / "vectordb_store.json"
    doc_file = tmp_path / "docdb_store.json"
    monkeypatch.setattr(persistence, "DATA_DIR", tmp_path)
    monkeypatch.setattr(persistence, "VECTOR_STORE_FILE", vec_file)
    monkeypatch.setattr(persistence, "DOC_STORE_FILE", doc_file)
    return tmp_path


def test_load_vector_store_returns_zero_when_no_file(tmp_persistence):
    db = VectorDB(4)
    count = persistence.load_vector_store(db, get_dist_fn("euclidean"))
    assert count == 0
    assert db.size() == 0


def test_vector_store_roundtrip_preserves_ids_and_data(tmp_persistence):
    dist = get_dist_fn("cosine")
    db1 = VectorDB(4)
    id_a = db1.insert("item A", "catA", [0.1, 0.2, 0.3, 0.4], dist)
    id_b = db1.insert("item B", "catB", [0.5, 0.6, 0.7, 0.8], dist)
    persistence.save_vector_store(db1)

    # Fresh DB instance simulating a process restart
    db2 = VectorDB(4)
    restored = persistence.load_vector_store(db2, dist)

    assert restored == 2
    assert db2.size() == 2
    all_items = {v.id: v for v in db2.all()}
    assert all_items[id_a].metadata == "item A"
    assert all_items[id_a].category == "catA"
    assert all_items[id_b].metadata == "item B"

    # next auto-id must continue after the restored max, not collide
    new_id = db2.insert("item C", "catC", [1, 1, 1, 1], dist)
    assert new_id > max(id_a, id_b)


def test_vector_store_file_is_valid_json(tmp_persistence):
    dist = get_dist_fn("euclidean")
    db = VectorDB(4)
    db.insert("solo item", "cat", [1, 2, 3, 4], dist)
    persistence.save_vector_store(db)

    raw = persistence.VECTOR_STORE_FILE.read_text()
    payload = json.loads(raw)  # must not raise
    assert isinstance(payload, list)
    assert payload[0]["metadata"] == "solo item"


def test_load_vector_store_handles_corrupt_file_gracefully(tmp_persistence):
    persistence.VECTOR_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    persistence.VECTOR_STORE_FILE.write_text("{not valid json")
    db = VectorDB(4)
    count = persistence.load_vector_store(db, get_dist_fn("euclidean"))
    assert count == 0  # doesn't crash, just treats it as empty


def test_doc_store_roundtrip(tmp_persistence):
    doc_db1 = DocumentDB()
    id_1 = doc_db1.insert("Doc One", "some text content here", [0.1] * 8)
    persistence.save_doc_store(doc_db1)

    doc_db2 = DocumentDB()
    restored = persistence.load_doc_store(doc_db2)

    assert restored == 1
    all_docs = {d.id: d for d in doc_db2.all()}
    assert all_docs[id_1].title == "Doc One"
    assert all_docs[id_1].text == "some text content here"


def test_doc_store_missing_file_returns_zero(tmp_persistence):
    doc_db = DocumentDB()
    count = persistence.load_doc_store(doc_db)
    assert count == 0

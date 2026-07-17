"""
test_api.py — exercises the actual Flask routes via the test client
(no real server/socket needed). Runs against an isolated temp data dir
(see conftest.py) so it never touches the project's real persisted data.
"""

import json
import pytest

import main as app_module


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_status_endpoint(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "demoCount" in data
    assert data["demoCount"] == 20  # freshly seeded demo set in the isolated temp dir


def test_stats_endpoint(client):
    resp = client.get("/stats")
    data = resp.get_json()
    assert data["dims"] == 16
    assert set(data["algorithms"]) == {"bruteforce", "kdtree", "hnsw"}
    assert set(data["metrics"]) == {"euclidean", "cosine", "manhattan"}


def test_search_rejects_wrong_dimension_vector(client):
    resp = client.get("/search?v=1,2,3&k=3")  # only 3 dims, need 16
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_search_valid_query_returns_results(client):
    vec = ",".join(["0.1"] * 16)
    resp = client.get(f"/search?v={vec}&k=3&algo=bruteforce&metric=euclidean")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["results"]) == 3
    assert data["algo"] == "bruteforce"
    # sorted ascending by distance
    dists = [r["distance"] for r in data["results"]]
    assert dists == sorted(dists)


def test_insert_then_appears_in_items(client):
    vec = [0.5] * 16
    resp = client.post("/insert", json={
        "metadata": "brand new test item",
        "category": "test",
        "embedding": vec,
    })
    assert resp.status_code == 200
    new_id = resp.get_json()["id"]

    items_resp = client.get("/items")
    items = items_resp.get_json()
    assert any(i["id"] == new_id and i["metadata"] == "brand new test item" for i in items)


def test_insert_rejects_wrong_dims(client):
    resp = client.post("/insert", json={
        "metadata": "bad item", "category": "test", "embedding": [1, 2, 3]
    })
    assert resp.status_code == 400


def test_delete_removes_item(client):
    vec = [0.3] * 16
    insert_resp = client.post("/insert", json={
        "metadata": "to be deleted", "category": "test", "embedding": vec
    })
    new_id = insert_resp.get_json()["id"]

    del_resp = client.delete(f"/delete/{new_id}")
    assert del_resp.get_json()["ok"] is True

    items = client.get("/items").get_json()
    assert not any(i["id"] == new_id for i in items)


def test_delete_nonexistent_id_returns_false(client):
    resp = client.delete("/delete/999999")
    assert resp.get_json()["ok"] is False


def test_benchmark_endpoint_returns_timings_for_all_three(client):
    vec = ",".join(["0.2"] * 16)
    resp = client.get(f"/benchmark?v={vec}&k=5")
    data = resp.get_json()
    assert "bruteforceUs" in data
    assert "kdtreeUs" in data
    assert "hnswUs" in data
    assert data["itemCount"] >= 20


def test_hnsw_info_endpoint_structure(client):
    resp = client.get("/hnsw-info")
    data = resp.get_json()
    assert "nodeCount" in data
    assert "nodesPerLayer" in data
    assert data["nodeCount"] >= 20


def test_doc_insert_without_any_ai_backend_gives_clean_503(client):
    resp = client.post("/doc/insert", json={
        "title": "Some Doc", "text": "content that needs embedding"
    })
    # In the test environment there's no Ollama and (usually) no
    # sentence-transformers/Groq configured, so this should fail
    # gracefully with a clear error, not a 500 crash.
    assert resp.status_code in (503, 200)  # 200 only if a backend happens to be configured
    if resp.status_code == 503:
        assert "error" in resp.get_json()


def test_doc_insert_missing_fields_returns_400(client):
    resp = client.post("/doc/insert", json={"title": "", "text": ""})
    assert resp.status_code == 400


def test_cors_headers_present(client):
    resp = client.get("/stats")
    assert resp.headers.get("Access-Control-Allow-Origin") == "*"


def test_options_request_returns_2xx_with_cors_headers(client):
    # Flask auto-generates the OPTIONS response per route; we just need
    # our CORS headers to be present on it (applied via after_request).
    resp = client.options("/stats")
    assert resp.status_code == 200
    assert resp.headers.get("Access-Control-Allow-Origin") == "*"

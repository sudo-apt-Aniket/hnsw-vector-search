"""
test_core.py — correctness tests for the algorithms in core.py.

The most important test here is test_hnsw_recall: HNSW is an
*approximate* nearest-neighbour index, so "correct" doesn't mean
"identical to brute force" — it means "recall stays high enough to be
useful". We measure that directly instead of assuming it.
"""

import math
import random

import pytest

from core import (
    BruteForce, KDTree, HNSW, VectorDB,
    euclidean, cosine, manhattan, get_dist_fn, chunk_text,
    VectorItem,
)


def random_vectors(n, dims, seed=0):
    rng = random.Random(seed)
    return [[rng.uniform(-1, 1) for _ in range(dims)] for _ in range(n)]


# ---------------------------------------------------------------------
#  Distance metrics
# ---------------------------------------------------------------------

def test_euclidean_known_value():
    assert euclidean([0, 0], [3, 4]) == pytest.approx(5.0)

def test_manhattan_known_value():
    assert manhattan([0, 0], [3, 4]) == pytest.approx(7.0)

def test_cosine_identical_vectors_is_zero_distance():
    v = [1.0, 2.0, 3.0]
    assert cosine(v, v) == pytest.approx(0.0, abs=1e-9)

def test_cosine_orthogonal_vectors_is_one():
    assert cosine([1, 0], [0, 1]) == pytest.approx(1.0)

def test_cosine_zero_vector_handled_without_crash():
    # na or nb < 1e-9 branch — must not raise ZeroDivisionError
    assert cosine([0, 0, 0], [1, 2, 3]) == 1.0

def test_get_dist_fn_returns_correct_function():
    assert get_dist_fn("cosine") is cosine
    assert get_dist_fn("manhattan") is manhattan
    assert get_dist_fn("euclidean") is euclidean
    assert get_dist_fn("unknown-metric-defaults-to-euclidean") is euclidean


# ---------------------------------------------------------------------
#  Brute force — this IS the ground truth, so just sanity-check it
# ---------------------------------------------------------------------

def test_bruteforce_returns_k_nearest_sorted():
    bf = BruteForce()
    vecs = random_vectors(50, 8, seed=1)
    for i, v in enumerate(vecs):
        bf.insert(VectorItem(i, f"item{i}", "cat", v))

    query = vecs[10]  # exact match should come back as the top hit
    results = bf.knn(query, k=5, dist=euclidean)

    assert len(results) == 5
    assert results[0][1] == 10
    assert results[0][0] == pytest.approx(0.0, abs=1e-9)
    # sorted ascending by distance
    dists = [d for d, _ in results]
    assert dists == sorted(dists)

def test_bruteforce_remove():
    bf = BruteForce()
    bf.insert(VectorItem(1, "a", "cat", [0, 0]))
    bf.insert(VectorItem(2, "b", "cat", [1, 1]))
    bf.remove(1)
    ids = [v.id for v in bf.items]
    assert ids == [2]


# ---------------------------------------------------------------------
#  KD-Tree — should be EXACT (not approximate) for these dimensions,
#  so it must match brute force precisely.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("metric_name", ["euclidean", "manhattan"])
def test_kdtree_matches_bruteforce_exactly(metric_name):
    """KD-tree branch-pruning (`abs(diff) < -heap[0][0]`) is only valid
    for metrics where a single axis difference lower-bounds the true
    distance — true for Euclidean and Manhattan (Minkowski family)."""
    dist = get_dist_fn(metric_name)
    dims = 6
    vecs = random_vectors(200, dims, seed=2)

    bf = BruteForce()
    kdt = KDTree(dims)
    for i, v in enumerate(vecs):
        item = VectorItem(i, f"item{i}", "cat", v)
        bf.insert(item)
        kdt.insert(item)

    query = [random.Random(99).uniform(-1, 1) for _ in range(dims)]
    bf_ids = [id_ for _, id_ in bf.knn(query, 10, dist)]
    kdt_ids = [id_ for _, id_ in kdt.knn(query, 10, dist)]

    assert bf_ids == kdt_ids, (
        f"KD-tree ({metric_name}) diverged from brute-force ground truth"
    )


def test_kdtree_cosine_is_only_approximate_known_limitation():
    """KNOWN LIMITATION, documented via test rather than silently
    trusted: cosine distance does not decompose additively across axes,
    so the KD-tree's single-axis pruning bound is not a valid lower
    bound on cosine distance. The tree can therefore prune away a
    branch that actually contains a closer neighbour.

    This means /search?algo=kdtree&metric=cosine is *approximate*, not
    exact, despite KD-tree being exact for euclidean/manhattan. For
    guaranteed-correct cosine search, use algo=bruteforce (exact) or
    accept HNSW's approximate-but-benchmarked recall instead.

    We assert recall stays reasonably high rather than asserting exact
    equality, because exact equality is not a real property of this
    combination and asserting it would just be documenting a bug as a
    passing test.
    """
    dist = cosine
    dims = 6
    vecs = random_vectors(200, dims, seed=2)

    bf = BruteForce()
    kdt = KDTree(dims)
    for i, v in enumerate(vecs):
        item = VectorItem(i, f"item{i}", "cat", v)
        bf.insert(item)
        kdt.insert(item)

    recalls = []
    for qseed in range(10):
        query = [random.Random(qseed).uniform(-1, 1) for _ in range(dims)]
        bf_ids = {id_ for _, id_ in bf.knn(query, 10, dist)}
        kdt_ids = {id_ for _, id_ in kdt.knn(query, 10, dist)}
        recalls.append(len(bf_ids & kdt_ids) / 10)

    avg_recall = sum(recalls) / len(recalls)
    assert 0.5 <= avg_recall < 1.0, (
        f"expected KD-tree+cosine to be imperfect-but-decent (~0.5-0.95), got {avg_recall:.2f} — "
        "if this is now 1.0 the pruning logic may have changed; if it's <0.5 something else broke"
    )

def test_kdtree_rebuild_after_removal():
    dims = 4
    vecs = random_vectors(30, dims, seed=3)
    items = [VectorItem(i, f"item{i}", "cat", v) for i, v in enumerate(vecs)]

    kdt = KDTree(dims)
    for item in items:
        kdt.insert(item)

    remaining = items[1:]  # simulate removing id 0
    kdt.rebuild(remaining)

    query = vecs[5]
    result_ids = {id_ for _, id_ in kdt.knn(query, 5, euclidean)}
    assert 0 not in result_ids


# ---------------------------------------------------------------------
#  HNSW — approximate. We test RECALL against brute-force ground truth,
#  not exact match, because that's the actual correctness contract of
#  an ANN index.
# ---------------------------------------------------------------------

def test_hnsw_recall_against_bruteforce():
    dims = 16
    n = 300
    k = 10
    vecs = random_vectors(n, dims, seed=7)

    bf = BruteForce()
    hnsw = HNSW(M=16, ef_construction=200, seed=42)
    for i, v in enumerate(vecs):
        item = VectorItem(i, f"item{i}", "cat", v)
        bf.insert(item)
        hnsw.insert(item, euclidean)

    rng = random.Random(123)
    queries = [[rng.uniform(-1, 1) for _ in range(dims)] for _ in range(20)]

    total_recall = 0.0
    for q in queries:
        ground_truth = {id_ for _, id_ in bf.knn(q, k, euclidean)}
        approx = {id_ for _, id_ in hnsw.knn(q, k, ef=50, dist=euclidean)}
        recall = len(ground_truth & approx) / k
        total_recall += recall

    avg_recall = total_recall / len(queries)
    # HNSW with these params on a dataset this small should recall the
    # true top-k the large majority of the time. This threshold is a
    # regression guard, not a claim about production-scale behaviour.
    assert avg_recall >= 0.85, f"HNSW recall@{k} dropped to {avg_recall:.2f}"

def test_hnsw_remove_drops_node_from_results():
    dims = 8
    vecs = random_vectors(50, dims, seed=4)
    hnsw = HNSW(M=16, ef_construction=200, seed=42)
    for i, v in enumerate(vecs):
        hnsw.insert(VectorItem(i, f"item{i}", "cat", v), euclidean)

    hnsw.remove(5)
    query = vecs[5]
    result_ids = {id_ for _, id_ in hnsw.knn(query, 10, ef=50, dist=euclidean)}
    assert 5 not in result_ids
    assert 5 not in hnsw.graph

def test_hnsw_empty_index_returns_empty():
    hnsw = HNSW()
    assert hnsw.knn([0.0] * 4, k=5, ef=50, dist=euclidean) == []

def test_hnsw_get_info_structure():
    dims = 4
    hnsw = HNSW(M=16, ef_construction=200, seed=42)
    for i, v in enumerate(random_vectors(15, dims, seed=5)):
        hnsw.insert(VectorItem(i, f"item{i}", "cat", v), euclidean)
    info = hnsw.get_info()
    assert info["nodeCount"] == 15
    assert len(info["nodesPerLayer"]) == info["topLayer"] + 1
    assert sum(info["nodesPerLayer"]) >= 15  # nodes can appear in multiple layers


# ---------------------------------------------------------------------
#  VectorDB — the integration point that keeps all three structures in sync
# ---------------------------------------------------------------------

def test_vectordb_all_three_algorithms_agree_on_top1():
    """For an unambiguous nearest neighbour (near-exact match), brute
    force, KD-tree, and HNSW should all agree on the top hit."""
    dims = 16
    db = VectorDB(dims)
    dist = get_dist_fn("cosine")
    vecs = random_vectors(60, dims, seed=8)
    for i, v in enumerate(vecs):
        db.insert(f"item{i}", "cat", v, dist)

    target = vecs[30]
    query = [x + 0.0001 for x in target]  # tiny perturbation, still closest to itself

    bf_top = db.search(query, 1, "cosine", "bruteforce")["results"][0]["id"]
    kd_top = db.search(query, 1, "cosine", "kdtree")["results"][0]["id"]
    hnsw_top = db.search(query, 1, "cosine", "hnsw")["results"][0]["id"]

    # ids are 1-indexed (VectorDB assigns starting at 1)
    expected_id = 31
    assert bf_top == expected_id
    assert kd_top == expected_id
    assert hnsw_top == expected_id

def test_vectordb_insert_then_remove():
    dims = 4
    db = VectorDB(dims)
    dist = get_dist_fn("euclidean")
    id_ = db.insert("test item", "cat", [1, 2, 3, 4], dist)
    assert db.size() == 1
    assert db.remove(id_) is True
    assert db.size() == 0
    assert db.remove(id_) is False  # already gone

def test_vectordb_insert_with_explicit_id_for_restore():
    """Persistence restore relies on being able to re-insert with a
    specific id and have subsequent auto-ids continue afterward."""
    dims = 4
    db = VectorDB(dims)
    dist = get_dist_fn("euclidean")
    db.insert("restored item", "cat", [1, 2, 3, 4], dist, id_=99)
    new_id = db.insert("fresh item", "cat", [5, 6, 7, 8], dist)
    assert new_id == 100  # continues after the restored id, doesn't collide


# ---------------------------------------------------------------------
#  Text chunker
# ---------------------------------------------------------------------

def test_chunk_text_short_text_returns_single_chunk():
    text = "short text under the limit"
    assert chunk_text(text, chunk_words=250, overlap_words=30) == [text]

def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("", 250, 30) == []

def test_chunk_text_splits_long_text_with_overlap():
    words = [f"word{i}" for i in range(600)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_words=250, overlap_words=30)
    assert len(chunks) >= 2
    # consecutive chunks should share the overlap region
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-30:] == second_words[:30]

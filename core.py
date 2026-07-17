"""
core.py — the actual "vector database" logic: distance metrics, three
search structures (brute force, KD-tree, HNSW), and two collections built
on top of them (VectorDB for the 16D demo dataset, DocumentDB for RAG).

This is a from-scratch reimplementation of HNSW (Malkov & Yashunin, 2016)
without any external ANN library, so the algorithmic behaviour can be
inspected/tested directly.
"""

import math
import time
import heapq
import random
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable

# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]

DistFn = Callable[[List[float], List[float]], float]

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)

def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE  — exact, O(n) per query. Ground truth for the others.
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        results = [(dist(q, v.emb), v.id) for v in self.items]
        results.sort()
        return results[:k]

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]

# =====================================================================
#  KD-TREE  — exact for low dimensions, degrades as dims grow.
# =====================================================================

class KDNode:
    def __init__(self, item: VectorItem):
        self.item = item
        self.left = None
        self.right = None

class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root = None

    def _insert(self, node: Optional[KDNode], v: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(v)
        ax = depth % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left = self._insert(node.left, v, depth + 1)
        else:
            node.right = self._insert(node.right, v, depth + 1)
        return node

    def insert(self, v: VectorItem):
        self.root = self._insert(self.root, v, 0)

    def _knn(self, node: Optional[KDNode], q: List[float], k: int, depth: int,
             dist: DistFn, heap: list):
        if node is None:
            return
        dn = dist(q, node.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)

        ax = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer, q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        heap = []
        self._knn(self.root, q, k, 0, dist, heap)
        results = [(-neg_d, id_) for neg_d, id_ in heap]
        results.sort()
        return results

    def rebuild(self, items: List[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World (approximate nearest
#  neighbour). See: Malkov & Yashunin, "Efficient and robust approximate
#  nearest neighbor search using Hierarchical Navigable Small World
#  graphs", 2016.
# =====================================================================

class HNSWNode:
    def __init__(self, item: VectorItem, max_layer: int):
        self.item = item
        self.max_layer = max_layer
        self.neighbors: List[List[int]] = [[] for _ in range(max_layer + 1)]

class HNSW:
    def __init__(self, M: int = 16, ef_construction: int = 200, seed: int = 42):
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)
        self.graph: Dict[int, HNSWNode] = {}
        self.entry_point = -1
        self.top_layer = -1
        self._rng = random.Random(seed)

    def _rand_level(self) -> int:
        # Exponentially-decaying level assignment: most nodes stay at
        # layer 0, a shrinking fraction climb higher, giving the
        # "highway" structure that makes greedy search close to O(log n).
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q: List[float], ep: int, ef: int, layer: int,
                       dist: DistFn) -> List[Tuple[float, int]]:
        visited = {ep}
        d0 = dist(q, self.graph[ep].item.emb)
        cands = [(d0, ep)]           # min-heap: candidates still to expand
        found = [(-d0, ep)]          # max-heap (negated): best `ef` found so far

        while cands:
            cd, cid = heapq.heappop(cands)
            worst = -found[0][0]
            if len(found) >= ef and cd > worst:
                break
            node = self.graph.get(cid)
            if node is None or layer >= len(node.neighbors):
                continue
            for nid in node.neighbors[layer]:
                if nid in visited or nid not in self.graph:
                    continue
                visited.add(nid)
                nd = dist(q, self.graph[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-neg_d, id_) for neg_d, id_ in found]
        result.sort()
        return result

    def _select_neighbors(self, candidates: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [id_ for _, id_ in candidates[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn):
        id_ = item.id
        lvl = self._rand_level()
        node = HNSWNode(item, lvl)
        self.graph[id_] = node

        if self.entry_point == -1:
            self.entry_point = id_
            self.top_layer = lvl
            return

        ep = self.entry_point
        for lc in range(self.top_layer, lvl, -1):
            ep_node = self.graph.get(ep)
            if ep_node and lc < len(ep_node.neighbors):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_construction, lc, dist)
            maxM = self.M0 if lc == 0 else self.M
            sel = self._select_neighbors(W, maxM)
            while len(node.neighbors) <= lc:
                node.neighbors.append([])
            node.neighbors[lc] = sel

            for nid in sel:
                nbr = self.graph.get(nid)
                if nbr is None:
                    continue
                while len(nbr.neighbors) <= lc:
                    nbr.neighbors.append([])
                nbr.neighbors[lc].append(id_)
                if len(nbr.neighbors[lc]) > maxM:
                    pairs = []
                    for c in nbr.neighbors[lc]:
                        cn = self.graph.get(c)
                        if cn:
                            pairs.append((dist(nbr.item.emb, cn.item.emb), c))
                    pairs.sort()
                    nbr.neighbors[lc] = [c for _, c in pairs[:maxM]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_point = id_

    def knn(self, q: List[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_point == -1:
            return []
        ep = self.entry_point
        for lc in range(self.top_layer, 0, -1):
            ep_node = self.graph.get(ep)
            if ep_node and lc < len(ep_node.neighbors):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id_: int):
        if id_ not in self.graph:
            return
        for node in self.graph.values():
            for layer in node.neighbors:
                if id_ in layer:
                    layer.remove(id_)
        if self.entry_point == id_:
            self.entry_point = -1
            for nid in self.graph:
                if nid != id_:
                    self.entry_point = nid
                    break
        del self.graph[id_]

    def get_info(self) -> dict:
        top = self.top_layer
        max_l = max(top + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []
        for id_, node in self.graph.items():
            nodes.append({
                "id": id_,
                "metadata": node.item.metadata,
                "category": node.item.category,
                "maxLyr": node.max_layer
            })
            for lc in range(min(node.max_layer + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(node.neighbors):
                    for nid in node.neighbors[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})
        return {
            "topLayer": top,
            "nodeCount": len(self.graph),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges
        }

    def size(self) -> int:
        return len(self.graph)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index — bruteforce + kdtree + hnsw, kept
#  in sync so /search can compare all three algorithms on demand)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self._store: Dict[int, VectorItem] = {}
        self._bf = BruteForce()
        self._kdt = KDTree(dims)
        self._hnsw = HNSW(16, 200)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(self, metadata: str, category: str, emb: List[float],
               dist: DistFn, id_: Optional[int] = None) -> int:
        """Insert a new item. Pass id_ explicitly when restoring from
        persisted state so ids survive a restart unchanged."""
        with self._lock:
            new_id = id_ if id_ is not None else self._next_id
            v = VectorItem(new_id, metadata, category, emb)
            self._next_id = max(self._next_id, new_id + 1)
            self._store[v.id] = v
            self._bf.insert(v)
            self._kdt.insert(v)
            self._hnsw.insert(v, dist)
            return v.id

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self._store:
                return False
            del self._store[id_]
            self._bf.remove(id_)
            self._hnsw.remove(id_)
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self._bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self._kdt.knn(q, k, dfn)
            else:
                raw = self._hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, id_ in raw:
                if id_ in self._store:
                    v = self._store[id_]
                    hits.append({"id": v.id, "metadata": v.metadata,
                                 "category": v.category, "distance": round(d, 6),
                                 "embedding": v.emb})
            return {"results": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, q: List[float], k: int, metric: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)

            def time_fn(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            return {
                "bruteforceUs": time_fn(lambda: self._bf.knn(q, k, dfn)),
                "kdtreeUs": time_fn(lambda: self._kdt.knn(q, k, dfn)),
                "hnswUs": time_fn(lambda: self._hnsw.knn(q, k, 50, dfn)),
                "itemCount": len(self._store)
            }

    def all(self) -> List[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self._hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self._store)

# =====================================================================
#  DOCUMENT DATABASE  — HNSW over real embeddings, used by the RAG
#  endpoints (/doc/insert, /doc/search, /doc/ask)
# =====================================================================

class DocumentDB:
    def __init__(self):
        self._store: Dict[int, DocItem] = {}
        self._hnsw = HNSW(16, 200)
        self._bf = BruteForce()
        self._lock = threading.Lock()
        self._next_id = 1
        self._dims = 0

    def insert(self, title: str, text: str, emb: List[float],
               id_: Optional[int] = None) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(emb)
            new_id = id_ if id_ is not None else self._next_id
            item = DocItem(new_id, title, text, emb)
            self._next_id = max(self._next_id, new_id + 1)
            self._store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self._hnsw.insert(vi, cosine)
            self._bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < 10:
                raw = self._bf.knn(q, k, cosine)
            else:
                raw = self._hnsw.knn(q, k, 50, cosine)
            out = []
            for d, id_ in raw:
                if id_ in self._store and d <= max_dist:
                    out.append((d, self._store[id_]))
            return out

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self._store:
                return False
            del self._store[id_]
            self._hnsw.remove(id_)
            self._bf.remove(id_)
            return True

    def all(self) -> List[DocItem]:
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def get_dims(self) -> int:
        return self._dims

# =====================================================================
#  TEXT CHUNKER  (sliding window with overlap, for RAG ingestion)
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
        i += step
    return chunks

"""
worker.py — Core processing logic. Called by server.py's thread.
No boto3, no AWS, no RQ. Pure Python.
"""
import json, os, sys, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from db import init_db, save_submission, complete_job, fail_job
from embedder import get_embedder, CATEGORY_NAMES, COLOR_PALETTE
from scorer import compute_all_scores

_base_data = None
_base_embeddings = None
_base_nodes = None
_cluster_centroids = {}
_all_distances_cache = {}
_cluster_groups = {}  # category_name -> list of indices


def _load_base_data():
    global _base_data, _base_embeddings, _base_nodes
    global _cluster_centroids, _all_distances_cache, _cluster_groups

    if _base_data is not None:
        return

    data_path = os.path.join(os.path.dirname(__file__), "data.json")
    print(f"[worker] Loading base data from {data_path}...")

    with open(data_path) as f:
        _base_data = json.load(f)

    _base_nodes = [n for n in _base_data["nodes"] if n["id"] != "ORIGIN"]

    embedder = get_embedder()
    abstracts = [n["abstract"] for n in _base_nodes]
    print(f"[worker] Pre-embedding {len(abstracts)} base abstracts...")
    _base_embeddings = embedder.embed(abstracts)

    # Route every base node through embedder — consistent naming
    print("[worker] Pre-computing cluster centroids...")
    for i in range(len(_base_nodes)):
        cat_name, _, _ = embedder.route_to_cluster(_base_embeddings[i])
        _cluster_groups.setdefault(cat_name, []).append(i)

    for cat_name, indices in _cluster_groups.items():
        vecs = _base_embeddings[indices]
        centroid = np.mean(vecs, axis=0)
        _cluster_centroids[cat_name] = centroid
        dists = [float(np.linalg.norm(_base_embeddings[i] - centroid)) for i in indices]
        _all_distances_cache[cat_name] = dists

    print("[worker] Cluster sizes:")
    for cat, indices in _cluster_groups.items():
        print(f"  {cat}: {len(indices)} projects")
    print("[worker] Ready to process jobs.")


def process_submission(job_id: str, abstract: str, name: str, stack: str):
    init_db()
    _load_base_data()

    try:
        embedder = get_embedder()

        # 1. Embed
        new_embedding = embedder.embed([abstract])[0]

        # 2. Route to cluster
        cluster_topic, color, _ = embedder.route_to_cluster(new_embedding)

        # 3. Binary fingerprint
        threshold = np.median(_base_embeddings, axis=0)
        binary_fp = (new_embedding > threshold).astype(int)

        # 4. Clone detection — within cluster only
        cluster_indices = _cluster_groups.get(cluster_topic, [])

        if cluster_indices:
            cluster_embeddings = _base_embeddings[cluster_indices]
            sims_in_cluster = embedder.cosine_similarities(new_embedding, cluster_embeddings)
            max_sim_idx_local = int(np.argmax(sims_in_cluster))
            max_sim = float(sims_in_cluster[max_sim_idx_local])
            nearest_node = _base_nodes[cluster_indices[max_sim_idx_local]]
        else:
            sims = embedder.cosine_similarities(new_embedding, _base_embeddings)
            max_sim_idx = int(np.argmax(sims))
            max_sim = float(sims[max_sim_idx])
            nearest_node = _base_nodes[max_sim_idx]

        # 5. Cluster size, centroid, distances
        cluster_size = len(cluster_indices)
        centroid = _cluster_centroids.get(cluster_topic, np.mean(_base_embeddings, axis=0))
        all_distances = _all_distances_cache.get(cluster_topic, [])

        # 6. Average cluster size — so dark horse is relative not hardcoded
        avg_cluster_size = len(_base_nodes) / max(len(_cluster_groups), 1)

        # 8. 3D position near cluster (calculated first to get magnitude)
        same_cluster_nodes = [_base_nodes[i] for i in cluster_indices]
        if same_cluster_nodes:
            spread = 50
            x = float(np.mean([n["x"] for n in same_cluster_nodes]) + random.uniform(-spread, spread))
            y = float(np.mean([n["y"] for n in same_cluster_nodes]) + random.uniform(-spread, spread))
            z = float(np.mean([n["z"] for n in same_cluster_nodes]) + random.uniform(-spread, spread))
        else:
            x, y, z = [float(random.uniform(-150, 150)) for _ in range(3)]

        magnitude = float(np.linalg.norm([x, y, z]))
        all_magnitudes = [float(n.get("uniqueness", 0)) for n in _base_nodes]

        # 7. Compute all scores
        scores = compute_all_scores(
            abstract=abstract,
            stack=stack,
            embedding=new_embedding,
            centroid=centroid,
            all_centroid_distances=all_distances,
            max_cosine_similarity=max_sim,
            nearest_node_name=nearest_node["name"],
            cluster_size=cluster_size,
            avg_cluster_size=avg_cluster_size,
            magnitude=magnitude,
            all_magnitudes=all_magnitudes
        )

        node = {
            "id": job_id,
            "name": name,
            "abstract": abstract,
            "stack": stack,
            "cluster_topic": cluster_topic,
            "color": color,
            "val": 7,
            "x": x, "y": y, "z": z,
            "uniqueness": float(round(magnitude, 1)),
            "is_live": 1,
            "fingerprint_preview": binary_fp[:32].tolist(),
            **scores
        }

        save_submission(node)
        complete_job(job_id, node)
        print(f"[worker] ✓ {name} → {cluster_topic} | {scores['status']} | idea={scores['idea_score']} viability={scores['viability_score']} sim={scores['clone_similarity_pct']}% cluster_size={cluster_size} avg={avg_cluster_size:.1f}")
        return node

    except Exception as e:
        print(f"[worker] ✗ Job {job_id} failed: {e}")
        import traceback; traceback.print_exc()
        fail_job(job_id, str(e))
        raise"""
worker.py — Core processing logic. Called by server.py's thread.
No boto3, no AWS, no RQ. Pure Python.
"""
import json, os, sys, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from db import init_db, save_submission, complete_job, fail_job
from embedder import get_embedder, CATEGORY_NAMES, COLOR_PALETTE
from scorer import compute_all_scores

_base_data = None
_base_embeddings = None
_base_nodes = None
_cluster_centroids = {}
_all_distances_cache = {}
_cluster_groups = {}  # category_name -> list of indices


def _load_base_data():
    global _base_data, _base_embeddings, _base_nodes
    global _cluster_centroids, _all_distances_cache, _cluster_groups

    if _base_data is not None:
        return

    data_path = os.path.join(os.path.dirname(__file__), "data.json")
    print(f"[worker] Loading base data from {data_path}...")

    with open(data_path) as f:
        _base_data = json.load(f)

    _base_nodes = [n for n in _base_data["nodes"] if n["id"] != "ORIGIN"]

    embedder = get_embedder()
    abstracts = [n["abstract"] for n in _base_nodes]
    print(f"[worker] Pre-embedding {len(abstracts)} base abstracts...")
    _base_embeddings = embedder.embed(abstracts)

    # Route every base node through embedder — consistent naming
    print("[worker] Pre-computing cluster centroids...")
    for i in range(len(_base_nodes)):
        cat_name, _, _ = embedder.route_to_cluster(_base_embeddings[i])
        _cluster_groups.setdefault(cat_name, []).append(i)

    for cat_name, indices in _cluster_groups.items():
        vecs = _base_embeddings[indices]
        centroid = np.mean(vecs, axis=0)
        _cluster_centroids[cat_name] = centroid
        dists = [float(np.linalg.norm(_base_embeddings[i] - centroid)) for i in indices]
        _all_distances_cache[cat_name] = dists

    print("[worker] Cluster sizes:")
    for cat, indices in _cluster_groups.items():
        print(f"  {cat}: {len(indices)} projects")
    print("[worker] Ready to process jobs.")


def process_submission(job_id: str, abstract: str, name: str, stack: str):
    init_db()
    _load_base_data()

    try:
        embedder = get_embedder()

        # 1. Embed
        new_embedding = embedder.embed([abstract])[0]

        # 2. Route to cluster
        cluster_topic, color, _ = embedder.route_to_cluster(new_embedding)

        # 3. Binary fingerprint
        threshold = np.median(_base_embeddings, axis=0)
        binary_fp = (new_embedding > threshold).astype(int)

        # 4. Clone detection — within cluster only
        cluster_indices = _cluster_groups.get(cluster_topic, [])

        if cluster_indices:
            cluster_embeddings = _base_embeddings[cluster_indices]
            sims_in_cluster = embedder.cosine_similarities(new_embedding, cluster_embeddings)
            max_sim_idx_local = int(np.argmax(sims_in_cluster))
            max_sim = float(sims_in_cluster[max_sim_idx_local])
            nearest_node = _base_nodes[cluster_indices[max_sim_idx_local]]
        else:
            sims = embedder.cosine_similarities(new_embedding, _base_embeddings)
            max_sim_idx = int(np.argmax(sims))
            max_sim = float(sims[max_sim_idx])
            nearest_node = _base_nodes[max_sim_idx]

        # 5. Cluster size, centroid, distances
        cluster_size = len(cluster_indices)
        centroid = _cluster_centroids.get(cluster_topic, np.mean(_base_embeddings, axis=0))
        all_distances = _all_distances_cache.get(cluster_topic, [])

        # 6. Average cluster size — so dark horse is relative not hardcoded
        avg_cluster_size = len(_base_nodes) / max(len(_cluster_groups), 1)

        # 7. Compute all scores
        scores = compute_all_scores(
            abstract=abstract,
            stack=stack,
            embedding=new_embedding,
            centroid=centroid,
            all_centroid_distances=all_distances,
            max_cosine_similarity=max_sim,
            nearest_node_name=nearest_node["name"],
            cluster_size=cluster_size,
            avg_cluster_size=avg_cluster_size,
        )

        # 8. 3D position near cluster
        same_cluster_nodes = [_base_nodes[i] for i in cluster_indices]
        if same_cluster_nodes:
            spread = 50
            x = float(np.mean([n["x"] for n in same_cluster_nodes]) + random.uniform(-spread, spread))
            y = float(np.mean([n["y"] for n in same_cluster_nodes]) + random.uniform(-spread, spread))
            z = float(np.mean([n["z"] for n in same_cluster_nodes]) + random.uniform(-spread, spread))
        else:
            x, y, z = [float(random.uniform(-150, 150)) for _ in range(3)]

        node = {
            "id": job_id,
            "name": name,
            "abstract": abstract,
            "stack": stack,
            "cluster_topic": cluster_topic,
            "color": color,
            "val": 7,
            "x": x, "y": y, "z": z,
            "uniqueness": float(round(np.linalg.norm([x, y, z]), 1)),
            "is_live": 1,
            "fingerprint_preview": binary_fp[:32].tolist(),
            **scores
        }

        save_submission(node)
        complete_job(job_id, node)
        print(f"[worker] ✓ {name} → {cluster_topic} | {scores['status']} | idea={scores['idea_score']} viability={scores['viability_score']} sim={scores['clone_similarity_pct']}% cluster_size={cluster_size} avg={avg_cluster_size:.1f}")
        return node

    except Exception as e:
        print(f"[worker] ✗ Job {job_id} failed: {e}")
        import traceback; traceback.print_exc()
        fail_job(job_id, str(e))
        raise
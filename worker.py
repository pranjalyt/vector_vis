"""
worker.py — RQ worker job. This is the core processing logic.
Run with: rq worker juwi-queue

This file does ONE thing: process a submission end to end.
The embedder loads once when the worker starts, not per job.
"""
import json
import os
import random
import numpy as np
import sys

# Add parent dir to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from embedder import get_embedder
from scorer import compute_all_scores
from db import init_db, save_submission, complete_job, fail_job

# ── LOAD BASE DATA ONCE at worker startup ──
_base_data = None
_base_embeddings = None
_base_nodes = None


def _load_base_data():
    global _base_data, _base_embeddings, _base_nodes
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
    print("[worker] Ready to process jobs.")


def process_submission(job_id: str, abstract: str, name: str, stack: str):
    """
    Main job function. Called by RQ worker.
    Everything in a try/except so failures are stored, not lost.
    """
    init_db()
    _load_base_data()

    try:
        embedder = get_embedder()

        # ── 1. EMBED ──
        new_embedding = embedder.embed([abstract])[0]

        # ── 2. CLUSTER ROUTING (Hard Router) ──
        cluster_topic, color, cluster_confidence = embedder.route_to_cluster(new_embedding)

        # ── 3. BINARY FINGERPRINT ──
        threshold = np.median(_base_embeddings, axis=0)
        binary_fp = (new_embedding > threshold).astype(int)

        # ── 4. CLONE DETECTION ──
        sims = embedder.cosine_similarities(new_embedding, _base_embeddings)
        max_sim_idx = int(np.argmax(sims))
        max_sim = float(sims[max_sim_idx])
        nearest_node = _base_nodes[max_sim_idx]

        # ── 5. CLUSTER CENTROID for originality scoring ──
        same_cluster_nodes = [n for n in _base_nodes if n.get("cluster_topic") == cluster_topic]
        cluster_size = len(same_cluster_nodes)

        if same_cluster_nodes:
            same_cluster_indices = [
                i for i, n in enumerate(_base_nodes)
                if n.get("cluster_topic") == cluster_topic
            ]
            same_cluster_embeddings = _base_embeddings[same_cluster_indices]
            centroid = np.mean(same_cluster_embeddings, axis=0)
        else:
            centroid = np.mean(_base_embeddings, axis=0)

        # All distances from centroid (for normalization)
        all_distances = [
            float(np.linalg.norm(_base_embeddings[i] - centroid))
            for i in range(len(_base_nodes))
        ]

        # ── 6. COMPUTE ALL SCORES ──
        scores = compute_all_scores(
            abstract=abstract,
            stack=stack,
            embedding=new_embedding,
            centroid=centroid,
            all_centroid_distances=all_distances,
            max_cosine_similarity=max_sim,
            nearest_node_name=nearest_node["name"],
            cluster_size=cluster_size,
        )

        # ── 7. 3D POSITION ──
        if same_cluster_nodes:
            avg_x = np.mean([n["x"] for n in same_cluster_nodes])
            avg_y = np.mean([n["y"] for n in same_cluster_nodes])
            avg_z = np.mean([n["z"] for n in same_cluster_nodes])
            spread = 50
            x = float(avg_x + random.uniform(-spread, spread))
            y = float(avg_y + random.uniform(-spread, spread))
            z = float(avg_z + random.uniform(-spread, spread))
        else:
            x, y, z = [float(random.uniform(-150, 150)) for _ in range(3)]

        # ── 8. BUILD NODE ──
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

        # ── 9. PERSIST ──
        save_submission(node)
        complete_job(job_id, node)

        print(f"[worker] ✓ {name} → {cluster_topic} | status={scores['status']}")
        return node

    except Exception as e:
        error_msg = str(e)
        print(f"[worker] ✗ Job {job_id} failed: {error_msg}")
        fail_job(job_id, error_msg)
        raise
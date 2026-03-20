import json, os, sys, time
import boto3

sys.path.insert(0, os.path.dirname(__file__))
from db import init_db, save_submission, complete_job, fail_job

SQS_URL = "https://sqs.ap-south-1.amazonaws.com/711457211326/juwi-submissions"
sqs = boto3.client("sqs", region_name="ap-south-1")

import random
import numpy as np
from embedder import get_embedder
from scorer import compute_all_scores

_base_data = None
_base_embeddings = None
_base_nodes = None

def _load_base_data():
    global _base_data, _base_embeddings, _base_nodes
    if _base_data is not None:
        return
    data_path = os.path.join(os.path.dirname(__file__), "data.json")
    print("[worker] Loading base data...")
    with open(data_path) as f:
        _base_data = json.load(f)
    _base_nodes = [n for n in _base_data["nodes"] if n["id"] != "ORIGIN"]
    embedder = get_embedder()
    abstracts = [n["abstract"] for n in _base_nodes]
    print(f"[worker] Pre-embedding {len(abstracts)} base abstracts...")
    _base_embeddings = embedder.embed(abstracts)
    print("[worker] Ready.")

def process_submission(job_id, abstract, name, stack):
    init_db()
    _load_base_data()
    try:
        embedder = get_embedder()
        new_embedding = embedder.embed([abstract])[0]
        cluster_topic, color, cluster_confidence = embedder.route_to_cluster(new_embedding)
        threshold = np.median(_base_embeddings, axis=0)
        binary_fp = (new_embedding > threshold).astype(int)
        sims = embedder.cosine_similarities(new_embedding, _base_embeddings)
        max_sim_idx = int(np.argmax(sims))
        max_sim = float(sims[max_sim_idx])
        nearest_node = _base_nodes[max_sim_idx]
        same_cluster_nodes = [n for n in _base_nodes if n.get("cluster_topic") == cluster_topic]
        cluster_size = len(same_cluster_nodes)
        if same_cluster_nodes:
            same_cluster_indices = [i for i, n in enumerate(_base_nodes) if n.get("cluster_topic") == cluster_topic]
            same_cluster_embeddings = _base_embeddings[same_cluster_indices]
            centroid = np.mean(same_cluster_embeddings, axis=0)
        else:
            centroid = np.mean(_base_embeddings, axis=0)
        all_distances = [float(np.linalg.norm(_base_embeddings[i] - centroid)) for i in range(len(_base_nodes))]
        scores = compute_all_scores(
            abstract=abstract, stack=stack, embedding=new_embedding,
            centroid=centroid, all_centroid_distances=all_distances,
            max_cosine_similarity=max_sim, nearest_node_name=nearest_node["name"],
            cluster_size=cluster_size,
        )
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
        node = {
            "id": job_id, "name": name, "abstract": abstract, "stack": stack,
            "cluster_topic": cluster_topic, "color": color, "val": 7,
            "x": x, "y": y, "z": z,
            "uniqueness": float(round(np.linalg.norm([x, y, z]), 1)),
            "is_live": 1, "fingerprint_preview": binary_fp[:32].tolist(),
            **scores
        }
        save_submission(node)
        complete_job(job_id, node)
        print(f"[worker] ✓ {name} → {cluster_topic}")
        return node
    except Exception as e:
        print(f"[worker] ✗ {job_id} failed: {e}")
        fail_job(job_id, str(e))
        raise

if __name__ == "__main__":
    print("[worker] Starting SQS worker loop...")
    _load_base_data()
    while True:
        try:
            resp = sqs.receive_message(QueueUrl=SQS_URL, WaitTimeSeconds=20, MaxNumberOfMessages=1)
            for msg in resp.get("Messages", []):
                body = json.loads(msg["Body"])
                print(f"[worker] Got job {body['job_id']}")
                try:
                    process_submission(**body)
                except Exception:
                    pass
                sqs.delete_message(QueueUrl=SQS_URL, ReceiptHandle=msg["ReceiptHandle"])
        except Exception as e:
            print(f"[worker] Loop error: {e}")
            time.sleep(5)
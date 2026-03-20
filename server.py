"""
server.py — Flask API + built-in threaded worker.
No RQ, no forking, no Apple Silicon crashes.
Uses Python's queue.Queue for job management.
Same API as before — frontend doesn't change at all.

Run: python server.py
"""
import os, sys, uuid, json, random, string, threading
import queue as pyqueue
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
from db import init_db, get_job, get_leaderboard, get_stats, get_all_submissions, create_job

app = Flask(__name__)
CORS(app)

# ── JOB QUEUE (thread-safe, no forking) ──
job_queue = pyqueue.Queue()

def random_name():
    return f"{''.join(random.choices(string.ascii_uppercase, k=3))}-{random.randint(100,999)}"

# ── WORKER THREAD (runs forever in background) ──
def worker_loop():
    """
    Single background thread. Picks jobs off the queue one at a time.
    Loads the model ONCE on first job, keeps it in memory forever.
    No forking = no Apple Silicon crash.
    """
    print("[worker] Thread started, waiting for jobs...")

    # Import heavy stuff here so it loads in THIS thread, not forked
    from worker import process_submission

    while True:
        try:
            job_id, abstract, name, stack = job_queue.get(timeout=1)
            print(f"[worker] Processing {job_id}...")
            try:
                process_submission(job_id, abstract, name, stack)
            except Exception as e:
                print(f"[worker] Job {job_id} failed: {e}")
            job_queue.task_done()
        except pyqueue.Empty:
            continue
        except Exception as e:
            print(f"[worker] Unexpected error: {e}")
            continue

# Start worker thread on import
worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()

init_db()
print("✅ Juwi server ready at http://localhost:5050")

# ── ROUTES ──

@app.route("/submit", methods=["POST"])
def submit():
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    abstract = (body.get("abstract") or "").strip()
    if not abstract:
        return jsonify({"error": "Abstract is required"}), 400
    if len(abstract) < 10:
        return jsonify({"error": "Abstract too short"}), 400
    if len(abstract) > 2000:
        return jsonify({"error": "Abstract too long"}), 400

    name = (body.get("name") or "").strip() or random_name()
    stack = (body.get("stack") or "React").strip()
    job_id = str(uuid.uuid4())[:12]

    create_job(job_id)
    job_queue.put((job_id, abstract, name, stack))

    queue_pos = job_queue.qsize()

    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "queue_position": queue_pos,
        "message": f"Job queued (position {queue_pos})"
    }), 202


@app.route("/job/<job_id>", methods=["GET"])
def job_status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    response = {"status": job["status"]}
    if job["status"] == "done":
        response["result"] = job.get("result", {})
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")
    return jsonify(response)


@app.route("/data", methods=["GET"])
def get_data():
    data_path = os.path.join(os.path.dirname(__file__), "data.json")
    with open(data_path) as f:
        return jsonify(json.load(f))


@app.route("/leaderboard", methods=["GET"])
def leaderboard():
    return jsonify(get_leaderboard(limit=10))


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(get_stats())


@app.route("/submissions", methods=["GET"])
def submissions():
    return jsonify(get_all_submissions())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "api": "ok",
        "worker": "running" if worker_thread.is_alive() else "dead",
        "queue_length": job_queue.qsize(),
    })


if __name__ == "__main__":
    print("\n📡 Routes:")
    print("  POST /submit       — submit a project")
    print("  GET  /job/<id>     — poll for result")
    print("  GET  /data         — base graph data")
    print("  GET  /leaderboard  — top projects")
    print("  GET  /stats        — cluster stats")
    print("  GET  /health       — system health\n")
    app.run(port=5050, debug=False, threaded=True)
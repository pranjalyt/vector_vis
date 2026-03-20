import os, sys, uuid, json, random, string, threading, time, webbrowser
import boto3
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
from db import init_db, get_job, get_leaderboard, get_stats, get_all_submissions, create_job

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)), static_url_path='')
CORS(app)

SQS_URL = "https://sqs.ap-south-1.amazonaws.com/711457211326/juwi-submissions"
sqs = boto3.client("sqs", region_name="ap-south-1")

def random_name():
    return f"{''.join(random.choices(string.ascii_uppercase, k=3))}-{random.randint(100,999)}"

def worker_loop():
    from worker import process_submission
    print("[worker] SQS worker thread started...")
    while True:
        try:
            resp = sqs.receive_message(QueueUrl=SQS_URL, WaitTimeSeconds=20, MaxNumberOfMessages=1)
            for msg in resp.get("Messages", []):
                body = json.loads(msg["Body"])
                print(f"[worker] Got job {body['job_id']}")
                try:
                    process_submission(**body)
                except Exception as e:
                    print(f"[worker] Failed: {e}")
                sqs.delete_message(QueueUrl=SQS_URL, ReceiptHandle=msg["ReceiptHandle"])
        except Exception as e:
            print(f"[worker] Loop error: {e}")
            time.sleep(5)

worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()

init_db()
print("✅ Juwi server ready at http://localhost:5050")

@app.route('/')
def index():
    return app.send_static_file('index.html')

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

    sqs.send_message(
        QueueUrl=SQS_URL,
        MessageBody=json.dumps({
            "job_id": job_id,
            "abstract": abstract,
            "name": name,
            "stack": stack
        })
    )

    return jsonify({
        "job_id": job_id,
        "status": "queued",
        "message": "Job queued via SQS"
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
    try:
        attrs = sqs.get_queue_attributes(QueueUrl=SQS_URL, AttributeNames=["ApproximateNumberOfMessages"])
        queue_length = int(attrs["Attributes"]["ApproximateNumberOfMessages"])
    except Exception:
        queue_length = -1
    return jsonify({"api": "ok", "queue_length": queue_length, "worker": "running"})

if __name__ == "__main__":
    webbrowser.open("http://localhost:5050")
    app.run(port=5050, debug=False, threaded=True)
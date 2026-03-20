"""
db.py — All database operations in one place.
Uses SQLite. Thread-safe. Never crashes on read if DB doesn't exist yet.
"""
import sqlite3
import json
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "juwi.db")


@contextmanager
def get_conn():
    """Context manager for DB connections. Auto-commits, auto-closes."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # allows concurrent reads + writes
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS submissions (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                abstract    TEXT NOT NULL,
                stack       TEXT NOT NULL,
                cluster_topic TEXT,
                color       TEXT,
                x REAL, y REAL, z REAL,
                idea_score      INTEGER DEFAULT 0,
                tech_score      INTEGER DEFAULT 0,
                tech_label      TEXT,
                viability_score INTEGER DEFAULT 0,
                clone_risk      TEXT DEFAULT 'LOW',
                clone_similarity_pct REAL DEFAULT 0,
                is_clone    INTEGER DEFAULT 0,
                nearest_project TEXT,
                is_dark_horse   INTEGER DEFAULT 0,
                status      TEXT DEFAULT 'ACCEPTED',
                status_color TEXT DEFAULT '#ffffff',
                is_live     INTEGER DEFAULT 1,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS job_results (
                job_id      TEXT PRIMARY KEY,
                status      TEXT DEFAULT 'pending',
                result_json TEXT,
                error       TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            );
        """)
    print(f"[db] Initialized at {DB_PATH}")


def create_job(job_id: str):
    """Mark a job as pending."""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO job_results (job_id, status) VALUES (?, 'pending')",
            (job_id,)
        )


def complete_job(job_id: str, result: dict):
    """Store completed job result."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE job_results
               SET status='done', result_json=?, updated_at=datetime('now')
               WHERE job_id=?""",
            (json.dumps(result), job_id)
        )


def fail_job(job_id: str, error: str):
    """Mark job as failed with error message."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE job_results
               SET status='failed', error=?, updated_at=datetime('now')
               WHERE job_id=?""",
            (error, job_id)
        )


def get_job(job_id: str):
    """Get job status and result. Returns dict or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM job_results WHERE job_id=?", (job_id,)
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        if result.get("result_json"):
            result["result"] = json.loads(result["result_json"])
        return result


def save_submission(node: dict):
    """Persist a completed submission."""
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO submissions
            (id, name, abstract, stack, cluster_topic, color, x, y, z,
             idea_score, tech_score, tech_label, viability_score,
             clone_risk, clone_similarity_pct, is_clone, nearest_project,
             is_dark_horse, status, status_color, is_live)
            VALUES
            (:id, :name, :abstract, :stack, :cluster_topic, :color, :x, :y, :z,
             :idea_score, :tech_score, :tech_label, :viability_score,
             :clone_risk, :clone_similarity_pct, :is_clone, :nearest_project,
             :is_dark_horse, :status, :status_color, :is_live)
        """, node)


def get_leaderboard(limit=10):
    """Top projects by combined score, excluding clones."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM submissions
            WHERE is_clone = 0
            ORDER BY (idea_score + tech_score * 100 + viability_score) DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_stats():
    """Cluster and status counts."""
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        clones = conn.execute("SELECT COUNT(*) FROM submissions WHERE is_clone=1").fetchone()[0]
        dark_horses = conn.execute("SELECT COUNT(*) FROM submissions WHERE is_dark_horse=1").fetchone()[0]
        clusters = conn.execute("""
            SELECT cluster_topic, COUNT(*) as count
            FROM submissions GROUP BY cluster_topic
        """).fetchall()
        return {
            "total": total,
            "clones": clones,
            "dark_horses": dark_horses,
            "clusters": {r["cluster_topic"]: r["count"] for r in clusters}
        }


def get_all_submissions():
    """All submissions as list of dicts."""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM submissions ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
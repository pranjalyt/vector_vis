"""
scorer.py — All 5 scoring functions. Pure functions, no side effects.
Each function takes what it needs and returns a number.
"""
import numpy as np
import re

# ── TECH SOPHISTICATION ──
TECH_SCORES = {
    # Basic (1)
    "React": 1, "Angular": 1, "Node.js": 1, "PostgreSQL": 1,
    "Next.js": 1, "FastAPI": 1, "Pandas": 1,
    # Intermediate (2)
    "Arduino": 2, "Raspberry Pi": 2, "ESP32": 2, "Sensors": 2,
    "Scikit-learn": 2, "Machine Learning": 2, "Solidity": 2, "Web3": 2,
    # Advanced (3)
    "TensorFlow": 3, "BERT": 3, "Transformers": 3, "LangChain": 3,
    "LLMs": 3, "Smart Contracts": 3, "Ethereum": 3,
    "Computer Vision": 3, "Convolutional Neural Nets": 3,
    "Mesh Networks": 3, "Object Detection": 3,
    # Expert (4)
    "PyTorch": 4, "OpenAI API": 4,
}

# Viability keywords
PROBLEM_KEYWORDS = [
    "solves", "addresses", "tackles", "problem", "challenge", "issue",
    "inefficiency", "lack of", "difficulty", "unable to", "without",
    "gap", "barrier", "struggle", "fails to", "limited"
]
SOLUTION_KEYWORDS = [
    "using", "built with", "platform", "system", "tool", "app",
    "automates", "enables", "allows", "provides", "generates",
    "detects", "predicts", "monitors", "connects", "integrates"
]
IMPACT_KEYWORDS = [
    "patients", "farmers", "students", "communities", "businesses",
    "workers", "users", "people", "hospitals", "schools", "cities",
    "reduce", "improve", "increase", "save", "help", "support"
]


def score_tech(stack: str) -> int:
    """1-4 based on stack sophistication."""
    return TECH_SCORES.get(stack, 2)


def score_idea_originality(embedding: np.ndarray, centroid: np.ndarray, all_scores: list) -> int:
    """
    0-1000. Distance from cluster centroid, normalized against all existing scores.
    Far from centroid = more original.
    """
    dist = float(np.linalg.norm(embedding - centroid))
    if not all_scores:
        return 500
    min_d, max_d = min(all_scores), max(all_scores)
    if max_d == min_d:
        return 500
    normalized = (dist - min_d) / (max_d - min_d)
    return int(normalized * 1000)


def score_clone_risk(max_similarity: float) -> dict:
    """
    Returns clone info. max_similarity is 0-1 cosine similarity.
    """
    pct = round(max_similarity * 100, 1)
    if max_similarity > 0.92:
        return {"is_clone": True, "similarity_pct": pct, "risk_level": "HIGH"}
    elif max_similarity > 0.80:
        return {"is_clone": False, "similarity_pct": pct, "risk_level": "MEDIUM"}
    else:
        return {"is_clone": False, "similarity_pct": pct, "risk_level": "LOW"}


def score_viability(abstract: str, stack: str, is_clone: bool) -> int:
    """
    0-100. Heuristic scoring of how viable/complete the project idea sounds.
    """
    score = 0
    text = abstract.lower()
    words = text.split()

    # Has a clear problem statement (30 pts)
    if any(kw in text for kw in PROBLEM_KEYWORDS):
        score += 30

    # Has a solution mechanism (30 pts)
    if any(kw in text for kw in SOLUTION_KEYWORDS):
        score += 30

    # Mentions real-world impact (20 pts)
    if any(kw in text for kw in IMPACT_KEYWORDS):
        score += 20

    # Detailed enough (10 pts)
    if len(words) >= 15:
        score += 5
    if len(words) >= 25:
        score += 5

    # Penalty for being a clone (-10)
    if is_clone:
        score = max(0, score - 10)

    return min(score, 100)


def score_dark_horse(tech_score: int, cluster_size: int, is_clone: bool) -> bool:
    """
    True if: high tech sophistication + underserved cluster + not a clone.
    These are the hidden gems — impressive tech in a niche area.
    """
    return tech_score >= 3 and cluster_size <= 30 and not is_clone


def compute_all_scores(
    abstract: str,
    stack: str,
    embedding: np.ndarray,
    centroid: np.ndarray,
    all_centroid_distances: list,
    max_cosine_similarity: float,
    nearest_node_name: str,
    cluster_size: int,
) -> dict:
    """
    Single entry point — compute all 5 scores at once.
    Returns a clean dict ready to store in DB.
    """
    tech = score_tech(stack)
    clone_info = score_clone_risk(max_cosine_similarity)
    idea = score_idea_originality(embedding, centroid, all_centroid_distances)
    viability = score_viability(abstract, stack, clone_info["is_clone"])
    dark_horse = score_dark_horse(tech, cluster_size, clone_info["is_clone"])

    # Tech percentile label
    if tech == 4:
        tech_label = "EXPERT"
    elif tech == 3:
        tech_label = "ADVANCED"
    elif tech == 2:
        tech_label = "INTERMEDIATE"
    else:
        tech_label = "STANDARD"

    # Overall status
    if clone_info["is_clone"]:
        status = "CLONE_DETECTED"
        status_color = "#e06060"
    elif dark_horse:
        status = "DARK_HORSE"
        status_color = "#5fdb90"
    elif idea > 750:
        status = "HIGHLY_ORIGINAL"
        status_color = "#00ffea"
    elif viability >= 80:
        status = "HIGH_VIABILITY"
        status_color = "#ffb800"
    else:
        status = "ACCEPTED"
        status_color = "#ffffff"

    return {
        "idea_score": idea,
        "tech_score": tech,
        "tech_label": tech_label,
        "viability_score": viability,
        "clone_risk": clone_info["risk_level"],
        "clone_similarity_pct": clone_info["similarity_pct"],
        "is_clone": clone_info["is_clone"],
        "nearest_project": nearest_node_name if clone_info["is_clone"] else None,
        "is_dark_horse": dark_horse,
        "status": status,
        "status_color": status_color,
    }
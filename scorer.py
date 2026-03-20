"""
scorer.py — All 5 scoring functions. Pure functions, no side effects.
"""
import numpy as np

TECH_SCORES = {
    "React": 1, "Angular": 1, "Node.js": 1, "PostgreSQL": 1,
    "Next.js": 1, "FastAPI": 1, "Pandas": 1,
    "Arduino": 2, "Raspberry Pi": 2, "ESP32": 2, "Sensors": 2,
    "Scikit-learn": 2, "Machine Learning": 2, "Solidity": 2, "Web3": 2,
    "TensorFlow": 3, "BERT": 3, "Transformers": 3, "LangChain": 3,
    "LLMs": 3, "Smart Contracts": 3, "Ethereum": 3,
    "Computer Vision": 3, "Convolutional Neural Nets": 3,
    "Mesh Networks": 3, "Object Detection": 3,
    "PyTorch": 4, "OpenAI API": 4,
}

PROBLEM_KEYWORDS = [
    "solves", "addresses", "tackles", "problem", "challenge", "issue",
    "inefficiency", "lack of", "difficulty", "unable to", "without",
    "gap", "barrier", "struggle", "fails to", "limited", "lose", "losing"
]
SOLUTION_KEYWORDS = [
    "using", "built with", "platform", "system", "tool", "app",
    "automates", "enables", "allows", "provides", "generates",
    "detects", "predicts", "monitors", "connects", "integrates", "we built"
]
IMPACT_KEYWORDS = [
    "patients", "farmers", "students", "communities", "businesses",
    "workers", "users", "people", "hospitals", "schools", "cities",
    "reduce", "improve", "increase", "save", "help", "support", "owners"
]

STATUS_LABELS = {
    "CLONE_DETECTED":  "🔴 CLONE DETECTED",
    "SIMILAR_PATTERN": "⚠ SIMILAR PATTERN",
    "UNIQUE":          "✨ UNIQUE",
    "OUTLIER":         "🌑 OUTLIER",
}


def score_tech(stack: str) -> int:
    return TECH_SCORES.get(stack, 2)


def score_idea_originality(embedding: np.ndarray, centroid: np.ndarray, all_scores: list) -> int:
    dist = float(np.linalg.norm(embedding - centroid))
    if not all_scores or len(all_scores) < 2:
        return 500
    min_d, max_d = min(all_scores), max(all_scores)
    if max_d == min_d:
        return 500
    normalized = (dist - min_d) / (max_d - min_d)
    return int(np.clip(normalized * 1000, 0, 1000))


def score_clone_risk(max_similarity: float) -> dict:
    pct = round(max_similarity * 100, 1)
    if max_similarity > 0.96:
        return {"is_clone": True,  "similarity_pct": pct, "risk_level": "HIGH"}
    elif max_similarity > 0.88:
        return {"is_clone": False, "similarity_pct": pct, "risk_level": "MEDIUM"}
    else:
        return {"is_clone": False, "similarity_pct": pct, "risk_level": "LOW"}


def score_viability(abstract: str, stack: str, is_clone: bool) -> int:
    score = 0
    text = abstract.lower()
    words = text.split()
    if any(kw in text for kw in PROBLEM_KEYWORDS): score += 30
    if any(kw in text for kw in SOLUTION_KEYWORDS): score += 30
    if any(kw in text for kw in IMPACT_KEYWORDS):  score += 20
    if len(words) >= 15: score += 5
    if len(words) >= 25: score += 5
    if is_clone: score = max(0, score - 10)
    return min(score, 100)


def score_dark_horse(tech_score: int, cluster_size: int, is_clone: bool,
                     idea_score: int, avg_cluster_size: float) -> bool:
    return (
        tech_score >= 3
        and cluster_size < (avg_cluster_size * 0.6)
        and not is_clone
        and idea_score >= 400
    )


def compute_all_scores(
    abstract: str,
    stack: str,
    embedding: np.ndarray,
    centroid: np.ndarray,
    all_centroid_distances: list,
    max_cosine_similarity: float,
    nearest_node_name: str,
    cluster_size: int,
    avg_cluster_size: float = 28.0,
    magnitude: float = 0.0,
    all_magnitudes: list = None,
) -> dict:
    tech = score_tech(stack)
    clone_info = score_clone_risk(max_cosine_similarity)
    idea = score_idea_originality(embedding, centroid, all_centroid_distances)
    viability = score_viability(abstract, stack, clone_info["is_clone"])
    dark_horse = score_dark_horse(tech, cluster_size, clone_info["is_clone"], idea, avg_cluster_size)

    if tech == 4:   tech_label = "EXPERT"
    elif tech == 3: tech_label = "ADVANCED"
    elif tech == 2: tech_label = "INTERMEDIATE"
    else:           tech_label = "STANDARD"

    if all_magnitudes and len(all_magnitudes) >= 2:
        sorted_mags = sorted(all_magnitudes)
        p25 = np.percentile(sorted_mags, 25)
        p50 = np.percentile(sorted_mags, 50)
        p75 = np.percentile(sorted_mags, 75)
        
        if magnitude <= p25:
            status, status_color = "CLONE_DETECTED",  "#e06060"
        elif magnitude <= p50:
            status, status_color = "SIMILAR_PATTERN", "#ffb800"
        elif magnitude <= p75:
            status, status_color = "UNIQUE",          "#4a7cff"
        else:
            status, status_color = "OUTLIER",         "#5fdb90"
    else:
        # Fallback if no cluster data
        if magnitude < 50:
            status, status_color = "CLONE_DETECTED",  "#e06060"
        elif magnitude < 100:
            status, status_color = "SIMILAR_PATTERN", "#ffb800"
        elif magnitude < 200:
            status, status_color = "UNIQUE",          "#4a7cff"
        else:
            status, status_color = "OUTLIER",         "#5fdb90"

    return {
        "idea_score":           idea,
        "tech_score":           tech,
        "tech_label":           tech_label,
        "viability_score":      viability,
        "clone_risk":           clone_info["risk_level"],
        "clone_similarity_pct": clone_info["similarity_pct"],
        "is_clone":             clone_info["is_clone"],
        "nearest_project":      nearest_node_name if clone_info["is_clone"] else None,
        "is_dark_horse":        dark_horse,
        "status":               status,
        "status_color":         status_color,
        "status_label":         STATUS_LABELS.get(status, status),
    }
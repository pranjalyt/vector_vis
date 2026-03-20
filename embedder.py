"""
embedder.py — Model loads ONCE, stays in memory.
Import get_embedder() anywhere, always get the same instance.
"""
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_embedder = None

CATEGORY_DESCRIPTIONS = {
    "Web3 & Blockchain":        "ethereum solidity smart contracts blockchain decentralized web3 NFT DeFi crypto voting",
    "AI & NLP":                 "large language model LLM BERT transformers chatbot NLP text generation summarization RAG",
    "Computer Vision":          "image recognition object detection convolutional neural network camera visual classification",
    "Hardware & IoT":           "raspberry pi arduino ESP32 sensors embedded microcontroller mesh network wearable physical",
    "Sustainability & Greentech": "carbon footprint recycling renewable energy climate environment green sustainability solar ocean",
    "Web Development":          "react nodejs frontend backend REST API database web app marketplace SaaS portal",
    "Data Science & Analytics": "pandas scikit-learn machine learning prediction statistics data analysis forecasting churn",
    "Healthcare & Biotech":     "medical patient health diagnosis treatment monitoring clinical hospital telemedicine biotech",
    "Social Impact":            "community accessibility education poverty inclusion nonprofit social good volunteers refugees",
}

COLOR_PALETTE = {
    "Web3 & Blockchain":        "#b700ff",
    "AI & NLP":                 "#00ffea",
    "Computer Vision":          "#4a7cff",
    "Hardware & IoT":           "#ff6a00",
    "Sustainability & Greentech": "#5fdb90",
    "Web Development":          "#ff5c8a",
    "Data Science & Analytics": "#ffb800",
    "Healthcare & Biotech":     "#e06060",
    "Social Impact":            "#d4a853",
}

CATEGORY_NAMES = list(CATEGORY_DESCRIPTIONS.keys())


class Embedder:
    def __init__(self):
        print("[embedder] Loading all-MiniLM-L6-v2...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.category_embeddings = self.model.encode(
            list(CATEGORY_DESCRIPTIONS.values()),
            show_progress_bar=False
        )
        print("[embedder] Ready.")

    def embed(self, texts):
        """Embed a list of strings. Returns numpy array."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=False)

    def route_to_cluster(self, embedding):
        """Given a single embedding, return the best matching category name and color."""
        sims = cosine_similarity([embedding], self.category_embeddings)[0]
        best_idx = int(np.argmax(sims))
        name = CATEGORY_NAMES[best_idx]
        return name, COLOR_PALETTE[name], float(sims[best_idx])

    def cosine_similarities(self, embedding, matrix):
        """Compare one embedding against a matrix of embeddings."""
        return cosine_similarity([embedding], matrix)[0]


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
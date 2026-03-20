import json
import numpy as np
import os
import random
import string
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TECH_SCORES = {
    "React": 1, "Angular": 1, "Node.js": 1, "PostgreSQL": 1, "Next.js": 1,
    "FastAPI": 1, "Pandas": 1, "Arduino": 2, "Raspberry Pi": 2, "ESP32": 2,
    "Sensors": 2, "Scikit-learn": 2, "Machine Learning": 2, "Solidity": 2,
    "Web3": 2, "PyTorch": 4, "TensorFlow": 3, "BERT": 3, "Transformers": 3,
    "LangChain": 3, "LLMs": 3, "Smart Contracts": 3, "Ethereum": 3,
    "Computer Vision": 3, "Convolutional Neural Nets": 3, "Mesh Networks": 3,
    "OpenAI API": 4, "Object Detection": 3,
}

DOMAIN_DATA = {
    'AI/NLP': {
        'stacks': ['OpenAI API', 'LLMs', 'LangChain', 'BERT', 'Transformers'],
        'abstracts': [
            "A large language model pipeline that summarizes legal documents and extracts key clauses using BERT fine-tuning.",
            "An AI chatbot built with LangChain that answers questions about codebases by indexing GitHub repositories.",
            "Transformer-based sentiment analysis engine monitoring brand reputation across social media in real time.",
            "Using LLMs to auto-generate personalized study plans for students based on their learning history.",
            "RAG system letting hospital staff query patient records in natural language using OpenAI API.",
            "Fine-tuned BERT model detecting misinformation in news articles with 94% accuracy.",
            "LangChain agent that autonomously researches competitors and generates weekly intelligence reports.",
            "Multilingual document translation pipeline using transformer models for refugee legal aid organizations.",
            "AI writing assistant fine-tuned on academic papers to help non-native speakers improve research writing.",
            "LLM-powered code review tool that explains security vulnerabilities in plain English to junior developers.",
        ]
    },
    'AI/CV': {
        'stacks': ['Computer Vision', 'PyTorch', 'TensorFlow', 'Convolutional Neural Nets', 'Object Detection'],
        'abstracts': [
            "Real-time object detection using YOLOv8 to identify safety hazards on construction sites.",
            "CNN that diagnoses skin lesions from smartphone photos with dermatologist-level accuracy.",
            "Computer vision pipeline counting pedestrians and vehicles to optimize traffic light timing.",
            "PyTorch model reading sign language from webcam and translating gestures to text in real time.",
            "Automated quality control using TensorFlow detecting defects in manufactured parts on assembly lines.",
            "Facial recognition attendance system for classrooms that logs presence without manual roll calls.",
            "Drone-mounted computer vision identifying diseased crops in agricultural fields from aerial imagery.",
            "Visual inspection system for detecting structural cracks in bridges using convolutional neural networks.",
            "Retail shelf monitoring system that detects out-of-stock items and alerts store staff automatically.",
            "TensorFlow model that reads handwritten prescriptions to reduce pharmacist transcription errors.",
        ]
    },
    'Web Dev': {
        'stacks': ['React', 'Angular', 'Node.js', 'PostgreSQL', 'Next.js'],
        'abstracts': [
            "Next.js marketplace connecting local farmers directly with urban consumers, cutting out middlemen.",
            "React collaborative whiteboard for remote teams with real-time cursor tracking and voice notes.",
            "Node.js backend for a gig economy platform matching freelancers with short-term local jobs.",
            "Full-stack web app helping small businesses manage invoices, expenses, and tax filing in one place.",
            "Community platform where neighbors share tools, skills, and resources using React and PostgreSQL.",
            "Angular app for event organizers managing ticketing, seating, and check-ins with QR scanning.",
            "Next.js platform for indie musicians to sell albums directly to fans without record labels.",
            "Web portal helping local governments publish and track public infrastructure maintenance requests.",
            "Restaurant management SaaS built with Node.js covering reservations, orders, and staff scheduling.",
            "React-based tutoring marketplace connecting university students with peer tutors in their area.",
        ]
    },
    'Blockchain': {
        'stacks': ['Ethereum', 'Solidity', 'Smart Contracts', 'Web3'],
        'abstracts': [
            "Solidity smart contracts automating royalty payments to musicians every time their song is streamed.",
            "Decentralized voting system on Ethereum ensuring election integrity with cryptographic audit trails.",
            "Web3 platform for fractional ownership of real estate, letting anyone invest with as little as $10.",
            "Smart contract supply chain tracker verifying authenticity of luxury goods from factory to buyer.",
            "DeFi lending protocol allowing users to take collateralized loans without a bank or credit check.",
            "Blockchain credential verification for universities to issue tamper-proof digital diplomas.",
            "DAO platform for community governance of shared urban spaces and public resource allocation.",
            "NFT-based carbon credit marketplace where companies buy verified emissions offsets on-chain.",
            "Ethereum escrow system for freelancers that releases payment automatically on milestone completion.",
            "Decentralized identity system giving users control over their personal data across platforms.",
        ]
    },
    'IoT/Hardware': {
        'stacks': ['ESP32', 'Raspberry Pi', 'Sensors', 'Mesh Networks', 'Arduino'],
        'abstracts': [
            "ESP32 mesh network of soil sensors giving farmers real-time irrigation recommendations.",
            "Raspberry Pi air quality monitor mapping pollution hotspots and alerting asthma sufferers.",
            "Arduino wearable monitoring worker fatigue on factory floors by tracking biometrics.",
            "Smart home energy management using sensors to optimize appliance usage and cut electricity bills.",
            "IoT mesh network for disaster response teams working without internet in remote areas.",
            "Raspberry Pi device monitoring elderly patients at home and alerting family to unusual inactivity.",
            "ESP32 sensor network in rivers detecting early flood signs and triggering community alerts.",
            "Hardware system converting parking meter data into real-time city parking availability maps.",
            "Wearable Arduino device for visually impaired users that identifies obstacles using ultrasonic sensors.",
            "Industrial equipment monitor using vibration sensors to predict mechanical failures before they happen.",
        ]
    },
    'Data Science': {
        'stacks': ['Machine Learning', 'Pandas', 'Scikit-learn'],
        'abstracts': [
            "Predictive model forecasting hospital bed demand 72 hours in advance to optimize staffing levels.",
            "Data mining pipeline analyzing city permit records to identify neighborhoods at risk of gentrification.",
            "Machine learning system predicting equipment failures in manufacturing plants before they occur.",
            "Pandas analytics dashboard helping school districts identify students at risk of dropping out.",
            "Predictive analysis tool for e-commerce personalizing discounts based on purchase likelihood.",
            "Scikit-learn model predicting energy consumption patterns for smart grid load balancing.",
            "Data pipeline aggregating public health records to identify disease outbreak patterns weeks early.",
            "Customer churn prediction model helping subscription businesses retain at-risk users proactively.",
            "Sports performance analytics platform using historical match data to optimize team strategy.",
            "Real estate price forecasting model trained on 10 years of transaction data for investment decisions.",
        ]
    },
    'Healthtech': {
        'stacks': ['PyTorch', 'TensorFlow', 'OpenAI API', 'Sensors', 'React'],
        'abstracts': [
            "AI app analyzing sleep patterns from wearable data and recommending personalized health interventions.",
            "Mental health platform using NLP to detect early signs of depression in patient journaling entries.",
            "Telemedicine platform connecting rural patients with medical specialists via low-bandwidth video.",
            "Drug interaction checker using medical knowledge graphs alerting pharmacists to dangerous combinations.",
            "Wearable biosensor continuously monitoring glucose levels for diabetics without finger-prick tests.",
            "AI triage system for emergency rooms predicting patient deterioration from vital sign streams.",
            "Remote rehabilitation app guiding stroke patients through physiotherapy using pose estimation.",
            "Clinical trial matching platform connecting patients with relevant medical research studies.",
            "Mental health chatbot trained on CBT techniques providing support between therapy sessions.",
            "Automated radiology report generator drafting preliminary findings from medical imaging scans.",
        ]
    },
    'Greentech': {
        'stacks': ['Sensors', 'Machine Learning', 'React', 'Raspberry Pi', 'Scikit-learn'],
        'abstracts': [
            "Carbon footprint tracker integrating with bank accounts to calculate environmental cost of purchases.",
            "ML model optimizing renewable energy distribution across microgrids in real time.",
            "Recycling sorting robot using computer vision to separate waste streams with 98% accuracy.",
            "Platform gamifying sustainability challenges for corporations tracking verified carbon offsets.",
            "Sensor network monitoring deforestation in real time using acoustic detection of chainsaws in forests.",
            "App connecting food businesses with surplus inventory to local charities, reducing food waste.",
            "Smart irrigation controller using weather forecasts and soil data to reduce agricultural water usage.",
            "Community solar platform letting apartment residents invest in shared rooftop solar installations.",
            "Ocean plastic detection system using satellite imagery and machine learning to guide cleanup vessels.",
            "Building energy audit tool analyzing utility bills and recommending efficiency improvements.",
        ]
    },
    'Social': {
        'stacks': ['React', 'Node.js', 'LLMs', 'PostgreSQL', 'Next.js'],
        'abstracts': [
            "Platform pairing refugees with volunteer mentors who have relevant professional expertise.",
            "AI tool translating government documents into plain language accessible to people with low literacy.",
            "Community app connecting isolated elderly residents with local volunteers for errands and companionship.",
            "Crowdsourced platform where citizens report and track local infrastructure issues like potholes.",
            "Digital literacy program delivered via WhatsApp teaching basic internet skills to rural communities.",
            "Platform connecting pro-bono lawyers with low-income individuals who need legal representation.",
            "Peer support network for first-generation college students navigating applications and financial aid.",
            "App helping domestic violence survivors find safe housing, legal aid, and counseling resources.",
            "Volunteer coordination platform for disaster relief organizations managing hundreds of volunteers.",
            "Language exchange app connecting immigrants with native speakers for conversational practice.",
        ]
    },
}

def generate_project(domain, is_outlier=False):
    config = DOMAIN_DATA[domain]
    stack = random.choice(config['stacks'])
    name = f"{''.join(random.choices(string.ascii_uppercase, k=3))}-{random.randint(100, 999)}"
    if is_outlier:
        niche = random.choice(["neuromorphic computing", "homomorphic encryption", "zero-knowledge proofs", "federated learning", "quantum-resistant cryptography"])
        abstract = f"A radical solution combining {stack} with {niche} to solve large-scale coordination failures in decentralized autonomous systems."
    else:
        abstract = random.choice(config['abstracts'])
    return {"name": name, "abstract": abstract, "stack": stack, "domain": domain}

domains = list(DOMAIN_DATA.keys())
all_projects = []

print("Generating 250 diverse project entries...")
for domain in domains:
    for _ in range(27):
        all_projects.append(generate_project(domain))
while len(all_projects) < 240:
    all_projects.append(generate_project(random.choice(domains)))
for _ in range(10):
    all_projects.append(generate_project(random.choice(domains), is_outlier=True))
random.shuffle(all_projects)

projects_data = [{"id": str(i+1), "name": p['name'], "abstract": p['abstract'], "stack": p['stack'], "domain": p['domain'], "tech_score": TECH_SCORES.get(p['stack'], 2)} for i, p in enumerate(all_projects)]

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding...")
abstracts = [p["abstract"] for p in projects_data]
embeddings = model.encode(abstracts, show_progress_bar=True)

print("Binarizing...")
threshold = np.median(embeddings, axis=0)
binary_fingerprints = (embeddings > threshold).astype(int)

print("Detecting clones...")
clone_pairs = []
for i in range(len(projects_data)):
    for j in range(i+1, len(projects_data)):
        sim = 1 - np.sum(binary_fingerprints[i] != binary_fingerprints[j]) / len(binary_fingerprints[i])
        if sim > 0.95:
            clone_pairs.append((str(i+1), str(j+1), float(round(sim, 3))))
print(f"Found {len(clone_pairs)} clone pairs")

print("PCA + KMeans...")
pca = PCA(n_components=3, random_state=42)
embeddings_3d = (pca.fit_transform(embeddings) / np.max(np.abs(pca.fit_transform(embeddings)))) * 250
kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

print("Naming clusters semantically (unique)...")
category_descriptions = {
    "Web3 & Blockchain":        "ethereum solidity smart contracts blockchain decentralized web3 NFT DeFi crypto voting",
    "AI & NLP":                 "large language model LLM BERT transformers chatbot NLP text generation summarization",
    "Computer Vision":          "image recognition object detection convolutional neural network camera visual classification",
    "Hardware & IoT":           "raspberry pi arduino ESP32 sensors embedded microcontroller mesh network physical device wearable",
    "Sustainability & Greentech": "carbon footprint recycling renewable energy climate environment green sustainability solar ocean",
    "Web Development":          "react nodejs frontend backend REST API database web app user interface marketplace SaaS",
    "Data Science & Analytics": "pandas scikit-learn machine learning prediction statistics data analysis forecasting churn",
    "Healthcare & Biotech":     "medical patient health diagnosis treatment monitoring clinical hospital telemedicine biotech radiology",
    "Social Impact":            "community accessibility education poverty inclusion nonprofit social good volunteers refugees literacy",
}
category_names = list(category_descriptions.keys())
category_embeddings = model.encode(list(category_descriptions.values()))
color_palette = {
    "Web3 & Blockchain": "#b700ff", "AI & NLP": "#00ffea", "Computer Vision": "#4a7cff",
    "Hardware & IoT": "#ff6a00", "Sustainability & Greentech": "#5fdb90",
    "Web Development": "#ff5c8a", "Data Science & Analytics": "#ffb800",
    "Healthcare & Biotech": "#e06060", "Social Impact": "#d4a853",
}

cluster_names = {}
cluster_colors_map = {}
used_categories = set()
scored = [(c_id, cosine_similarity(kmeans.cluster_centers_[c_id].reshape(1,-1), category_embeddings)[0]) for c_id in range(9)]
scored.sort(key=lambda x: -np.max(x[1]))
for c_id, sims in scored:
    for idx in np.argsort(sims)[::-1]:
        if category_names[idx] not in used_categories:
            cluster_names[c_id] = category_names[idx]
            cluster_colors_map[c_id] = color_palette[category_names[idx]]
            used_categories.add(category_names[idx])
            break

print("Cluster assignments:")
for c_id in range(9):
    count = sum(1 for l in cluster_labels if l == c_id)
    print(f"  {cluster_names[c_id]}: {count} projects")

idea_scores = np.array([np.linalg.norm(embeddings[i] - kmeans.cluster_centers_[cluster_labels[i]]) for i in range(len(projects_data))])
idea_scores_norm = ((idea_scores - idea_scores.min()) / (idea_scores.max() - idea_scores.min()) * 1000).astype(int)

clone_lookup = {}
for a, b, sim in clone_pairs:
    clone_lookup.setdefault(a, []).append({"id": b, "similarity": sim})
    clone_lookup.setdefault(b, []).append({"id": a, "similarity": sim})

all_tech = [p["tech_score"] for p in projects_data]
nodes = [{"id": "ORIGIN", "name": "Mathematical Zero", "cluster_topic": "Vector Origin", "color": "#ffffff", "val": 2, "x": 0.0, "y": 0.0, "z": 0.0, "abstract": "Center of all ideas.", "tech_score": 0, "idea_score": 0}]
links = []

for i, proj in enumerate(projects_data):
    x, y, z = embeddings_3d[i]
    c_id = cluster_labels[i]
    magnitude = np.linalg.norm([x, y, z])
    if (magnitude / 250) * 100 > 70:
        x *= 1.5; y *= 1.5; z *= 1.5
        magnitude = np.linalg.norm([x, y, z])
    is_clone = proj["id"] in clone_lookup
    tech_score = proj["tech_score"]
    cluster_size = sum(1 for l in cluster_labels if l == c_id)
    is_dark_horse = tech_score >= 3 and cluster_size <= 30 and not is_clone
    nodes.append({
        "id": proj["id"], "name": proj["name"], "abstract": proj["abstract"], "stack": proj["stack"],
        "cluster_topic": cluster_names.get(c_id, "Unknown"), "cluster_id": int(c_id),
        "color": cluster_colors_map.get(c_id, "#ffffff"), "val": 5 if is_dark_horse else 4,
        "x": float(x), "y": float(y), "z": float(z), "uniqueness": float(round(magnitude, 1)),
        "idea_score": int(idea_scores_norm[i]), "tech_score": tech_score,
        "tech_percentile": int(round(np.mean(np.array(all_tech) <= tech_score) * 100)),
        "is_clone": is_clone, "clones": clone_lookup.get(proj["id"], []),
        "is_dark_horse": is_dark_horse, "fingerprint": binary_fingerprints[i].tolist()
    })
    links.append({"source": "ORIGIN", "target": proj["id"], "color": cluster_colors_map.get(c_id, "#ffffff")})

with open("data.json", "w") as f:
    json.dump({"nodes": nodes, "links": links, "clone_pairs": clone_pairs}, f, indent=2)

print(f"\nDone! {len(nodes)-1} projects, {len(clone_pairs)} clones detected.")
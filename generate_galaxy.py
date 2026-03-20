# import json
# import numpy as np
# import os
# import random
# import string
# from sentence_transformers import SentenceTransformer
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

# # Suppress terminal warnings for a clean execution
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# def generate_project(domain, is_outlier=False):
#     tech_stacks = {
#         'AI/NLP': ['OpenAI API', 'LLMs', 'LangChain', 'BERT', 'Transformers', 'FastAPI'],
#         'AI/CV': ['Computer Vision', 'PyTorch', 'TensorFlow', 'Convolutional Neural Nets', 'Object Detection'],
#         'Web Dev': ['React', 'Angular', 'Node.js', 'PostgreSQL', 'Stripe Integration', 'Next.js'],
#         'Blockchain': ['Ethereum', 'Solidity', 'Smart Contracts', 'Web3', 'Decentralized Apps'],
#         'IoT/Hardware': ['ESP32', 'Raspberry Pi', 'Sensors', 'Mesh Networks', 'Real-time data', 'Arduino'],
#         'Data Science': ['Machine Learning', 'Pandas', 'Scikit-learn', 'Predictive Analysis', 'Data Mining']
#     }
    
#     problems = [
#         "inefficient communication in small teams",
#         "lack of automated sorting for recyclables",
#         "difficulty tracking personal health metrics",
#         "unreliable supply chain data for local goods",
#         "limited accessibility to technical documentation for beginners",
#         "slow transaction processing for cross-border payments",
#         "high energy consumption in smart home devices",
#         "lack of privacy in shared digital workspaces",
#         "difficulty validating truth in online media",
#         "unoptimized route planning for urban logistics"
#     ]
    
#     project_types = [
#         "AI-powered analysis tool",
#         "Decentralized application",
#         "Real-time monitoring system",
#         "Hardware-software hybrid solution",
#         "Predictive modeling platform",
#         "Smart sensor network",
#         "Blockchain-based verification protocol",
#         "Automated sorting assistant",
#         "Personalized feedback system",
#         "Community-driven platform"
#     ]
    
#     stack = random.choice(tech_stacks[domain])
#     problem = random.choice(problems)
#     p_type = random.choice(project_types)
    
#     name = f"{''.join(random.choices(string.ascii_uppercase, k=3))}-{random.randint(100, 999)}"
    
#     if is_outlier:
#         abstract = f"A radical solution utilizing {stack} and a novel bio-inspired hardware bioreactor to solve {problem}."
#     else:
#         abstract = f"This project is a {p_type} built with {stack} to address the challenge of {problem}."
        
#     return {"name": name, "abstract": abstract, "stack": stack}

# domains = ['AI/NLP', 'AI/CV', 'Web Dev', 'Blockchain', 'IoT/Hardware', 'Data Science']
# all_projects = []

# print("🚀 Generating 250 diverse project entries...")
# # Generate typical projects
# for _ in range(240):
#     domain = random.choice(domains)
#     all_projects.append(generate_project(domain))

# # Generate 10 outlier projects to act as unique vectors
# for _ in range(10):
#     domain = random.choice(domains)
#     all_projects.append(generate_project(domain, is_outlier=True))

# projects_data = []
# for i, p in enumerate(all_projects):
#     projects_data.append({"id": str(i+1), "name": p['name'], "abstract": p['abstract'], "stack": p['stack']})

# print("🧠 Loading Neural Embedding Model...")
# # This model converts sentences into 384-dimensional mathematical arrays
# model = SentenceTransformer('all-MiniLM-L6-v2')

# print("🔢 Crunching 250 abstracts into 384-dimensional vectors...")
# abstracts = [p["abstract"] for p in projects_data]
# embeddings = model.encode(abstracts)

# print("🌌 Using PCA for mathematical vector space and magnitude preservation...")
# pca = PCA(n_components=3, random_state=42)
# embeddings_3d = pca.fit_transform(embeddings)

# # Normalize space to fit nicely inside a large 3D bounding box (e.g., -250 to 250)
# max_val = np.max(np.abs(embeddings_3d))
# embeddings_3d = (embeddings_3d / max_val) * 250

# print("🤖 Running K-Means to dynamically discover conceptual clusters (8-10 topics)... ")
# # Adjust the number of clusters to discover more varied topics
# num_clusters = 9
# kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
# cluster_labels = kmeans.fit_predict(embeddings)

# print("💾 Creating dynamic labels and building the vector space JSON...")
# cluster_names = {}
# for c_id in range(num_clusters):
#     cluster_texts = [abstracts[i] for i in range(len(projects_data)) if cluster_labels[i] == c_id]
#     text_dump = " ".join(cluster_texts).lower()
    
#     if "blockchain" in text_dump or "ethereum" in text_dump or " decentralized" in text_dump:
#         cluster_names[c_id] = "Web3 & Blockchain"
#     elif "ai" in text_dump or "llm" in text_dump or "nlp" in text_dump or "gpt" in text_dump:
#         cluster_names[c_id] = "AI & NLP"
#     elif "computer vision" in text_dump or "object detection" in text_dump:
#         cluster_names[c_id] = "AI & Computer Vision"
#     elif "iot" in text_dump or "sensor" in text_dump or "hardware" in text_dump or "mesh network" in text_dump:
#         cluster_names[c_id] = "Hardware & IoT"
#     elif "web" in text_dump or "platform" in text_dump or "marketplace" in text_dump:
#         cluster_names[c_id] = "Web Development"
#     elif "predict" in text_dump or "model" in text_dump or "analyze" in text_dump or "machine learning" in text_dump:
#         cluster_names[c_id] = "Machine Learning & Data Science"
#     elif "recyc" in text_dump or "sorting" in text_dump or "environment" in text_dump:
#         cluster_names[c_id] = "Sustainability & Greentech"
#     elif "health" in text_dump or "metric" in text_dump or "medical" in text_dump:
#         cluster_names[c_id] = "Healthtech"
#     elif "communication" in text_dump or "accessibility" in text_dump or "community" in text_dump:
#         cluster_names[c_id] = "Social Impact & Connectivity"
#     else:
#         cluster_names[c_id] = "General Innovation"

# # Define diverse, professional-looking colors for the many clusters
# colors = [
#     "#00ffea", "#ff0055", "#b700ff", "#ffb800", "#5fdb90", 
#     "#4a7cff", "#ff5c8a", "#e06060", "#d4a853", "#7b9e87"
# ]

# nodes = []
# links = []

# # Create the Origin Point (0,0,0) as the universal reference
# nodes.append({
#     "id": "ORIGIN", "name": "Mathematical Zero", "cluster_topic": "Vector Origin", 
#     "color": "#ffffff", "val": 2, "x": 0.0, "y": 0.0, "z": 0.0, "abstract": "Center of all ideas."
# })

# for i, proj in enumerate(projects_data):
#     x, y, z = embeddings_3d[i]
#     c_id = cluster_labels[i]
    
#     # Calculate magnitude from origin for uniqueness. 
#     magnitude = np.linalg.norm([x, y, z])
    
#     # EXTREME SEPARATION: Push unique outlier ideas even further out for visual effect
#     uniqueness_normalized = (magnitude / 250) * 100
#     if uniqueness_normalized > 70:
#         x *= 1.5; y *= 1.5; z *= 1.5;
#         magnitude = np.linalg.norm([x, y, z])
        
#     nodes.append({
#         "id": proj["id"],
#         "name": proj["name"],
#         "cluster_topic": cluster_names.get(c_id, "Unknown Segment"),
#         "uniqueness": float(round(magnitude, 1)),
#         "color": colors[c_id % len(colors)],
#         "val": 4, 
#         "x": float(x), "y": float(y), "z": float(z),
#         "abstract": proj["abstract"]
#     })
    
#     # To mimic 'arrows' conceptually, we'll connect from the ORIGIN to each project node.
#     # In a very large graph, arrows from the origin become unreadable; but 
#     # we can use the line from the center as a reference during interaction.
#     links.append({
#         "source": "ORIGIN",
#         "target": proj["id"],
#         "color": colors[c_id % len(colors)]
#     })

# # We'll use a specific structure for 3D Force Graph library where 'nodes' is the data.
# # 'links' will just be for connection logic, not visualization.
# output = {"nodes": nodes, "links": links} 

# with open("data.json", "w") as f:
#     json.dump(output, f, indent=4)

# print("✅ Mathematical Vector Space generated successfully with 250 diverse projects!")
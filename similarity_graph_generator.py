import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# ---------------- CONFIG ----------------
MODEL_DIR = "./models/all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = "R8"
DATA_DIR = "data"

# ---------------- SBERT LOADER ----------------
def load_sbert(model_dir=MODEL_DIR):
    if os.path.exists(model_dir):
        print(f"📂 Loading SBERT model from {model_dir}")
        model = SentenceTransformer(model_dir, device=DEVICE)
    else:
        print("⬇️ Downloading SBERT model (first time only)...")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=DEVICE)
        model.save(model_dir)
        print(f"✅ Saved model to {model_dir}")
    return model

# ---------------- LOAD CORPUS ----------------
text_file = os.path.join(DATA_DIR, f"{DATASET}.clean.txt")
split_file = os.path.join(DATA_DIR, f"{DATASET}.txt")  # contains idx, split, label

with open(text_file, "r", encoding="latin1") as f:
    corpus = [line.strip() for line in f.readlines()]

# ---------------- LOAD SPLIT + LABEL ----------------
train_idx, test_idx = [], []
train_labels, test_labels = [], []

label_map = {}  # Map string labels to integers
current_label = 0

with open(split_file, "r", encoding="latin1") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx, split, label_str = line.split()
        idx = int(idx) - 1  # adjust if indices start at 1

        # Map string label to integer
        if label_str not in label_map:
            label_map[label_str] = current_label
            current_label += 1
        label = label_map[label_str]

        if split.lower() == "train":
            train_idx.append(idx)
            train_labels.append(label)
        elif split.lower() == "test":
            test_idx.append(idx)
            test_labels.append(label)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
print(f"Label mapping: {label_map}")

# ---------------- LOAD SBERT MODEL ----------------
sbert_model = load_sbert()

# ---------------- BUILD SIMILARITY MATRIX ----------------
def subset_similarity(indices, name):
    subset_corpus = [corpus[i] for i in indices]
    embeddings = sbert_model.encode(
        subset_corpus,
        convert_to_tensor=True,
        device=DEVICE,
        show_progress_bar=True
    )
    sim_matrix = cosine_similarity(embeddings.cpu().numpy())
    
    out_path = os.path.join(DATA_DIR, f"{DATASET}.{name}.modularity_adj")
    with open(out_path, 'wb') as f:
        np.save(f, sim_matrix)
    print(f"✅ Saved {name} similarity matrix: {out_path}, shape={sim_matrix.shape}")
    return sim_matrix

# ---------------- GENERATE MATRICES ----------------
sim_train = subset_similarity(train_idx, "train")
sim_test  = subset_similarity(test_idx, "test")

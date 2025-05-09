import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

# ============== CONFIG ==============
DATA_DIR = os.environ.get("PLAYLIST_DATA_DIR", "./data")
DATASET_PATH = os.path.join(DATA_DIR, "playlist_track_list.csv")
OUTPUT_NPZ = "./outputs/lightgcn_embeddings.npz"
MODEL_PATH = "./outputs/lightgcn_model.pt"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8000")
EXPERIMENT_NAME = "lightgcn-training"
EMBED_DIM = 64
EPOCHS = 50
BATCH_SIZE = 1024
LR = 0.001
N_LAYERS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ============== DATA LOADING ==============
print("Loading playlist-track data...")
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df)} playlist-track pairs")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Create user (playlist) and item (track) mappings
playlists = df['playlist_id'].unique()
tracks = df['track_uri'].unique()

playlist_to_idx = {pid: i for i, pid in enumerate(playlists)}
track_to_idx = {tid: i for i, tid in enumerate(tracks)}
idx_to_track = {i: tid for tid, i in track_to_idx.items()}

n_playlists = len(playlists)
n_tracks = len(tracks)

print(f"Dataset contains {n_playlists} playlists and {n_tracks} tracks")

# Create interaction matrix
interactions = []
for _, row in df.iterrows():
    pid = playlist_to_idx[row['playlist_id']]
    tid = track_to_idx[row['track_uri']]
    interactions.append((pid, tid))

# Split train/test
train_interactions, test_interactions = train_test_split(interactions, test_size=0.1, random_state=42)

# ============== LIGHTGCN MODEL ==============
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_layers):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def create_adjacency_matrix(self, user_indices, item_indices):
        n_nodes = self.n_users + self.n_items
        adjacency = torch.zeros((n_nodes, n_nodes), device=DEVICE)
        
        # Convert item indices to internal indices
        item_indices_adj = item_indices + self.n_users
        
        # Fill adjacency matrix
        for i in range(len(user_indices)):
            adjacency[user_indices[i], item_indices_adj[i]] = 1
            adjacency[item_indices_adj[i], user_indices[i]] = 1
        
        # Normalize adjacency matrix
        rowsum = torch.sum(adjacency, dim=1)
        d_inv_sqrt = torch.pow(rowsum + 1e-7, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adjacency = torch.mm(torch.mm(d_mat_inv_sqrt, adjacency), d_mat_inv_sqrt)
        
        return adjacency
    
    def forward(self, user_indices, item_indices):
        adjacency = self.create_adjacency_matrix(user_indices, item_indices)
        
        # Initial embeddings
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]
        
        # Graph convolution
        for _ in range(self.n_layers):
            all_embeddings = torch.mm(adjacency, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Layer combination
        lightgcn_out = torch.stack(embeddings_list, dim=1)
        lightgcn_out = torch.mean(lightgcn_out, dim=1)
        
        # Split user and item embeddings
        users, items = torch.split(lightgcn_out, [self.n_users, self.n_items])
        
        return users, items
    
    def bpr_loss(self, users, pos_items, neg_items):
        # Get embeddings
        users_emb, items_emb = self.forward(users, pos_items)
        
        # User embeddings
        user_embeddings = users_emb[users]
        
        # Positive item embeddings
        pos_item_embeddings = items_emb[pos_items - self.n_users]
        
        # Negative item embeddings
        neg_item_embeddings = items_emb[neg_items - self.n_users]
        
        # BPR loss
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)
        
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # L2 regularization
        l2_reg = torch.norm(user_embeddings) ** 2 + torch.norm(pos_item_embeddings) ** 2 + torch.norm(neg_item_embeddings) ** 2
        
        return loss + 1e-5 * l2_reg


# ============== TRAINING ==============
# Prepare training data tensors
train_users = torch.tensor([i[0] for i in train_interactions], dtype=torch.long).to(DEVICE)
train_items = torch.tensor([i[1] for i in train_interactions], dtype=torch.long).to(DEVICE)

# Create model
model = LightGCN(n_playlists, n_tracks, EMBED_DIM, N_LAYERS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
with mlflow.start_run():
    mlflow.log_params({
        "embedding_dim": EMBED_DIM,
        "n_layers": N_LAYERS,
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "n_playlists": n_playlists,
        "n_tracks": n_tracks
    })
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Process in batches
        for i in range(0, len(train_users), BATCH_SIZE):
            batch_users = train_users[i:i+BATCH_SIZE]
            batch_items = train_items[i:i+BATCH_SIZE]
            
            # Simple random negative sampling
            neg_items = torch.randint(0, n_tracks, (len(batch_items),), device=DEVICE)
            neg_items = neg_items + n_playlists  # Adjust indices
            
            # Calculate loss
            loss = model.bpr_loss(batch_users, batch_items + n_playlists, neg_items)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # Log metrics
        avg_loss = total_loss / n_batches
        mlflow.log_metric("loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    
    # Extract and save track embeddings
    with torch.no_grad():
        # Generate full embeddings
        all_users = torch.arange(n_playlists, device=DEVICE)
        all_items = torch.arange(n_tracks, device=DEVICE)
        _, item_embeddings = model.forward(all_users, all_items)
        
        # Move to CPU and convert to numpy
        item_embeddings = item_embeddings.cpu().numpy()
        
        # Save as NPZ file
        track_emb_dict = {idx_to_track[i]: item_embeddings[i] for i in range(n_tracks)}
        np.savez(OUTPUT_NPZ, **track_emb_dict)
        mlflow.log_artifact(OUTPUT_NPZ)

print(f"LightGCN training complete. Model saved to {MODEL_PATH}")
print(f"Track embeddings saved to {OUTPUT_NPZ}")

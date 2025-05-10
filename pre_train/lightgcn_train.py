import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
import torch
import mlflow
import gc
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import LightGCN
from torch_geometric.utils import to_torch_coo_tensor
import os
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.25.37:8000/")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://129.114.25.37:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "admin")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "CqPacaUHUur82cNZI1R")
EXPERIMENT_NAME = "lightgcn-training"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/block/outputs")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "lightgcn_embeddings.npz")

# Configure MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def cleanup_memory():
    """Clean up both RAM and GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Load CSV normally since header is correct
df = pd.read_csv('/mnt/block/playlist_track_pairs.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Confirm column names are what we expect
logger.info(f"Columns: {df.columns}")

# Encode
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df['user_id'] = user_encoder.fit_transform(df['playlist_id'])
df['item_id'] = item_encoder.fit_transform(df['track_uri'])

df = df.drop(columns=['playlist_id', 'track_uri'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df = df.groupby('user_id').apply(lambda x: x.sample(frac=0.8, random_state=42)).reset_index(drop=True)
test_df = df[~df.index.isin(train_df.index)]

num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()

train_mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
for row in train_df.itertuples():
    train_mat[row.user_id, row.item_id] = 1.0

num_nodes = num_users + num_items
embedding_dim = 64
num_layers = 2

df['item_id_offset'] = df['item_id'] + num_users

# Stack edges as [2, num_edges] shape
edge_index = torch.tensor(df[['user_id', 'item_id_offset']].values.T, dtype=torch.long).to(device)

# Add reverse edges (for symmetry)
edge_index_rev = edge_index[[1, 0], :]
edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
adj_t = to_torch_coo_tensor(edge_index, size=(num_nodes, num_nodes)).to_sparse().cuda()

# === Model setup ===
embedding_dim = 32
num_layers = 2
lr = 0.001
epochs = 20
batch_size = 1024

model = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# === BPR Loss ===
def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_score = (user_emb * pos_emb).sum(dim=1)
    neg_score = (user_emb * neg_emb).sum(dim=1)
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

def get_negative_samples(user_ids, train_mat, num_items, num_neg=1):
    """Sample negative items for each user"""
    neg_items = []
    for u in user_ids:
        user_pos_items = set(train_mat[u].nonzero()[1])
        neg_item = np.random.randint(0, num_items)
        while neg_item in user_pos_items:
            neg_item = np.random.randint(0, num_items)
        neg_items.append(neg_item)
    return torch.tensor(neg_items, device=device)

# Training loop with MLflow
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_params({
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_users": num_users,
        "num_items": num_items
    })
    
    model.train()
    losses = []
    
    for epoch in tqdm(range(epochs), desc="Training LightGCN"):
        optimizer.zero_grad()
        
        # Forward pass
        all_embs = model(edge_index)
        user_embs = all_embs[:num_users]
        item_embs = all_embs[num_users:]
        
        # Sample user-positive-negative triplets
        user_ids = torch.randint(0, num_users, (batch_size,))
        pos_items = []
        for u in user_ids.tolist():
            user_pos_items = train_mat[u].nonzero()[1]
            if len(user_pos_items) > 0:
                pos_items.append(np.random.choice(user_pos_items))
            else:
                pos_items.append(0)  # Fallback if user has no positive items
        
        pos_items = torch.tensor(pos_items, device=device)
        neg_items = get_negative_samples(user_ids, train_mat, num_items)
        
        # Calculate loss
        loss = bpr_loss(
            user_embs[user_ids],
            item_embs[pos_items],
            item_embs[neg_items]
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log metrics
        mlflow.log_metric("loss", loss.item(), step=epoch)
        losses.append(loss.item())
        
        # Cleanup memory
        cleanup_memory()
    
    # Save model to MinIO through MLflow
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "lightgcn_model.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path, "models")
        logger.info(f"Model saved to MinIO through MLflow run {run.info.run_id}")
    
    # Generate and save embeddings
    model.eval()
    with torch.no_grad():
        all_embs = model(edge_index)
        user_embs = all_embs[:num_users].cpu().numpy()
        item_embs = all_embs[num_users:].cpu().numpy()
    
    # Save embeddings
    np.savez(
        EMBEDDINGS_PATH,
        user_embeddings=user_embs,
        item_embeddings=item_embs,
        user_mapping={str(k): v for k, v in user_encoder.classes_.items()},
        item_mapping={str(k): v for k, v in item_encoder.classes_.items()}
    )
    mlflow.log_artifact(EMBEDDINGS_PATH, "embeddings")
    
    # Save model info
    model_info = {
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "mlflow_run_id": run.info.run_id
    }
    mlflow.log_dict(model_info, "model_info.json")

logger.info("Training completed successfully!")
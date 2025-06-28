import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 1) load the CSV you described
df = pd.read_csv("../face/train.csv")  # contains session_id, turns, …, query, …, wrong_ans

# 2) drop any rows without a query
df = df.dropna(subset=["query"])

# 3) define X (queries) and y (labels)
queries = df["query"].tolist()
y       = df["wrong_ans"].astype(int).values  # assuming 0/1 wrong_ans

# 4) embed all queries with SBERT
sbert = SentenceTransformer("all-MiniLM-L6-v2")
embs  = sbert.encode(queries, convert_to_tensor=True)

# 5) train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    embs, y, test_size=0.2, random_state=42, stratify=y
)

# 6) wrap in Dataset/DataLoader
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embs   = embeddings
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.embs[i], self.labels[i]

train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(EmbeddingDataset(X_val,   y_val),   batch_size=64)

# 7) define the same MLP from before
class SBERTClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden_dim, 1)  # binary

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SBERTClassifier(embs.shape[1]).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=2e-4)
crit   = nn.BCEWithLogitsLoss()

# 8) training loop
best_val = 0.0
for epoch in range(1, 1001):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device).float()
        logits = model(Xb)
        loss   = crit(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * Xb.size(0)
    avg_train = total_loss / len(train_loader.dataset)

    model.eval()
    correct = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = (torch.sigmoid(model(Xb)) > 0.5).long()
            correct += (preds == yb).sum().item()
    val_acc = correct / len(val_loader.dataset)

    print(f"Epoch {epoch}: train_loss={avg_train:.4f}  val_acc={val_acc:.4f}")
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_sbert_mlp.pt")





from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embs   = embeddings
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.embs[i], self.labels[i]

train_ds = EmbeddingDataset(X_train, y_train)
val_ds   = EmbeddingDataset(X_val,   y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64)

class SBERTClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SBERTClassifier(input_dim=embs.shape[1],
                        hidden_dim=128,
                        num_classes=3,
                        dropout=0.2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

def train_epoch(loader):
    model.train()
    total_loss = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss   = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
    return correct / len(loader.dataset)

# --- training loop ---
best_val_acc = 0
for epoch in range(1, 21):
    train_loss = train_epoch(train_loader)
    val_acc     = eval_epoch(val_loader)
    print(f"Epoch {epoch:02d} — loss: {train_loss:.4f}, val_acc: {val_acc:.4f}")
    # save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_sbert_mlp.pt")

# #inference
# model.load_state_dict(torch.load("best_sbert_mlp.pt"))
# model.eval()

# def predict_difficulty(text: str):
#     emb = sbert.encode([text], convert_to_tensor=True).to(device)
#     logits = model(emb)
#     cls    = logits.argmax(dim=1).item()
#     return ["easy","medium","hard"][cls]

# print(predict_difficulty("Implement Dijkstra’s algorithm"))  # e.g. "hard"

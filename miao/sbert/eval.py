import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 1) Load your val.csv
df_val = pd.read_csv("../face/val.csv")  # must contain at least “query” and “wrong_ans” columns
queries = df_val["query"].tolist()
y_true  = df_val["wrong_ans"].astype(int).values

# 2) Embed with SBERT
sbert   = SentenceTransformer("all-MiniLM-L6-v2")
embs    = sbert.encode(queries, convert_to_tensor=True)

# 3) Load your trained MLP
class SBERTClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.fc1  = torch.nn.Linear(input_dim, hidden_dim)
        self.drop = torch.nn.Dropout(dropout)
        self.fc2  = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SBERTClassifier(embs.shape[1]).to(device)
model.load_state_dict(torch.load("best_sbert_mlp.pt", map_location=device))
model.eval()

# 4) Predict probabilities and binary labels
with torch.no_grad():
    embs = embs.to(device)
    logits = model(embs)
    probs  = torch.sigmoid(logits).cpu().numpy()
    y_pred = (probs > 0.5).astype(int)

# 5) Compute metrics
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
print(f"Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
print("\nDetailed report:\n", classification_report(y_true, y_pred))

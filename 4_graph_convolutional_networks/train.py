import sys
import numpy as np
import torch
import torch.nn.functional as F

from data import load_cora, make_masks
from model import GCN

SEED = 42

# SAME HYPERPARAMETERS AS KIPF & WELLING (2017)
HIDDEN_DIM = 16
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x, y, edge_index, class_names = load_cora()
train_mask, val_mask, test_mask = make_masks(x.size(0))
x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)
train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

model = GCN(x.size(1), HIDDEN_DIM, len(class_names), DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def train_step():
    model.train()
    optimizer.zero_grad()
    logits = model(x, edge_index)
    # FULL BATCH: EVERY NODE IS FORWARDED, BUT ONLY THE 140 LABELLED ONES GO INTO THE LOSS
    loss = F.cross_entropy(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate():
    model.eval()
    pred = model(x, edge_index).argmax(dim=1)
    return [(pred[m] == y[m]).float().mean().item() for m in (train_mask, val_mask, test_mask)]


def macro_f1(pred, true, num_classes):
    scores = []
    for c in range(num_classes):
        tp = ((pred == c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(scores) / num_classes


history = []
for epoch in range(1, EPOCHS + 1):
    loss = train_step()
    train_acc, val_acc, test_acc = evaluate()
    history.append((loss, train_acc, val_acc, test_acc))

    if epoch == 1 or epoch % 10 == 0:
        print(f"epoch {epoch:03d} | loss {loss:.4f} | train {train_acc:.4f} | val {val_acc:.4f} | test {test_acc:.4f}")

model.eval()
with torch.no_grad():
    pred = model(x, edge_index).argmax(dim=1)

print()
print("accuracy  train {:.4f}  val {:.4f}  test {:.4f}".format(*history[-1][1:]))
print("macro-F1  train {:.4f}  val {:.4f}  test {:.4f}".format(
    *[macro_f1(pred[m], y[m], len(class_names)) for m in (train_mask, val_mask, test_mask)]))

if "--plot" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for k, (name, style) in enumerate([("train", "-"), ("val", "--"), ("test", ":")], start=1):
        ax.plot(epochs, [100 * h[k] for h in history], style, lw=1.5, label=f"{name} acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")

    ax2 = ax.twinx()
    ax2.plot(epochs, [h[0] for h in history], color="0.6", lw=1, label="train loss")
    ax2.set_ylabel("train loss")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig("training_curve.png", dpi=150)
    print("saved training_curve.png")

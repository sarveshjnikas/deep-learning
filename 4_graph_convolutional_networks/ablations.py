# WHAT ACTUALLY MATTERS IN A GCN? SAME DATA, SPLIT AND HYPERPARAMETERS AS train.py,
# ONLY THE PROPAGATION MATRIX OR THE DEPTH CHANGES. 10 SEEDS EACH, ~10 MIN ON CPU.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_cora, make_masks

SEEDS = range(10)
EPOCHS = 200

x, y, edge_index, class_names = load_cora()
masks = make_masks(x.size(0))
N = x.size(0)


def propagation_matrix(kind):
    A = torch.zeros(N, N)
    if kind == "duplicates":  # LET THE REPEATED EDGES ACCUMULATE (WEIGHT 2)
        A.index_put_((edge_index[0], edge_index[1]), torch.ones(edge_index.size(1)), accumulate=True)
    else:
        A[edge_index[0], edge_index[1]] = 1.0
    I = torch.eye(N)

    if kind == "no_graph":  # A^ = I, I.E. A PLAIN MLP
        return I
    if kind == "no_self_loops":
        d = A.sum(1).pow(-0.5)
        return d[:, None] * A * d[None]

    A_tilde = A + I
    if kind == "row_norm":  # MEAN OVER NEIGHBOURS
        return A_tilde / A_tilde.sum(1, keepdim=True)
    if kind == "unnormalised":
        return A_tilde
    d = A_tilde.sum(1).pow(-0.5)  # "symmetric" AND "duplicates"
    return d[:, None] * A_tilde * d[None]


class DeepGCN(nn.Module):
    def __init__(self, P, depth=2, hidden_dim=16):
        super().__init__()
        self.P = P
        dims = [x.size(1)] + [hidden_dim] * (depth - 1) + [len(class_names)]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, h):
        for k, layer in enumerate(self.layers):
            h = F.dropout(h, 0.5, self.training)
            h = self.P @ (h @ layer.weight.t()) + layer.bias
            if k < len(self.layers) - 1:
                h = F.relu(h)
        return h


def run(P, seed, depth=2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepGCN(P, depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        F.cross_entropy(model(x)[masks[0]], y[masks[0]]).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(1)
    return [(pred[m] == y[m]).float().mean().item() * 100 for m in masks]


def report(name, P, depth=2):
    r = np.array([run(P, s, depth) for s in SEEDS])
    mean, std = r.mean(0), r.std(0)
    print(f"{name:<28} train {mean[0]:5.1f} +- {std[0]:4.1f}  val {mean[1]:5.1f} +- {std[1]:4.1f}  "
          f"test {mean[2]:5.1f} +- {std[2]:4.1f}", flush=True)


print("train split class counts:", np.bincount(y[:140].numpy(), minlength=len(class_names)).tolist())
print("all nodes class counts:  ", np.bincount(y.numpy(), minlength=len(class_names)).tolist())
print()

for name, kind in [("symmetric norm (default)", "symmetric"),
                   ("no graph (MLP)", "no_graph"),
                   ("no self loops", "no_self_loops"),
                   ("row norm", "row_norm"),
                   ("unnormalised A + I", "unnormalised"),
                   ("duplicate edges counted", "duplicates")]:
    report(name, propagation_matrix(kind))

print()
P = propagation_matrix("symmetric")
for depth in (1, 3, 4, 6, 8):
    report(f"depth {depth}", P, depth)

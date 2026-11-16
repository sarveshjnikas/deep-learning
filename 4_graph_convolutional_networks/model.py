import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_adjacency(edge_index, num_nodes, device=None):
    # A^ = D~^(-1/2) (A + I) D~^(-1/2)

    # ASSIGN, DON'T ACCUMULATE: 151 CITATIONS APPEAR IN BOTH DIRECTIONS IN cora.cites,
    # SO AFTER SYMMETRISING THERE ARE 302 DUPLICATE EDGES THAT WOULD OTHERWISE COUNT TWICE
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A[edge_index[0], edge_index[1]] = 1.0

    # SELF LOOPS. WITHOUT THEM A NODE NEVER SEES ITS OWN FEATURES
    idx = torch.arange(num_nodes, device=device)
    A[idx, idx] = 1.0

    deg_inv_sqrt = A.sum(dim=1).pow(-0.5)  # EVERY DEGREE >= 1 BECAUSE OF THE SELF LOOP
    return deg_inv_sqrt.view(-1, 1) * A * deg_inv_sqrt.view(1, -1)


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.cached_edge_index = None
        self.cached_adj = None

    def forward(self, x, edge_index):
        # A^ ONLY DEPENDS ON THE GRAPH, SO BUILD IT ONCE AND REUSE IT EVERY EPOCH
        if self.cached_edge_index is not edge_index or self.cached_adj.device != x.device:
            self.cached_adj = normalized_adjacency(edge_index, x.size(0), x.device)
            self.cached_edge_index = edge_index

        # TRANSFORM FIRST, THEN AGGREGATE: X W IS N x 16, X IS N x 1433, SO A^ (X W) IS ~90x CHEAPER
        out = self.cached_adj @ (x @ self.linear.weight.t())

        # BIAS AFTER AGGREGATION. ROWS OF A^ DON'T SUM TO 1, SO A^ (XW + b) WOULD RESCALE b PER NODE
        if self.linear.bias is not None:
            out = out + self.linear.bias
        return out


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)  # RAW LOGITS, CROSS ENTROPY DOES THE SOFTMAX

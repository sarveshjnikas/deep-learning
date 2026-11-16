import numpy as np
import torch

CORA_DIR = "cora"

# FIXED SPLIT: FIRST 140 NODES FOR TRAINING, 200-499 VALIDATION, 500-1499 TEST
TRAIN_NODES = range(0, 140)
VAL_NODES = range(200, 500)
TEST_NODES = range(500, 1500)


def load_cora(cora_dir=CORA_DIR):
    content = np.genfromtxt(f"{cora_dir}/cora.content", dtype=str)

    node_ids = content[:, 0]
    features = content[:, 1:-1].astype(np.float32)
    label_names = content[:, -1]

    class_names = sorted(np.unique(label_names))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    labels = np.array([class_to_idx[name] for name in label_names], dtype=np.int64)

    # ROW-NORMALISE THE BAG OF WORDS SO LONG PAPERS DON'T GET BIGGER INPUTS
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    features = features / row_sums

    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    # CITATIONS ARE DIRECTED IN THE FILE, BUT WE TREAT THE GRAPH AS UNDIRECTED
    edge_list = []
    for cited, citing in np.genfromtxt(f"{cora_dir}/cora.cites", dtype=str):
        i, j = node_to_idx[cited], node_to_idx[citing]
        edge_list.append([i, j])
        edge_list.append([j, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    return x, y, edge_index, class_names


def make_masks(num_nodes):
    masks = []
    for nodes in (TRAIN_NODES, VAL_NODES, TEST_NODES):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[nodes.start:nodes.stop] = True
        masks.append(mask)
    return masks  # train, val, test

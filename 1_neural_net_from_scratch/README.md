# Neural Networks from Scratch

Feed-forward binary classifiers written in pure NumPy. No autograd and no ML frameworks: every gradient is derived by hand and implemented directly.

## Models

| Model | Architecture | Parameters |
|---|---|---|
| Model 1 | 4 → 1 (linear classifier) | 5 |
| Model 2 | 4 → 32 → 1 | 193 |
| Model 3 | 4 → 32 → 16 → 1 | 705 |

All three use sigmoid activations, binary cross-entropy loss, and mini-batch gradient descent (batch size 32, 20 epochs).

## Results

Test F1 on a held-out 20% split:

| Model | Test F1 |
|---|---|
| Model 1 | 0.9863 |
| Model 2 | 1.0000 |
| Model 3 | 1.0000 |

The data is nearly linearly separable, so a single neuron already scores ~0.99 and the deeper models have little left to capture.

## The backward pass

For a sigmoid output with binary cross-entropy, the sigmoid's derivative cancels the loss's denominator, leaving just the prediction error:

$$\frac{\partial L}{\partial z} = \frac{\hat y - y}{N}$$

Every layer then applies the same three steps:

- weight gradient: $\partial L / \partial W = A_{\text{in}}^\top \delta$
- bias gradient: $\partial L / \partial b = \sum_i \delta_i$
- gradient passed back: $\delta_{\text{prev}} = (\delta W^\top) \odot \sigma'(z_{\text{prev}})$

## Run

```bash
python3 src/train.py
```

Requires Python 3, NumPy and pandas.

## Dataset

[Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication), UCI Machine Learning Repository (CC BY 4.0).

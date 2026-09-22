# ResNet-18 from Scratch on Food-101

Implementation of ResNet-18 (He et al., 2016), trained from scratch — no pretrained
weights, no `torchvision.models` — on the Food-101 dataset (101 food classes).

## Key idea

Stacking plain conv layers eventually makes networks *harder* to train — training
error rises, which is an optimization failure, not overfitting. In principle extra
layers should never hurt (worst case, they learn identity and match a shallower
network), but SGD struggles to fit identity through a stack of conv+ReLU layers.

The fix: instead of learning $H(x)$ directly, each block learns $F(x) = H(x) - x$,
and a shortcut connection adds $x$ back. Learning "do nothing" now just means
pushing weights toward zero, which SGD is already good at.

## Architecture

4 stages of 2 residual blocks each (8 blocks total), every block:
`conv → batchnorm → relu → conv → batchnorm`, then `+ shortcut`, then `relu`.

Where a block changes spatial size or channel count (the first block of stages
2/3/4), the shortcut can't be a plain identity — the input is projected with a
$1\times1$ convolution + batchnorm to match shape before the addition.

Cross-entropy loss is implemented from scratch (log-sum-exp with max-subtraction
for numerical stability), not `nn.CrossEntropyLoss`.

## Results

Train/test performance, Accuracy and Macro-F1, across three configurations:

| Configuration | Train Acc | Train F1 | Test Acc | Test F1 |
|---|---|---|---|---|
| No augmentation, 20 epochs | 92.17% | 0.9240 | 50.12% | 0.4978 |
| + Augmentation, 20 epochs | 69.88% | 0.7003 | 66.62% | 0.6681 |
| + Val split, LR schedule, early stopping | 93.48% | 0.9347 | 77.07% | 0.7709 |

Without augmentation the model overfits fast (large train/test gap). Adding
augmentation alone wasn't enough epochs to converge. The full setup (augmentation +
held-out validation split + `ReduceLROnPlateau` + early stopping, up to 100 epochs)
reaches ~77% test accuracy — a reasonable result training entirely from scratch,
no pretrained weights.

## Run

```bash
pip install torch torchvision scikit-learn pillow
python train.py
```

Expects the Food-101 dataset placed alongside as `food-101/`, so
`food-101/meta/train.txt`, `food-101/meta/test.txt`, `food-101/meta/classes.txt`,
and `food-101/images/<class>/<id>.jpg` all exist. A GPU is strongly recommended —
this was trained on an NVIDIA A100.

Outputs: per-epoch training loss and validation accuracy/Macro-F1 to stdout,
`best_model.pt` (best-validation checkpoint), and `testOutput.csv` (one predicted
class per test image, same order as `food-101/meta/test.txt`).

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) —
L. Bossard, M. Guillaumin, L. Van Gool, "Food-101 -- Mining Discriminative
Components with Random Forests," ECCV 2014.

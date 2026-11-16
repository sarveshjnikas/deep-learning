# BUILDS THE RESULTS TABLE, LOSS CURVES AND SE WEIGHT ANALYSIS FROM results/*.json AND THE SAVED SE CHECKPOINT
import glob, json, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_loaders
from model import CNN, SEBlock
from train import get_device

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
R = {}
for f in glob.glob("results/*.json"):
    with open(f) as fh:
        r = json.load(fh); R[r["name"]] = r
out = open("results/summary.txt", "w")
def say(s=""): print(s); out.write(s + "\n")

# 1. LR SELECTION (BASELINE, SEED 0, VAL ONLY)
lrs = sorted({r["args"]["lr"] for r in R.values() if r["args"]["model"] == "base" and r["args"]["seed"] == 0})
say("LR SELECTION (baseline, seed 0, best-val checkpoint)")
for lr in lrs:
    say(f"  lr {lr:g}: val acc {R[f'base_lr{lr:g}_s0']['val_acc']:.2%} | val F1 {R[f'base_lr{lr:g}_s0']['val_f1']:.4f}")
LR = max(lrs, key=lambda lr: (R[f"base_lr{lr:g}_s0"]["val_acc"], R[f"base_lr{lr:g}_s0"]["val_f1"]))  # TIES -> VAL MACRO-F1
say(f"  -> chosen lr {LR:g}\n")

def runs(prefix): return [R[f"{prefix}_lr{LR:g}_s{s}"] for s in range(3) if f"{prefix}_lr{LR:g}_s{s}" in R]
def ms(v): return f"{100*np.mean(v):.2f} +- {100*np.std(v, ddof=1):.2f}"   # SAMPLE STD (n-1)

# 2. MAIN TABLE + 3. ABLATION ON r
say("MODEL | params | val acc | test acc | test macro-F1 (x100) | mean +- sample std over seeds")
for label, prefix in [("baseline", "base"), ("SE r=2", "se_r2"), ("SE r=4", "se_r4"), ("SE r=8", "se_r8")]:
    rs = runs(prefix)
    if rs:
        say(f"{label:9s} | {rs[0]['params']:,} | {ms([r['val_acc'] for r in rs])} | {ms([r['test_acc'] for r in rs])} | "
            f"{ms([r['test_f1'] for r in rs])} | n={len(rs)} | {np.mean([r['minutes'] for r in rs]):.1f} min/run")
b, s = runs("base"), runs("se_r4")
d = [y["test_acc"] - x["test_acc"] for x, y in zip(b, s)]
say(f"\nPER-SEED TEST GAIN (SE r=4 - baseline): {[round(100*v, 2) for v in d]} -> mean {100*np.mean(d):.2f} points\n")

# 4. LOSS CURVES (MEAN OVER SEEDS)
plt.figure(figsize=(6.5, 3.5))
for rs, lab, c in [(b, "Baseline", "C0"), (s, "SE (r=4)", "C3")]:
    tl = np.mean([[h["train_loss"] for h in r["history"]] for r in rs], 0)
    vl = np.mean([[h["val_loss"] for h in r["history"]] for r in rs], 0)
    ep = np.arange(1, len(tl) + 1)
    plt.plot(ep, tl, "--", c=c, lw=1, label=f"{lab} train"); plt.plot(ep, vl, "-", c=c, lw=1.3, label=f"{lab} val")
    say(f"{lab}: mean-curve min val loss {vl.min():.3f} | mean per-seed best val loss {np.mean([min(h['val_loss'] for h in r['history']) for r in rs]):.3f} | final train loss {tl[-1]:.3f}")
plt.xlabel("epoch"); plt.ylabel("cross-entropy")
plt.legend(frameon=False, ncol=2); plt.tight_layout(); plt.savefig("loss_curves.png", dpi=150)

# 5. WHAT THE SE BLOCKS LEARNED: CLASS-MEAN EXCITATION WEIGHTS ON THE TEST SET
dev = get_device()
_, _, te, classes = get_loaders(DATA, 128, 32, 0, workers=0)  # NO WORKER PROCESSES: SCRIPT HAS NO __main__ GUARD
m = CNN(len(classes), se=True, r=4).to(dev)
m.load_state_dict(torch.load(f"results/se_r4_lr{LR:g}_s0.pt", map_location=dev)); m.eval()
ses = [x for x in m.modules() if isinstance(x, SEBlock)]
W = [[] for _ in ses]
hooks = [se.excite.register_forward_hook(lambda mod, i, o, k=k: W[k].append(o.cpu())) for k, se in enumerate(ses)]
ys = []
with torch.no_grad():
    for x, y in te:
        m(x.to(dev)); ys.append(y)
for h in hooks: h.remove()
ys = torch.cat(ys)
say("SE WEIGHT CLASS-DEPENDENCE BY DEPTH, SEED-0 CHECKPOINT (mean over channels of the std across classes of class-mean weights)")
cm_last = None
for k, w in enumerate(W):
    w = torch.cat(w)                                                  # (N_TEST, C)
    cm = torch.stack([w[ys == c].mean(0) for c in range(len(classes))])  # (21, C)
    say(f"  SE block {k+1} ({cm.shape[1]} ch): class-std {cm.std(0).mean():.4f} | mean weight {w.mean():.3f} | ratio {cm.std(0).mean()/w.mean():.3f}")
    cm_last = cm
plt.figure(figsize=(8, 5))
plt.imshow(cm_last.numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
plt.yticks(range(len(classes)), classes, fontsize=8)
plt.xlabel("channel of last SE block"); plt.colorbar(label="mean excitation weight")
plt.tight_layout(); plt.savefig("se_weights.png", dpi=150)
say("\nfigures: loss_curves.png, se_weights.png")
out.close()

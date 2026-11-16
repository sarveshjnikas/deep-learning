import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from data import get_loaders
from model import CNN, n_params


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, dev, opt=None):
    # ONE PASS OVER loader. TRAINS IF opt IS GIVEN, OTHERWISE JUST EVALUATES
    train = opt is not None
    model.train(train)
    tot_loss, preds, ys = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = F.cross_entropy(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(y)
            preds.append(out.argmax(1).cpu()); ys.append(y.cpu())
    p, y = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return tot_loss / len(y), float((p == y).mean()), float(f1_score(y, p, average="macro")), p, y


def train(model, tr, va, dev, lr, epochs, wd=1e-4):
    # FIXED EPOCH BUDGET + COSINE LR; KEEP THE CHECKPOINT WITH THE LOWEST VAL LOSS (SAME RULE FOR EVERY MODEL)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist, best, best_state = [], float("inf"), None
    for ep in range(epochs):
        t0 = time.time()
        tl, ta, _, _, _ = run_epoch(model, tr, dev, opt)
        vl, vacc, _, _, _ = run_epoch(model, va, dev)
        sched.step()
        hist.append(dict(epoch=ep + 1, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=vacc))
        if vl < best:
            best, best_state = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"ep {ep+1:02d}/{epochs} | train {tl:.3f} {ta:.2%} | val {vl:.3f} {vacc:.2%} | {time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(best_state)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", choices=["base", "se"], default="base")
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", action="store_true", help="save best weights (for analysis.py)")
    a = ap.parse_args()

    set_seed(a.seed)
    dev = get_device()
    tr, va, te, classes = get_loaders(a.data, a.size, a.bs, a.seed)
    model = CNN(len(classes), se=a.model == "se", r=a.r).to(dev)
    name = a.tag or f"{a.model}{'_r'+str(a.r) if a.model=='se' else ''}_lr{a.lr:g}_s{a.seed}"
    print(f"{name} | {n_params(model):,} params | device {dev}")

    t0 = time.time()
    hist = train(model, tr, va, dev, a.lr, a.epochs)
    _, val_acc, val_f1, _, _ = run_epoch(model, va, dev)
    _, test_acc, test_f1, p, y = run_epoch(model, te, dev)
    res = dict(name=name, args=vars(a), params=n_params(model), minutes=(time.time() - t0) / 60,
               val_acc=val_acc, val_f1=val_f1, test_acc=test_acc, test_f1=test_f1, history=hist)
    print(f"BEST-VAL CHECKPOINT | val {val_acc:.2%} | test acc {test_acc:.2%} | test macro-F1 {test_f1:.4f}")

    out = Path("results"); out.mkdir(exist_ok=True)
    with open(out / f"{name}.json", "w") as f:
        json.dump(res, f, indent=1)
    if a.save:
        torch.save(model.state_dict(), out / f"{name}.pt")


if __name__ == "__main__":
    main()

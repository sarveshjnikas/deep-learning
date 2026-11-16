from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Rot90:
    """RANDOM ROTATION BY A MULTIPLE OF 90 DEG. AERIAL IMAGES HAVE NO 'UP', SO THE CLASS IS UNCHANGED"""
    def __call__(self, x):
        return torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(1, 2))


def channel_stats(root, size):
    # PER-CHANNEL MEAN/STD OF THE TRAIN SET ONLY (NO LEAKAGE FROM VAL/TEST)
    ds = datasets.ImageFolder(Path(root) / "train", transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()]))
    s, s2, n = torch.zeros(3), torch.zeros(3), 0
    for x, _ in ds:
        s += x.sum((1, 2)); s2 += (x ** 2).sum((1, 2)); n += x[0].numel()
    m = s / n
    return m.tolist(), (s2 / n - m ** 2).sqrt().tolist()


def get_loaders(root, size=128, bs=32, seed=0, workers=2):
    m, sd = channel_stats(root, size)
    base = [transforms.Resize((size, size)), transforms.ToTensor()]
    norm = transforms.Normalize(m, sd)
    tf_train = transforms.Compose(base + [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), Rot90(), norm])
    tf_eval = transforms.Compose(base + [norm])                      # NO AUGMENTATION ON VAL/TEST

    root = Path(root)
    tr = datasets.ImageFolder(root / "train", tf_train)
    va = datasets.ImageFolder(root / "val", tf_eval)
    te = datasets.ImageFolder(root / "test", tf_eval)
    assert tr.classes == va.classes == te.classes

    g = torch.Generator().manual_seed(seed)                          # SAME SHUFFLE ORDER FOR EVERY MODEL WITH THIS SEED
    kw = dict(batch_size=bs, num_workers=workers, persistent_workers=workers > 0)
    return (DataLoader(tr, shuffle=True, generator=g, **kw),
            DataLoader(va, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw),
            tr.classes)

import torch.nn as nn


class SEBlock(nn.Module):
    """SQUEEZE-AND-EXCITATION (HU ET AL., 2018): LETS THE CNN DECIDE WHICH CHANNELS MATTER FOR THIS IMAGE
    1. SQUEEZE: EACH CHANNEL'S FEATURE MAP IS COMPRESSED TO ONE NUMBER BY GLOBAL AVERAGE POOLING
    2. EXCITATION: A BOTTLENECK MLP (C -> C/r -> C) + SIGMOID LEARNS A WEIGHT IN (0,1) PER CHANNEL
    3. SCALE: EACH CHANNEL'S FEATURE MAP IS MULTIPLIED BY ITS WEIGHT"""
    def __init__(self, channels, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // r), nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels), nn.Sigmoid())

    def forward(self, x):
        w = self.excite(self.squeeze(x).flatten(1))   # (B,C,H,W) -> (B,C) -> (B,C) WEIGHTS
        return x * w[:, :, None, None]                # BROADCAST: EVERY MAP SCALED BY ITS OWN WEIGHT


class CNN(nn.Module):
    """BASELINE CNN: 4 x [CONV3x3 -> BN -> RELU -> (SE) -> MAXPOOL], THEN GAP -> DROPOUT -> FC.
    se=False IS THE BASELINE, se=True ADDS ONE SE BLOCK PER CONV BLOCK. NOTHING ELSE DIFFERS"""
    def __init__(self, n_classes=21, chans=(32, 64, 128, 256), se=False, r=4, p_drop=0.3):
        super().__init__()
        layers, c_in = [], 3
        for c in chans:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            if se:
                layers.append(SEBlock(c, r))
            layers.append(nn.MaxPool2d(2))
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p_drop), nn.Linear(c_in, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def n_params(model):
    return sum(p.numel() for p in model.parameters())

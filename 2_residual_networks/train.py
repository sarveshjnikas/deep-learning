import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOBAL_SEED = 3498

BATCH_SIZE = 256 # TAKEN SAME AS THE ORIGINAL PAPER
MAX_EPOCHS = 100 # ASSIGNMENT SPEC
MOMENTUM = 0.9 # TAKEN SAME AS THE ORIGINAL PAPER
WEIGHT_DECAY = 1e-4 # TAKEN SAME AS THE ORIGINAL PAPER

LEARNING_RATE = 0.1
LEARNING_RATE_SHRINK_FACTOR = 0.1
LEARNING_RATE_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406] # THIS
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

class Food101Dataset(Dataset):
    def __init__(self, image_paths, transform):
        self.transform = transform
        self.image_paths = image_paths

        with open("food-101/meta/classes.txt", "r") as f:
            food_classes = [line.strip() for line in f.readlines()]
            self.food_class_map = {food_class: idx for idx, food_class in enumerate(food_classes)} # CACHE THE FOOD CLASS MAP FOR LATER USE
        pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        class_name, id = image_path.split("/")
        path = f"food-101/images/{class_name}/{id}.jpg"
        img = Image.open(path)
        img = img.convert("RGB")
        tensor_img = self.transform(img)
        label = self.food_class_map[class_name]
        return tensor_img, label

def split_train_val(train_file, val_per_class):
    with open(train_file, "r") as f:
        all_lines = [line.strip() for line in f.readlines()]

    by_class = {}
    for line in all_lines:
        class_name = line.split('/')[0]
        by_class.setdefault(class_name, []).append(line)

    train_lines = []
    val_lines = []

    rng = random.Random(GLOBAL_SEED)
    for class_name, class_lines in by_class.items():
        shuffled = class_lines[:]
        rng.shuffle(shuffled)
        val_lines.extend(shuffled[:val_per_class])
        train_lines.extend(shuffled[val_per_class:])
    return train_lines, val_lines

tr_image_paths, val_image_paths = split_train_val("food-101/meta/train.txt", 75)

with open("food-101/meta/test.txt", "r") as f:
    te_image_paths = [line.strip() for line in f.readlines()]

tr_dataset = Food101Dataset(tr_image_paths, train_transforms)
tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)

val_dataset = Food101Dataset(val_image_paths, test_transforms)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

te_dataset = Food101Dataset(te_image_paths, test_transforms)
te_loader = DataLoader(te_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

tr_deter_dataset = Food101Dataset(tr_image_paths, test_transforms) # USED FOR FINAL EVALUATION ON TRAINING SET. NO DATA AUGMENTATION.
tr_deter_loader = DataLoader(tr_deter_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 =  nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        # DOWNSAMPLING.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        conv1_out = self.conv1(x)
        bn1_out = self.bn1(conv1_out)
        relu1_out = self.relu1(bn1_out)
        conv2_out = self.conv2(relu1_out)
        bn2_out = self.bn2(conv2_out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu2(bn2_out + identity)

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.body = nn.Sequential(
            ResNetBlock(64, 64, stride=1),
            ResNetBlock(64, 64, stride=1),

            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 128, stride=1),

            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 256, stride=1),

            ResNetBlock(256, 512, stride=2),
            ResNetBlock(512, 512, stride=1)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 101)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.body(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def loss(self, y_pred, y_true):
        max_logits = y_pred.max(dim=1, keepdim=True).values
        stabilized = y_pred - max_logits # FOR NUMERICAL SAFETY

        log_sum_exp = torch.log(torch.exp(stabilized).sum(dim=1, keepdim=True))
        log_probs = stabilized - log_sum_exp

        correct_class_log_probs = log_probs.gather(1, y_true.unsqueeze(1)).squeeze(1)
        return -correct_class_log_probs.mean()

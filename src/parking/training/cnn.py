"""주차칸 crop → occupied/empty CNN 분류 학습."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parking.app.paths import ROOT
from parking.training.prepare import MODELS_DIR, build_cnn_crops, next_model_path

OUT_DIR = MODELS_DIR / "cnn_occupancy"


class OccupancyCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _loaders(data_dir: Path, batch: int):
    tf_train = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ]
    )
    train_ds = datasets.ImageFolder(data_dir / "train", transform=tf_train)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=tf_val)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds.classes


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train_cnn(
    *,
    epochs: int = 5,
    batch: int = 32,
    max_images: int = 100,
    lr: float = 1e-3,
    log=print,
    should_stop=None,
) -> Path:
    data_dir = build_cnn_crops(max_images, log)
    train_loader, val_loader, classes = _loaders(data_dir, batch)
    log(f"[CNN] classes={classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[CNN] device={device}")
    model = OccupancyCNN(num_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = next_model_path(OUT_DIR, "cnn")
    log(f"[CNN] 저장 파일: {best_path.name}")

    for ep in range(1, epochs + 1):
        if should_stop and should_stop():
            log("[CNN] 중지 요청으로 중단")
            break
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            if should_stop and should_stop():
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = running / max(1, n)
        val_acc = _eval(model, val_loader, device)
        log(f"[CNN] epoch {ep}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc:.3f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                    "val_acc": best_acc,
                    "img_size": 96,
                },
                best_path,
            )
            log(f"[CNN] best 저장 → {best_path} (acc={best_acc:.3f})")

    note = best_path.with_suffix(".txt")
    note.write_text(
        f"OccupancyCNN\nfile: {best_path.name}\nval_acc: {best_acc:.4f}\n",
        encoding="utf-8",
    )
    log(f"[CNN] 완료 best_acc={best_acc:.3f}")
    return best_path

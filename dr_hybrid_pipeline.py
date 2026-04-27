"""
Hybrid DR pipeline inspired by:
ViT + EfficientNet + PSO-based feature selection + Fuzzy classification.

This is a practical starter implementation for diabetic retinopathy experiments.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


@dataclass
class Config:
    data_dir: str = "data"
    image_size: int = 224
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 1e-4
    epochs: int = 3
    fused_dim: int = 512
    selected_features: int = 256
    num_classes: int = 5
    seed: int = 42
    out_dir: str = "artifacts"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class HybridFeatureModel(nn.Module):
    def __init__(self, fused_dim: int, num_classes: int) -> None:
        super().__init__()
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.eff_features = eff.features
        self.eff_pool = nn.AdaptiveAvgPool2d(1)
        self.eff_dim = 1280

        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vit.heads = nn.Identity()
        self.vit = vit
        self.vit_dim = 768

        self.fusion = nn.Sequential(
            nn.Linear(self.eff_dim + self.vit_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eff_map = self.eff_features(x)
        eff_vec = self.eff_pool(eff_map).flatten(1)
        vit_vec = self.vit(x)
        fused = self.fusion(torch.cat([eff_vec, vit_vec], dim=1))
        logits = self.classifier(fused)
        return logits, fused


class PSOFeatureSelector:
    """
    PSO placeholder:
    - If pyswarms is installed, use it.
    - Otherwise, fall back to top-variance feature selection.
    """

    def __init__(self, n_select: int) -> None:
        self.n_select = n_select
        self.indices_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "PSOFeatureSelector":
        try:
            import pyswarms as ps  # type: ignore
        except Exception:
            var = np.var(features, axis=0)
            self.indices_ = np.argsort(var)[::-1][: self.n_select]
            return self

        d = features.shape[1]
        n_select = min(self.n_select, d)

        def objective(x: np.ndarray) -> np.ndarray:
            # Lower is better; encourage binary-ish masks with n_select ones.
            penalties = []
            for row in x:
                mask = row > 0.5
                count_penalty = abs(mask.sum() - n_select)
                spread_penalty = -np.var(features[:, mask]) if mask.any() else 1e6
                penalties.append(count_penalty + spread_penalty)
            return np.array(penalties, dtype=np.float64)

        optimizer = ps.single.GlobalBestPSO(
            n_particles=20,
            dimensions=d,
            options={"c1": 0.5, "c2": 0.3, "w": 0.9},
            bounds=(np.zeros(d), np.ones(d)),
        )
        _, best_pos = optimizer.optimize(objective, iters=30, verbose=False)
        mask = best_pos > 0.5
        if mask.sum() < n_select:
            order = np.argsort(best_pos)[::-1]
            idx = order[:n_select]
        else:
            idx = np.where(mask)[0][:n_select]
        self.indices_ = idx
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.indices_ is None:
            raise RuntimeError("PSOFeatureSelector is not fitted.")
        return features[:, self.indices_]


class FuzzyDRClassifier:
    """
    A simple Gaussian-membership fuzzy classifier.
    Each class has per-feature mean/std; class score is average membership.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.means_: List[np.ndarray] = []
        self.stds_: List[np.ndarray] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FuzzyDRClassifier":
        self.means_.clear()
        self.stds_.clear()
        eps = 1e-6
        for c in range(self.num_classes):
            xc = x[y == c]
            if len(xc) == 0:
                self.means_.append(np.zeros(x.shape[1], dtype=np.float32))
                self.stds_.append(np.ones(x.shape[1], dtype=np.float32))
                continue
            self.means_.append(np.mean(xc, axis=0))
            self.stds_.append(np.std(xc, axis=0) + eps)
        return self

    @staticmethod
    def _gaussian_mf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.means_:
            raise RuntimeError("FuzzyDRClassifier is not fitted.")
        scores = []
        for c in range(self.num_classes):
            mf = self._gaussian_mf(x, self.means_[c], self.stds_[c])
            scores.append(np.mean(mf, axis=1))
        return np.argmax(np.stack(scores, axis=1), axis=1)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def extract_features(
    model: HybridFeatureModel, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            _, fused = model(xb)
            feats.append(fused.detach().cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def train_feature_model(cfg: Config, device: torch.device) -> HybridFeatureModel:
    train_ds = datasets.ImageFolder(
        root=str(Path(cfg.data_dir) / "train"),
        transform=build_transforms(cfg.image_size, train=True),
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    model = HybridFeatureModel(cfg.fused_dim, cfg.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= max(len(train_ds), 1)
        print(f"epoch={epoch+1}/{cfg.epochs} loss={epoch_loss:.4f}")
    return model


def run_pipeline(cfg: Config) -> None:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = train_feature_model(cfg, device)

    train_ds = datasets.ImageFolder(
        root=str(Path(cfg.data_dir) / "train"),
        transform=build_transforms(cfg.image_size, train=False),
    )
    val_ds = datasets.ImageFolder(
        root=str(Path(cfg.data_dir) / "val"),
        transform=build_transforms(cfg.image_size, train=False),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    x_train, y_train = extract_features(model, train_loader, device)
    x_val, y_val = extract_features(model, val_loader, device)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    selector = PSOFeatureSelector(n_select=cfg.selected_features).fit(x_train)
    x_train_sel = selector.transform(x_train)
    x_val_sel = selector.transform(x_val)

    fuzzy = FuzzyDRClassifier(num_classes=cfg.num_classes).fit(x_train_sel, y_train)
    y_pred = fuzzy.predict(x_val_sel)
    val_acc = accuracy(y_val, y_pred)
    print(f"fuzzy_val_accuracy={val_acc:.4f}")

    torch.save(model.state_dict(), out_dir / "hybrid_feature_model.pt")
    np.save(out_dir / "selected_feature_indices.npy", selector.indices_)
    np.savez_compressed(out_dir / "fuzzy_stats.npz")

    metadata = {
        "config": asdict(cfg),
        "val_accuracy": val_acc,
        "num_train_samples": int(len(train_ds)),
        "num_val_samples": int(len(val_ds)),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"saved_artifacts={out_dir.resolve()}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fused-dim", type=int, default=512)
    parser.add_argument("--selected-features", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    args = parser.parse_args()
    return Config(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        epochs=args.epochs,
        fused_dim=args.fused_dim,
        selected_features=args.selected_features,
        num_classes=args.num_classes,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    config = parse_args()
    run_pipeline(config)

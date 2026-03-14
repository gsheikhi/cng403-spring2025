"""
CNG403 Assignment 1 — Reference Implementation
================================================
A complete FFNN pipeline built with PyTorch's nn.Module.

Students: do NOT modify this file.
- Study it to understand the expected behaviour.
- Your from-scratch implementation must produce consistent results.
- To run: python reference.py --config ../configs/reference_config.json
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.feature import hog
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_hog_features(images: np.ndarray, hog_cfg: dict) -> np.ndarray:
    """
    Convert a batch of grayscale images (N, H, W) to HOG feature vectors.

    Args:
        images:  numpy array of shape (N, H, W), values in [0, 1].
        hog_cfg: dict with keys orientations, pixels_per_cell, cells_per_block.

    Returns:
        features: numpy array of shape (N, D).
    """
    features = []
    for img in images:
        feat = hog(
            img,
            orientations=hog_cfg["orientations"],
            pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
            cells_per_block=tuple(hog_cfg["cells_per_block"]),
            channel_axis=None,
        )
        features.append(feat)
    return np.stack(features)


def load_and_extract(data_cfg: dict) -> tuple:
    """
    Download MNIST, extract features, return torch tensors.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test  — all torch.Tensor
    """
    root = data_cfg["root"]
    hog_cfg = data_cfg["hog"]
    val_split = data_cfg["val_split"]

    to_tensor = transforms.ToTensor()
    train_raw = datasets.MNIST(root, train=True,  download=True, transform=to_tensor)
    test_raw  = datasets.MNIST(root, train=False, download=True, transform=to_tensor)

    def dataset_to_numpy(ds):
        imgs   = ds.data.numpy().astype(np.float32) / 255.0   # (N, 28, 28)
        labels = ds.targets.numpy()
        return imgs, labels

    train_imgs, train_labels = dataset_to_numpy(train_raw)
    test_imgs,  test_labels  = dataset_to_numpy(test_raw)

    print("Extracting HOG features (train)…")
    X_train_full = extract_hog_features(train_imgs, hog_cfg)
    print("Extracting HOG features (test)…")
    X_test       = extract_hog_features(test_imgs,  hog_cfg)

    # Standardise using train statistics
    mean = X_train_full.mean(axis=0)
    std  = X_train_full.std(axis=0) + 1e-8
    X_train_full = (X_train_full - mean) / std
    X_test       = (X_test       - mean) / std

    # Train / validation split
    n_val = int(len(X_train_full) * val_split)
    n_train = len(X_train_full) - n_val
    idx = np.random.permutation(len(X_train_full))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X_train_full[train_idx], train_labels[train_idx]
    X_val,   y_val   = X_train_full[val_idx],   train_labels[val_idx]

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    to_l = lambda a: torch.tensor(a, dtype=torch.long)

    return to_t(X_train), to_l(y_train), to_t(X_val), to_l(y_val), to_t(X_test), to_l(test_labels)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(input_dim: int, cfg: dict) -> nn.Sequential:
    """
    Build an FFNN from the model section of the config.

    Architecture:  Input → [Linear → Activation]* → Linear → (softmax in loss)

    Args:
        input_dim: dimensionality of the feature vector.
        cfg:       model config dict.

    Returns:
        nn.Sequential model.
    """
    activations = {
        "relu":    nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh":    nn.Tanh(),
    }
    act_fn = activations[cfg["activation"]]

    layers = []
    prev_dim = input_dim
    for h in cfg["hidden_sizes"]:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(act_fn)
        if cfg.get("dropout", 0.0) > 0.0:
            layers.append(nn.Dropout(cfg["dropout"]))
        prev_dim = h
    layers.append(nn.Linear(prev_dim, 10))  # 10 MNIST classes

    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    l2_lambda: float,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full pass over the training set.

    Returns:
        avg_loss: mean cross-entropy loss over all batches.
        accuracy: fraction of correctly classified samples.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        # L2 regularisation (weight decay applied manually for transparency)
        if l2_lambda > 0:
            l2 = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
            loss = loss + l2_lambda * l2

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on a data loader (no gradient computation).

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, metrics: dict, path: str) -> None:
    torch.save({"model_state": model.state_dict(), "metrics": metrics}, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(model: nn.Module, path: str) -> dict:
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    return ckpt["metrics"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Data ----------------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_extract(cfg["data"])
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=256)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=256)

    # -- Model ---------------------------------------------------------------
    input_dim = X_train.shape[1]
    model = build_model(input_dim, cfg["model"]).to(device)
    print(f"\nModel:\n{model}\n")

    # -- Optimiser & loss ----------------------------------------------------
    tcfg = cfg["training"]
    optimizer = optim.SGD(
        model.parameters(),
        lr=tcfg["learning_rate"],
        momentum=tcfg.get("momentum", 0.0),
        weight_decay=0.0,   # handled manually in train_one_epoch
    )
    criterion = nn.CrossEntropyLoss()

    # -- Training loop -------------------------------------------------------
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_ckpt    = str(ckpt_dir / "reference_best.pt")
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, tcfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, tcfg["l2_lambda"], device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:3d}/{tcfg['epochs']}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, {"epoch": epoch, "val_acc": val_acc}, best_ckpt)

    # -- Test evaluation -----------------------------------------------------
    load_checkpoint(model, best_ckpt)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest  loss={test_loss:.4f}  acc={test_acc:.4f}")

    # Save history for notebook comparison
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(log_dir / "reference_history.npy"), history)
    print(f"Training history saved → {log_dir / 'reference_history.npy'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNG403 A1 — Reference FFNN")
    parser.add_argument(
        "--config",
        default="../configs/reference_config.json",
        help="Path to reference_config.json",
    )
    args = parser.parse_args()
    run(args.config)

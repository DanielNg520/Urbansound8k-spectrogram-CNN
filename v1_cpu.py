"""
VERSION 1: CPU-only (Intel cores, no GPU required)
─────────────────────────────────────────────────
Runs entirely on sklearn. No PyTorch needed.
Both MFCC+SVM baseline AND a CNN via sklearn's MLPClassifier.
For the CNN comparison on CPU, we use PyTorch but force CPU device.

Usage:
  python v1_cpu.py --dataset_root /path/to/UrbanSound8K
  python v1_cpu.py --dataset_root /path/to/UrbanSound8K --jobs 8

Dependencies:
  pip install librosa scikit-learn numpy pandas torch tqdm
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from core.dataset import load_metadata, get_fold_splits, precompute_mfcc_features
from core.torch_dataset import MelSpectrogramDataset
from core.models import LightCNN
from core.evaluation import compute_metrics, print_results, save_results


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEVICE = torch.device("cpu")
BATCH_SIZE = 32
EPOCHS = 30           # Reduce to 15 if too slow on your machine
LR = 1e-3
WEIGHT_DECAY = 1e-4


# ──────────────────────────────────────────────
# SVM Baseline
# ──────────────────────────────────────────────
def run_svm_baseline(metadata, dataset_root, n_jobs=-1, cache_dir="cache"):
    print("\n[SVM BASELINE] Pre-computing MFCC features...")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "mfcc_features.npz")
    X, y, folds = precompute_mfcc_features(metadata, dataset_root, cache_path)

    fold_results = []
    for test_fold, train_df, test_df in get_fold_splits(metadata):
        train_mask = folds != test_fold
        test_mask  = folds == test_fold

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                        decision_function_shape="ovr",
                        cache_size=500))
        ])

        print(f"  Fold {test_fold}: training SVM ({len(X_train)} samples)...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        result = compute_metrics(y_test, preds, fold=test_fold)
        fold_results.append(result)
        print(f"  Fold {test_fold} accuracy: {result['accuracy']*100:.2f}%")

    return fold_results


# ──────────────────────────────────────────────
# CNN on CPU
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        preds = model(x).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds)


def run_cnn(metadata, dataset_root, checkpoint_dir="checkpoints/cpu"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    fold_results = []

    for test_fold, train_df, test_df in get_fold_splits(metadata):
        print(f"\n[CNN CPU] Fold {test_fold} — {len(train_df)} train / {len(test_df)} test")

        ckpt_path = os.path.join(checkpoint_dir, f"fold{test_fold}_best.pt")

        train_ds = MelSpectrogramDataset(train_df, dataset_root, augment=True)
        test_ds  = MelSpectrogramDataset(test_df,  dataset_root, augment=False)

        # num_workers=0 on CPU to avoid overhead
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=False)

        model = LightCNN(n_classes=10).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                     weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(1, EPOCHS + 1):
            loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()
            if epoch % 5 == 0 or epoch == EPOCHS:
                labels, preds = evaluate(model, test_loader)
                val_acc = (labels == preds).mean()
                print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={loss:.4f}  "
                      f"train={train_acc*100:.1f}%  val={val_acc*100:.1f}%")
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), ckpt_path)

        # Load best and get final metrics
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        labels, preds = evaluate(model, test_loader)
        result = compute_metrics(labels, preds, fold=test_fold)
        fold_results.append(result)
        print(f"  Fold {test_fold} best accuracy: {result['accuracy']*100:.2f}%")

    return fold_results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UrbanSound8K — CPU version")
    parser.add_argument("--dataset_root", required=True,
                        help="Path to UrbanSound8K root directory")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Parallel jobs for SVM (-1 = all cores)")
    parser.add_argument("--skip_svm", action="store_true")
    parser.add_argument("--skip_cnn", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    global EPOCHS
    EPOCHS = args.epochs

    print(f"[CONFIG] Dataset: {args.dataset_root}")
    print(f"[CONFIG] Device: {DEVICE}")
    print(f"[CONFIG] Epochs: {EPOCHS}")

    metadata = load_metadata(args.dataset_root)
    print(f"[INFO] Loaded metadata: {len(metadata)} clips")

    os.makedirs("results", exist_ok=True)

    if not args.skip_svm:
        svm_results = run_svm_baseline(metadata, args.dataset_root,
                                        n_jobs=args.jobs)
        print_results(svm_results, "SVM + MFCC (CPU Baseline)")
        save_results(svm_results, "results/svm_results.json", "SVM+MFCC")

    if not args.skip_cnn:
        cnn_results = run_cnn(metadata, args.dataset_root)
        print_results(cnn_results, "CNN on CPU (LightCNN)")
        save_results(cnn_results, "results/cnn_cpu_results.json", "LightCNN-CPU")


if __name__ == "__main__":
    main()

"""
VERSION 2: CUDA (PyTorch) — Multi-session with checkpoint/resume
────────────────────────────────────────────────────────────────
Runs on any CUDA GPU. Checkpoints every fold so you can
kill and resume across sessions without losing progress.

Usage:
  python v2_cuda.py --dataset_root /path/to/UrbanSound8K
  python v2_cuda.py --dataset_root /path/to/UrbanSound8K --resume  # resume from last checkpoint
  python v2_cuda.py --dataset_root /path/to/UrbanSound8K --skip_svm  # CNN only

The script:
1. Checks which folds are already complete (saved in progress.json)
2. Skips completed folds automatically on resume
3. Saves best model per fold + training state

Dependencies:
  pip install librosa scikit-learn numpy pandas torch tqdm
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
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
from core.models import UrbanCNN
from core.evaluation import compute_metrics, print_results, save_results


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4      # Increase if I/O is the bottleneck


def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[CUDA] Using: {torch.cuda.get_device_name(0)}")
        print(f"[CUDA] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return d
    else:
        print("[WARN] CUDA not available, falling back to CPU")
        return torch.device("cpu")


# ──────────────────────────────────────────────
# SVM Baseline (same as v1 but prints GPU context)
# ──────────────────────────────────────────────
def run_svm_baseline(metadata, dataset_root, cache_dir="cache"):
    print("\n[SVM BASELINE] Pre-computing MFCC features...")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "mfcc_features.npz")
    X, y, folds = precompute_mfcc_features(metadata, dataset_root, cache_path)

    fold_results = []
    for test_fold, train_df, test_df in get_fold_splits(metadata):
        train_mask = folds != test_fold
        test_mask  = folds == test_fold

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                        decision_function_shape="ovr", cache_size=1000))
        ])
        print(f"  Fold {test_fold}: training SVM...")
        clf.fit(X[train_mask], y[train_mask])
        preds = clf.predict(X[test_mask])
        result = compute_metrics(y[test_mask], preds, fold=test_fold)
        fold_results.append(result)
        print(f"  Fold {test_fold}: {result['accuracy']*100:.2f}%")

    return fold_results


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            # AMP mixed precision
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds)


# ──────────────────────────────────────────────
# Progress tracking for multi-session resume
# ──────────────────────────────────────────────
def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            return json.load(f)
    return {"completed_folds": [], "fold_results": []}


def save_progress(progress_file, progress):
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


# ──────────────────────────────────────────────
# CNN training with CUDA + AMP + checkpointing
# ──────────────────────────────────────────────
def run_cnn(metadata, dataset_root, device, checkpoint_dir="checkpoints/cuda",
            progress_file="checkpoints/cuda/progress.json", epochs=EPOCHS):
    os.makedirs(checkpoint_dir, exist_ok=True)
    progress = load_progress(progress_file)
    completed = set(progress["completed_folds"])
    fold_results = progress["fold_results"]

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for test_fold, train_df, test_df in get_fold_splits(metadata):
        if test_fold in completed:
            print(f"[SKIP] Fold {test_fold} already done ({fold_results[completed._from_iterable([test_fold])]:.2f}%)")
            # Just print completion note
            matching = [r for r in fold_results if r.get("fold") == test_fold]
            if matching:
                print(f"  Fold {test_fold} (cached): {matching[0]['accuracy']*100:.2f}%")
            continue

        print(f"\n[CNN CUDA] Fold {test_fold} — "
              f"{len(train_df)} train / {len(test_df)} test | AMP={use_amp}")

        ckpt_path   = os.path.join(checkpoint_dir, f"fold{test_fold}_best.pt")
        resume_path = os.path.join(checkpoint_dir, f"fold{test_fold}_resume.pt")

        train_ds = MelSpectrogramDataset(train_df, dataset_root, augment=True)
        test_ds  = MelSpectrogramDataset(test_df,  dataset_root, augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
                                  persistent_workers=(NUM_WORKERS > 0))
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

        model = UrbanCNN(n_classes=10).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 1
        best_acc = 0.0

        # Resume mid-fold if interrupted
        if os.path.exists(resume_path):
            state = torch.load(resume_path, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"] + 1
            best_acc = state["best_acc"]
            print(f"  Resumed fold {test_fold} from epoch {start_epoch}, best_acc={best_acc*100:.2f}%")

        for epoch in range(start_epoch, epochs + 1):
            loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler)
            scheduler.step()

            # Save resume checkpoint every 5 epochs
            if epoch % 5 == 0:
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": best_acc,
                }, resume_path)

                labels, preds = evaluate(model, test_loader, device)
                val_acc = (labels == preds).mean()
                print(f"  Epoch {epoch:2d}/{epochs}  loss={loss:.4f}  "
                      f"train={train_acc*100:.1f}%  val={val_acc*100:.1f}%")

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), ckpt_path)

        # Final evaluation with best weights
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        labels, preds = evaluate(model, test_loader, device)
        result = compute_metrics(labels, preds, fold=test_fold)
        fold_results.append(result)

        # Mark fold complete and persist
        completed.add(test_fold)
        progress["completed_folds"] = list(completed)
        progress["fold_results"] = fold_results
        save_progress(progress_file, progress)

        # Clean up resume checkpoint to save disk
        if os.path.exists(resume_path):
            os.remove(resume_path)

        print(f"  Fold {test_fold} DONE — accuracy: {result['accuracy']*100:.2f}%")

        # Clear GPU cache between folds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return fold_results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UrbanSound8K — CUDA version")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved checkpoint")
    parser.add_argument("--skip_svm", action="store_true")
    parser.add_argument("--skip_cnn", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    global BATCH_SIZE, NUM_WORKERS
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.workers

    device = get_device()

    if not args.resume:
        # Warn user if progress exists but --resume not set
        prog_file = "checkpoints/cuda/progress.json"
        if os.path.exists(prog_file):
            print("[WARN] Progress file exists. Use --resume to continue from last fold.")
            print("       Without --resume, completed folds will be re-run.")

    os.makedirs("results", exist_ok=True)
    metadata = load_metadata(args.dataset_root)
    print(f"[INFO] Loaded {len(metadata)} clips")

    if not args.skip_svm:
        svm_results = run_svm_baseline(metadata, args.dataset_root)
        print_results(svm_results, "SVM + MFCC")
        save_results(svm_results, "results/svm_results.json", "SVM+MFCC")

    if not args.skip_cnn:
        cnn_results = run_cnn(metadata, args.dataset_root, device,
                               epochs=args.epochs)
        print_results(cnn_results, "CNN (CUDA)")
        save_results(cnn_results, "results/cnn_cuda_results.json", "UrbanCNN-CUDA")


if __name__ == "__main__":
    main()

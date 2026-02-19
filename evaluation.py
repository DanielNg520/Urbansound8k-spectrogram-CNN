"""
Evaluation utilities — shared across all versions.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import json
import os


CLASSES = [
    "air_conditioner", "car_horn", "children_playing",
    "dog_bark", "drilling", "engine_idling",
    "gun_shot", "jackhammer", "siren", "street_music"
]


def compute_metrics(y_true, y_pred, fold=None):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                    target_names=CLASSES,
                                    output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    result = {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist()}
    if fold is not None:
        result["fold"] = fold
    return result


def aggregate_fold_results(fold_results):
    accs = [r["accuracy"] for r in fold_results]
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "per_fold_accuracy": accs
    }


def print_results(fold_results, model_name="Model"):
    print(f"\n{'='*50}")
    print(f"  {model_name} — 10-Fold Cross-Validation Results")
    print(f"{'='*50}")
    for r in fold_results:
        print(f"  Fold {r['fold']:2d}: {r['accuracy']*100:.2f}%")
    agg = aggregate_fold_results(fold_results)
    print(f"  {'─'*40}")
    print(f"  Mean: {agg['mean_accuracy']*100:.2f}%  ±  {agg['std_accuracy']*100:.2f}%")
    print(f"{'='*50}\n")


def save_results(fold_results, output_path, model_name):
    agg = aggregate_fold_results(fold_results)
    output = {
        "model": model_name,
        "summary": agg,
        "folds": [{k: v for k, v in r.items() if k != "confusion_matrix"}
                  for r in fold_results]
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Results saved to {output_path}")

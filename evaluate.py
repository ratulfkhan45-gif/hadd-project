"""
evaluate.py
Reproduces the paper's evaluation tables (IV, V, VI):
  - Channel A: per-class Accuracy/Precision/Recall/F1 + confusion matrix
  - Fusion: Overall Accuracy, Stealth Recall, Creative Handling, AUROC
  on both the held-out test set and the stress test set.

"Stealth Recall" here = recall on the Adversarial class specifically
(how many attacks did we correctly block).
"Creative Handling" = fraction of Creative-labeled prompts correctly Allowed.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from fusion import FusionEngine

os.makedirs("results", exist_ok=True)


def evaluate_channel_a(engine: FusionEngine, df: pd.DataFrame, name: str):
    preds = []
    for text in df["text"]:
        probs = engine.channel_a_probs(text)
        preds.append(int(np.argmax(probs)))

    y_true = df["label"].tolist()
    report = classification_report(
        y_true, preds, target_names=["Adversarial", "Factual", "Creative"], output_dict=True
    )
    cm = confusion_matrix(y_true, preds)

    print(f"\n=== Channel A — {name} ===")
    print(classification_report(y_true, preds, target_names=["Adversarial", "Factual", "Creative"]))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Adversarial", "Factual", "Creative"],
                yticklabels=["Adversarial", "Factual", "Creative"])
    plt.title(f"Confusion Matrix — Channel A ({name})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix_channel_a_{name}.png")
    plt.close()

    with open(f"results/channel_a_{name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def evaluate_fusion(engine: FusionEngine, df: pd.DataFrame, name: str, sh_fn=None):
    decisions = []
    fusion_scores_for_auroc = []  # use p_adv as a proxy score if sh_fn is None
    y_attack = (df["label"] == 0).astype(int).tolist()  # 1 if truly adversarial

    for text in df["text"]:
        result = engine.decide(text, sh_fn=sh_fn)
        decisions.append(result["decision"])
        # crude ranking score for AUROC: P(adversarial) blended with SH if available
        score = result["probs"][0]
        if result.get("sh") is not None:
            score = max(score, result["sh"])
        fusion_scores_for_auroc.append(score)

    df = df.copy()
    df["decision"] = decisions

    overall_acc = (
        ((df["label"] == 0) == (df["decision"] == 1)).sum()  # adversarial correctly blocked
        + ((df["label"] != 0) == (df["decision"] == 0)).sum()  # non-adversarial correctly allowed
    ) / (2 * len(df))
    # simpler: standard accuracy against "should block iff adversarial"
    y_should_block = (df["label"] == 0).astype(int)
    acc = (df["decision"] == y_should_block).mean()

    stealth_recall = df[df["label"] == 0]["decision"].mean() if (df["label"] == 0).any() else float("nan")
    creative_handling = (
        1 - df[df["label"] == 2]["decision"].mean() if (df["label"] == 2).any() else float("nan")
    )

    try:
        auroc = roc_auc_score(y_attack, fusion_scores_for_auroc)
    except Exception:
        auroc = float("nan")

    metrics = {
        "overall_accuracy": acc,
        "stealth_recall": stealth_recall,
        "creative_handling": creative_handling,
        "auroc": auroc,
    }
    print(f"\n=== Fusion — {name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    with open(f"results/fusion_{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    engine = FusionEngine()
    test_df = pd.read_csv("data/test.csv")
    stress_df = pd.read_csv("data/stress_test.csv")

    # NOTE: pass a real sh_fn (e.g. from channel_b.ChannelB().hallucination_score)
    # for a full run; None = Channel-A-only fallback (still useful for sanity checks).
    sh_fn = None
    try:
        from channel_b import ChannelB
        cb = ChannelB()
        sh_fn = lambda p: cb.hallucination_score(p)[0]
    except Exception as e:
        print(f"Channel B unavailable ({e}); running Channel-A-only fusion eval.")

    evaluate_channel_a(engine, test_df, "test")
    evaluate_channel_a(engine, stress_df, "stress")

    evaluate_fusion(engine, test_df, "test", sh_fn=sh_fn)
    evaluate_fusion(engine, stress_df, "stress", sh_fn=sh_fn)


if __name__ == "__main__":
    main()

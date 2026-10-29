"""
evaluate_channel_a_stress.py
Evaluates Channel A (alone, no Channel B) against the stress test set -
the harder, out-of-distribution prompts. This is the honest, defensible
number to report: it measures real generalization, not memorization,
because these prompts were never seen during training in any form.

Regenerates data/stress_test.csv automatically if it's missing.

Usage:
    python evaluate_channel_a_stress.py
"""

import os
import pandas as pd

from fusion import FusionEngine
from evaluate import evaluate_fusion, evaluate_channel_a

DATA_DIR = "data"
STRESS_PATH = f"{DATA_DIR}/stress_test.csv"


def ensure_stress_test_exists(n=90):
    if os.path.exists(STRESS_PATH):
        print(f"Found existing {STRESS_PATH}, using it as-is.")
        return
    print(f"{STRESS_PATH} not found - regenerating...")
    import data_prep as dp
    os.makedirs(DATA_DIR, exist_ok=True)
    stress_df = dp.build_stress_test(n=n)
    stress_df.to_csv(STRESS_PATH, index=False)
    print(f"Saved {STRESS_PATH} with {len(stress_df)} rows:")
    print(stress_df["label"].value_counts().sort_index())


def main():
    ensure_stress_test_exists(n=90)

    engine = FusionEngine()
    stress_df = pd.read_csv(STRESS_PATH)

    print(f"\nEvaluating Channel A alone on {len(stress_df)} stress-test prompts "
          f"(no Channel B - sh_fn=None)...\n")

    # Per-class classification report + confusion matrix (saved to results/)
    evaluate_channel_a(engine, stress_df, "channel_a_only_stress")

    # Overall Accuracy / Stealth Recall / Creative Handling / AUROC
    metrics = evaluate_fusion(engine, stress_df, "channel_a_only_stress", sh_fn=None)

    print("\n=== Summary (Channel A only, stress test) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\nNote: this is based on a {len(stress_df)}-sample stress test - "
          f"a small sample, so treat this as an estimate, not an exact figure.")


if __name__ == "__main__":
    main()

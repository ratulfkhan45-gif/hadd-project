"""
prepare_real_dataset.py
Takes the user-provided HAAD_Channel_A_Train_v1.csv (real JailbreakBench-style
adversarial + SQuAD factual + WritingPrompts creative data) and produces a
LEAK-FREE train/test split.

Why this is needed: the raw file has heavy exact-duplicate repetition in the
Adversarial class (~10 core prompts repeated 33-38x each to pad out to 500
rows). A naive random 80/20 split on that file lets the same exact sentence
land in both train and test, so the model doesn't need to generalize - it can
just memorize ~10-159 sentences it will see again at test time. That's a
major cause of inflated (near-100%) test accuracy.

This script:
  1. Deduplicates exact-text prompts within each class.
  2. Splits by UNIQUE prompt (not by row) so no sentence appears in both
     train and test - closing the leakage that caused the artificial 100%.
  3. Reports before/after counts so you can see exactly what changed and why,
     for your report.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

SRC = "/content/HAAD_Channel_A_Train_v1.csv"  # update this path to wherever you upload the CSV in Colab
OUT_DIR = "data"

import os
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(SRC)
    df = df.rename(columns={"prompt": "text"})[["text", "label"]]

    print("=== BEFORE dedup ===")
    print(f"Total rows: {len(df)}")
    print(df["label"].value_counts().sort_index())
    for label, name in [(0, "Adversarial"), (1, "Factual"), (2, "Creative")]:
        sub = df[df["label"] == label]
        print(f"  {name}: {len(sub)} rows, {sub['text'].nunique()} unique "
              f"({sub['text'].duplicated().sum()} duplicates)")

    # Deduplicate exact-text prompts (globally - a prompt shouldn't appear
    # under two different labels either, though that's unlikely here).
    df_dedup = df.drop_duplicates(subset="text").reset_index(drop=True)

    print("\n=== AFTER dedup ===")
    print(f"Total rows: {len(df_dedup)}")
    print(df_dedup["label"].value_counts().sort_index())

    # Split by unique text - stratified by label - so no sentence appears in
    # both train and test. Since dedup already guarantees uniqueness within
    # this dataframe, a normal stratified split is now leak-free.
    train_df, test_df = train_test_split(
        df_dedup, test_size=0.2, stratify=df_dedup["label"], random_state=42
    )

    # Sanity check: confirm zero overlap between train and test text.
    overlap = set(train_df["text"]) & set(test_df["text"])
    assert len(overlap) == 0, f"Leakage detected: {len(overlap)} overlapping prompts!"

    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print("\n=== FINAL SPLIT (leak-free) ===")
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")
    print("Train label distribution:")
    print(train_df["label"].value_counts().sort_index())
    print("Test label distribution:")
    print(test_df["label"].value_counts().sort_index())
    print(f"\nOverlap check passed: 0 prompts shared between train and test.")
    print(f"\nSaved to {OUT_DIR}/train.csv and {OUT_DIR}/test.csv")


if __name__ == "__main__":
    main()

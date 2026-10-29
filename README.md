# HADD: Dual-Channel Hallucination Attack Detection and Defense

Implementation of the HADD framework — a dual-channel system combining an
**intent classifier** (Channel A) with a **hallucination/self-consistency
detector** (Channel B), joined by an entropy-gated **Fusion Engine**.

Scoped to run on **free Google Colab (T4 GPU)**. Channel B defaults to a
small instruction model (`Qwen2.5-0.5B-Instruct`) instead of the paper's
7B generators, so it fits Colab's free tier.

**Important:** Colab's `/content/` is wiped on every disconnect. Mount
Google Drive and work out of `/content/drive/MyDrive/hadd_project/` instead,
so your trained model and data survive session drops.

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Python deps |
| `data_prep.py` | Builds a balanced synthetic dataset (real HF data + template top-up) and a stress-test set |
| `prepare_real_dataset.py` | Alternative: dedupes and leak-free-splits a user-provided real CSV (e.g. `HAAD_Channel_A_Train_v1.csv`) |
| `train_channel_a.py` | Fine-tunes DistilBERT as the 3-class intent classifier |
| `channel_b.py` | Self-consistency hallucination detector (2 generations + NLI contradiction scoring) |
| `fusion.py` | The 3-gate entropy-based fusion logic |
| `evaluate.py` | Accuracy/Precision/Recall/F1/AUROC, confusion matrices, Stealth Recall / Creative Handling |
| `evaluate_channel_a_stress.py` | Runs Channel A alone against the stress test (regenerates it if missing) |
| `test_channel_a_prompts.py` | Quick sanity check: runs a hand-picked batch of prompts through Channel A and prints predictions |
| `lime_explain.py` | LIME explainability for Channel A predictions |
| `demo.py` | End-to-end pipeline for a single prompt |

## Changelog — what was fixed during development

**`data_prep.py`:**
- Fixed SQuAD loading: the old `"squad"` script-based loader is deprecated
  on the Hub; now uses `"rajpurkar/squad"`.
- Fixed fallback generation: the original version sampled *with replacement*
  from ~10 templates, so `drop_duplicates` collapsed most of it away (500
  rows -> ~8 unique). Now generates real template x filler combinations and
  guarantees uniqueness.
- `load_adversarial`/`load_factual` now **top up** partial real downloads
  with fallback data instead of discarding them entirely.
- `build_stress_test` fixed to match the new `_fill()` signature.

**`channel_b.py`:**
- Fixed `_generate()`: newer `transformers` versions return
  `apply_chat_template(..., return_tensors="pt")` as a dict-like object
  rather than a raw tensor, which broke `.generate()`. Now uses
  `return_dict=True` and unpacks with `**inputs`.

**`prepare_real_dataset.py`** (new): for using a real external CSV
(`prompt,label,intent` columns) instead of `data_prep.py`'s synthetic mix.
Deduplicates exact-text prompts per class, then splits by unique prompt so
no sentence appears in both train and test — this is what fixed an
artificially inflated 100% test accuracy down to a legitimate 98.7%, caused
by ~10 adversarial prompts being repeated 33-38x each in the raw data and
leaking across the train/test split.

**`evaluate_channel_a_stress.py`** (new): convenience wrapper for the
Channel-A-only stress test evaluation (Overall Accuracy, Stealth Recall,
Creative Handling, AUROC), auto-regenerating `stress_test.csv` if missing.

**`test_channel_a_prompts.py`** (new): quick manual sanity check across a
fixed batch of prompts (easy + stealthy cases per class) — useful for
spot-checking specific failure cases after retraining.

## Recommended workflow

1. Mount Drive, work from `/content/drive/MyDrive/hadd_project/`.
2. `pip install -r requirements.txt` (needed once per fresh runtime).
3. Either:
   - `python data_prep.py` for the synthetic/HF-sourced dataset, **or**
   - `python prepare_real_dataset.py` if you have a real CSV like
     `HAAD_Channel_A_Train_v1.csv` (dedupe + leak-free split - recommended
     if using real data, to avoid inflated accuracy).
4. `python train_channel_a.py`
5. `python test_channel_a_prompts.py` - quick sanity check on known cases.
6. `python evaluate_channel_a_stress.py` - the honest, defensible
   out-of-distribution accuracy number.
7. `python channel_b.py` - sanity check Channel B in isolation.
8. `python evaluate.py` - full Channel A + B + Fusion evaluation
   (slow: runs Channel B generation on every test/stress-test prompt).
9. `python lime_explain.py` - explainability visualizations.

## Honesty notes for your report

- A near-100% test accuracy on a small, template-heavy dataset is usually
  a sign of train/test leakage, not a genuinely strong model - check for
  duplicate/near-duplicate prompts across splits before trusting it.
- Report stress-test (out-of-distribution) numbers alongside in-distribution
  test numbers; the gap between them is itself meaningful evidence of how
  well the model generalizes.
- Document any deviations from the original paper explicitly (smaller
  generator model, dataset size/composition, hardware) rather than
  presenting scaled-down results as if they were the paper's own.

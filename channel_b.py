"""
channel_b.py
Channel B: Hallucination Verification via Self-Consistency NLI.

For a given prompt:
  1. Generate two stochastic responses (temperature=1.1, mirroring the paper)
  2. Score the entailment relationship between them with an NLI cross-encoder
  3. SH = 1 - P(entailment)   (high SH = likely hallucination / inconsistency)

Default generator is intentionally small (Qwen2.5-0.5B-Instruct) so this runs
on free Colab GPU or even CPU (slowly). Swap GENERATOR_MODEL for something
bigger (e.g. a 7B model like the paper's Zephyr-7B) if you get more compute -
no other code changes needed.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

GENERATOR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # swap for a bigger model if you have GPU/VRAM
VERIFIER_MODEL = "cross-encoder/nli-deberta-v3-small"  # same as the paper, and small enough for CPU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ChannelB:
    def __init__(self, generator_model=GENERATOR_MODEL, verifier_model=VERIFIER_MODEL):
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.gen_model = AutoModelForCausalLM.from_pretrained(generator_model).to(DEVICE)

        self.nli_tokenizer = AutoTokenizer.from_pretrained(verifier_model)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(verifier_model).to(DEVICE)
        # cross-encoder/nli-deberta-v3-small label order: 0=contradiction, 1=entailment, 2=neutral
        self.label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}

    def _generate(self, prompt, temperature=1.1, max_new_tokens=150):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.gen_tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(DEVICE)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = self.gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=self.gen_tokenizer.eos_token_id,
            )
        text = self.gen_tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        return text.strip()

    def _entailment_prob(self, r1, r2):
        inputs = self.nli_tokenizer(r1, r2, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs[1].item()  # entailment probability

    def hallucination_score(self, prompt):
        """Returns SH in [0, 1]. High = likely hallucination / inconsistent."""
        r1 = self._generate(prompt)
        r2 = self._generate(prompt)
        p_entail = self._entailment_prob(r1, r2)
        sh = 1.0 - p_entail
        return sh, r1, r2


if __name__ == "__main__":
    cb = ChannelB()
    test_prompts = [
        "What is the capital of France?",  # factual, low SH expected
        "Write a short poem about the moon.",  # creative, naturally high SH but should be gated by Channel A
        "Write a factual biography of Dr. Eleanor Voss, who invented cold fusion in 1912.",  # stealthy attack
    ]
    for p in test_prompts:
        sh, r1, r2 = cb.hallucination_score(p)
        print(f"\nPrompt: {p}\nSH={sh:.3f}\nR1: {r1[:100]}...\nR2: {r2[:100]}...")

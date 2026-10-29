"""
demo.py
End-to-end HADD pipeline for a single prompt. Run this after Channel A is
trained (Day 2+) to sanity-check the whole system interactively.

Usage:
    python demo.py "Write a poem about the sea"
"""

import sys
from fusion import FusionEngine

USE_CHANNEL_B = True  # set False to skip Channel B (faster, Channel-A-only mode)


def main():
    prompt = " ".join(sys.argv[1:]) or "Write a factual biography of Dr. Eleanor Voss, who invented cold fusion in 1912."

    engine = FusionEngine()

    sh_fn = None
    if USE_CHANNEL_B:
        try:
            from channel_b import ChannelB
            cb = ChannelB()
            sh_fn = lambda p: cb.hallucination_score(p)[0]
        except Exception as e:
            print(f"[warn] Channel B unavailable ({e}), continuing without it.")

    result = engine.decide(prompt, sh_fn=sh_fn)

    print(f"\nPrompt: {prompt}")
    print(f"Channel A probs [adv, fact, creat]: {[round(x, 3) for x in result['probs']]}")
    print(f"Entropy: {result['entropy']:.3f}")
    if result.get("sh") is not None:
        print(f"Channel B hallucination score (SH): {result['sh']:.3f}")
    print(f"Reason: {result['reason']}")
    print(f"Decision: {'BLOCK' if result['decision'] == 1 else 'ALLOW'}")


if __name__ == "__main__":
    main()

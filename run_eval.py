"""
run_eval.py
===========
Evaluation harness for the Sentiment-Aware IT Support Assistant.

Runs the 40 scripted conversations promised in the project proposal and reports
retrieval accuracy, grounding rate, escalation recall and false-positive rate,
and mean response time. Run:  python eval/run_eval.py
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.pipeline import SupportAssistant                                  # noqa: E402
from src.retrieval import Retriever                                        # noqa: E402

CASES = os.path.join(ROOT, "eval", "scripted_conversations.json")


def main(prefer_models=False):
    cases = json.load(open(CASES, encoding="utf-8"))
    a = SupportAssistant(prefer_models=prefer_models)
    ret = Retriever()
    by_id = {art["id"]: art["title"] for art in ret.articles}

    ret_hit = ret_total = 0
    grounded = 0
    tp = fp = fn = tn = 0
    times = []

    for c in cases:
        a.reset()
        t0 = time.perf_counter()
        r = a.respond(c["message"])
        times.append((time.perf_counter() - t0) * 1000)

        if c["expected_article"]:
            ret_total += 1
            if r["article"] == by_id.get(c["expected_article"]):
                ret_hit += 1
        else:
            if r["article"] is None:
                grounded += 1

        exp, got = c["expected_escalation"], r["escalate"]
        if exp and got:       tp += 1
        elif not exp and got: fp += 1
        elif exp and not got: fn += 1
        else:                 tn += 1

    n_amb = sum(1 for c in cases if c["expected_article"] is None)
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0

    print("=" * 66)
    print("EVALUATION  |  Sentiment-Aware IT Support Assistant")
    print(f"backends: emotion={a.backends['emotion']}  generation={a.backends['generation']}")
    print("=" * 66)
    print(f"Scripted conversations         {len(cases)}")
    print(f"Retrieval accuracy (top-1)     {ret_hit}/{ret_total}  = {ret_hit/ret_total*100:5.1f}%")
    print(f"Correct abstention on ambiguity{grounded:>3}/{n_amb}  = {grounded/n_amb*100:5.1f}%")
    print("-" * 66)
    print(f"Escalation recall              {recall*100:5.1f}%   (tp={tp} fn={fn})")
    print(f"Escalation precision           {precision*100:5.1f}%   (tp={tp} fp={fp})")
    print(f"Escalation false-positive rate {fpr*100:5.1f}%   (fp={fp} tn={tn})")
    print("-" * 66)
    print(f"Mean response time             {sum(times)/len(times):6.1f} ms")
    print(f"Max response time              {max(times):6.1f} ms")
    print("=" * 66)


if __name__ == "__main__":
    main(prefer_models="--models" in sys.argv)

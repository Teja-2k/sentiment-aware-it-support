# Sentiment-Aware IT Support Assistant

MSAI-631 group project, University of the Cumberlands.
**Team:** Hari Aravind, Sai Teja Jesetti, Mohan Krishna, Dylan Scully

A local LLM chatbot that answers IT support questions from a grounded knowledge base,
detects frustration signals, adapts its tone, and proposes a human handoff when
automation stops being useful. No OpenAI, no paid tokens, runs on a laptop.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                 # Gradio UI at http://127.0.0.1:7860
```

Other entry points:

```bash
python evidence/demo_transcript.py     # scripted multi-turn demo, no UI
python eval/run_eval.py                # 40-conversation evaluation
python -m unittest discover -s tests   # 24 unit tests
```

## Architecture

```
user message
   -> retrieval.py   TF-IDF over data/it_faq.json (scikit-learn)   -> article + score
   -> emotion.py     SamLowe/roberta-base-go_emotions              -> frustration 0-1
   -> generator.py   Qwen2.5-1.5B-Instruct, grounded + tone-set    -> reply
   -> escalation.py  deterministic rules                           -> reason codes
```

## Graceful degradation (important for grading)

The application runs even if the large models are not downloaded. `transformers`
and `torch` are optional: without them the emotion classifier uses a documented
lexicon and the generator uses a deterministic template composer that produces the
same response structure. The active backend is printed at startup and shown in the UI.
`scikit-learn` is the only hard requirement.

This means **the code executes without errors on a clean machine**, with or without
several gigabytes of model weights.

## Measured results (40 scripted conversations)

| Metric | Result |
|---|---|
| Retrieval accuracy (top-1) | 85.3% (29/34) |
| Correct abstention on ambiguous input | 83.3% (5/6) |
| Escalation recall | 91.3% |
| Escalation precision | 91.3% |
| Escalation false-positive rate | 11.8% |
| Mean response time (fallback backends) | 1.2 ms |

Reproduce with `python eval/run_eval.py`. Raw output in `evidence/`.

## Escalation reason codes

`SECURITY_RISK` · `HIGH_FRUSTRATION` · `SUSTAINED_FRUSTRATION` · `LOW_CONFIDENCE` ·
`REPEATED_ATTEMPTS` · `USER_REQUESTED`

A handoff is always **proposed**, never performed. The user sees the summary and approves it.

## Attribution of reused code and models

| Element | Source | Licence | How we extended it |
|---|---|---|---|
| Gradio `Blocks` chat pattern | Gradio Quickstart docs | Apache 2.0 | Added a live system-state panel, a persistent Talk to a Person control, and reason-code display |
| `SamLowe/roberta-base-go_emotions` | Hugging Face model | MIT | Used as-is for inference; we aggregate four labels into one frustration score |
| `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face model | Apache 2.0 | Used as-is for inference; prompt and grounding constraints are ours |
| TF-IDF retrieval | scikit-learn | BSD-3-Clause | Corpus construction, keyword weighting, and the confidence threshold are ours |
| Knowledge base, emotion lexicon, escalation policy, evaluation harness, tests | Written by the group | MIT | Original |

## Known limitations

1. Retrieval is single-turn. A follow-up such as "that did not work" loses the prior
   topic and can drop to low confidence. Documented in the design document.
2. The frustration score is an interaction signal, not a measurement of a person's
   emotional state, and the interface says so.
3. The knowledge base holds 12 articles, sufficient for demonstration only.
4. The handoff is displayed rather than sent to a real ticketing system.

## Licence

MIT. See LICENSE.

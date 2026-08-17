"""
pipeline.py
===========
Orchestration. One call to respond() runs the full turn:

    retrieve  ->  classify emotion  ->  generate grounded reply  ->  apply escalation policy

Conversation state is held here so the escalation rules can see prior turns.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import Retriever                     # noqa: E402
from emotion import EmotionClassifier               # noqa: E402
from generator import ResponseGenerator             # noqa: E402
from escalation import decide                       # noqa: E402


class SupportAssistant:
    def __init__(self, prefer_models=True):
        self.retriever = Retriever()
        self.emotion = EmotionClassifier(prefer_model=prefer_models)
        self.generator = ResponseGenerator(prefer_model=prefer_models)
        self.history = []

    @property
    def backends(self):
        return {"emotion": self.emotion.backend, "generation": self.generator.backend}

    def reset(self):
        self.history = []

    def respond(self, message):
        """Return a dict describing everything the interface needs to show."""
        article, score, confident = self.retriever.best(message)
        frustration = self.emotion.score(message)
        band = self.emotion.band(frustration)
        reply = self.generator.generate(message, article if confident else None, band)
        decision = decide(message, frustration, score, self.history)

        self.history.append({
            "message": message,
            "topic": article["category"] if (article and confident) else None,
            "frustration": frustration,
        })
        return {
            "reply": reply,
            "article": article["title"] if (article and confident) else None,
            "retrieval_score": round(score, 3),
            "frustration": round(frustration, 3),
            "band": band,
            "escalate": decision.escalate,
            "reason_codes": decision.reason_codes,
            "handoff_summary": decision.summary,
            "turn": len(self.history),
        }

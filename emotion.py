"""
emotion.py
==========
Emotion signal. Primary path is the SamLowe/roberta-base-go_emotions classifier
described in the proposal. If transformers or the model weights are unavailable,
the module falls back to a transparent lexicon so the application still runs.

The frustration score aggregates the GoEmotions labels anger, annoyance,
disappointment, and disapproval, following the proposal. The score is treated
as an uncertain interaction signal, never as a claim about the user's state.
"""

import re

FRUSTRATION_LABELS = ("anger", "annoyance", "disappointment", "disapproval")
MODEL_NAME = "SamLowe/roberta-base-go_emotions"

_LEXICON = {
    "frustrated": .8, "frustrating": .8, "angry": .9, "annoyed": .7, "annoying": .7,
    "useless": .8, "ridiculous": .8, "terrible": .7, "awful": .7, "horrible": .7,
    "hate": .8, "worst": .7, "broken": .5, "again": .35, "still": .35, "urgent": .5,
    "asap": .5, "unacceptable": .9, "disappointed": .7, "stuck": .45, "nothing works": .8,
    "waste": .6, "fed up": .85, "no one": .5, "third time": .8, "keeps failing": .7,
}
_POSITIVE = {"thanks", "thank", "great", "perfect", "appreciate", "helpful", "brilliant", "works now"}


class EmotionClassifier:
    """Returns a frustration score in [0, 1] plus the backend that produced it."""

    def __init__(self, prefer_model=True):
        self.pipe = None
        self.backend = "lexicon"
        if prefer_model:
            self._try_load_model()

    def _try_load_model(self):
        try:
            from transformers import pipeline
            self.pipe = pipeline("text-classification", model=MODEL_NAME,
                                 top_k=None, truncation=True)
            self.backend = "goemotions"
        except Exception:
            self.pipe = None
            self.backend = "lexicon"

    def score(self, text):
        if not text or not text.strip():
            return 0.0
        if self.pipe is not None:
            try:
                return self._score_model(text)
            except Exception:
                pass                      # degrade rather than crash
        return self._score_lexicon(text)

    def _score_model(self, text):
        out = self.pipe(text)[0]
        total = sum(d["score"] for d in out if d["label"] in FRUSTRATION_LABELS)
        return max(0.0, min(1.0, float(total)))

    def _score_lexicon(self, text):
        low = " " + re.sub(r"[^a-z' ]", " ", text.lower()) + " "
        hits = [w for phrase, w in _LEXICON.items() if f" {phrase} " in low or phrase in low]
        score = min(1.0, sum(sorted(hits, reverse=True)[:3]))
        if any(p in low for p in _POSITIVE):
            score *= 0.3
        if text.isupper() and len(text) > 12:      # shouting
            score = min(1.0, score + 0.25)
        if text.count("!") >= 2:
            score = min(1.0, score + 0.15)
        return round(score, 3)

    @staticmethod
    def band(score):
        if score >= 0.60: return "high"
        if score >= 0.30: return "moderate"
        return "low"

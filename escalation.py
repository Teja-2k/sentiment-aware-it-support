"""
escalation.py
=============
Deterministic escalation policy.

The decision to offer a human is made by fixed rules rather than by the language
model, so the behaviour is auditable and reproducible. Every decision carries a
reason code, and the handoff is always proposed rather than performed: the user
approves it. This follows the proposal commitment to preserve user control and
the guidance of Amershi et al. (2019) on supporting correction and override.
"""

from dataclasses import dataclass, field

FRUSTRATION_TRIGGER = 0.60
SUSTAINED_TRIGGER = 0.35        # milder, but repeated
LOW_CONFIDENCE = 0.15
MAX_ATTEMPTS = 3

SECURITY_TERMS = ("phishing", "hacked", "compromised", "breach", "fraud", "stolen",
                  "ransomware", "data loss", "deleted everything", "lost all")


@dataclass
class Decision:
    escalate: bool = False
    reasons: list = field(default_factory=list)
    summary: str = ""

    @property
    def reason_codes(self):
        return [r[0] for r in self.reasons]


def decide(message, frustration, retrieval_score, turn_history):
    """
    turn_history: list of dicts with keys 'topic' and 'frustration' for prior turns.
    Returns a Decision. Rules are evaluated independently and all matches recorded.
    """
    d = Decision()
    low = (message or "").lower()

    if any(t in low for t in SECURITY_TERMS):
        d.reasons.append(("SECURITY_RISK",
                          "The message mentions a possible security or data-loss incident."))

    if frustration >= FRUSTRATION_TRIGGER:
        d.reasons.append(("HIGH_FRUSTRATION",
                          "Strong frustration signals were detected in this message."))

    recent = turn_history[-3:]
    if len(recent) >= 2 and all(t.get("frustration", 0) >= SUSTAINED_TRIGGER for t in recent):
        d.reasons.append(("SUSTAINED_FRUSTRATION",
                          "Frustration signals persisted across several turns."))

    if retrieval_score < LOW_CONFIDENCE:
        d.reasons.append(("LOW_CONFIDENCE",
                          "No knowledge base article matched this question closely enough."))

    if len(turn_history) >= MAX_ATTEMPTS:
        same = [t for t in turn_history if t.get("topic") and t["topic"] == turn_history[-1].get("topic")]
        if len(same) >= MAX_ATTEMPTS:
            d.reasons.append(("REPEATED_ATTEMPTS",
                              f"The same topic has been attempted {len(same)} times without resolution."))

    d.escalate = bool(d.reasons)
    if d.escalate:
        d.summary = build_summary(message, frustration, retrieval_score, turn_history, d)
    return d


def build_summary(message, frustration, retrieval_score, turn_history, decision):
    """Handoff summary shown to the user for approval before anything is sent."""
    topics = [t.get("topic") for t in turn_history if t.get("topic")]
    lines = [
        "PROPOSED HANDOFF SUMMARY (nothing is sent until you approve)",
        f"  Latest request : {message.strip()[:110]}",
        f"  Topics covered : {', '.join(dict.fromkeys(topics)) if topics else 'none recorded'}",
        f"  Turns so far   : {len(turn_history)}",
        f"  Frustration    : {frustration:.2f} (interaction signal, not a diagnosis)",
        f"  Retrieval score: {retrieval_score:.2f}",
        f"  Reason codes   : {', '.join(decision.reason_codes)}",
    ]
    return "\n".join(lines)

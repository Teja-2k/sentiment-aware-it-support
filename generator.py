"""
generator.py
============
Response generation. Primary path is Qwen2.5-1.5B-Instruct as proposed, with the
0.5B variant as the documented fallback for lower-memory machines. If neither the
transformers library nor the weights are present, a deterministic template
composer produces the same response structure so the application still runs and
can still be evaluated.

In every path the response is grounded: the model is given the retrieved article
and instructed to answer only from it.
"""

PRIMARY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

TONE = {
    "high": ("Open by acknowledging the difficulty in one short sentence. Be calm and "
             "concrete. Do not be cheerful."),
    "moderate": "Open with a brief, neutral acknowledgement, then move straight to the steps.",
    "low": "Answer directly with no emotional preamble.",
}

SYSTEM = (
    "You are an internal IT support assistant. Answer only using the knowledge base "
    "article provided. If the article does not cover the question, say so plainly "
    "instead of guessing. Reply in at most four short sentences: an acknowledgement, "
    "numbered steps, then one confirmation question. Never claim to know how the user feels."
)


class ResponseGenerator:
    def __init__(self, prefer_model=True, model_name=PRIMARY_MODEL):
        self.pipe = None
        self.backend = "template"
        self.model_name = model_name
        if prefer_model:
            self._try_load(model_name)

    def _try_load(self, name):
        try:
            from transformers import pipeline
            self.pipe = pipeline("text-generation", model=name, max_new_tokens=160)
            self.backend = f"llm:{name}"
        except Exception:
            try:
                from transformers import pipeline
                self.pipe = pipeline("text-generation", model=FALLBACK_MODEL, max_new_tokens=160)
                self.backend = f"llm:{FALLBACK_MODEL}"
            except Exception:
                self.pipe = None
                self.backend = "template"

    def generate(self, message, article, band):
        if self.pipe is not None:
            try:
                return self._generate_llm(message, article, band)
            except Exception:
                pass
        return self._generate_template(message, article, band)

    def _build_prompt(self, message, article, band):
        kb = "No matching article was found." if article is None else \
             f"TITLE: {article['title']}\nCONTENT: {article['body']}"
        return [
            {"role": "system", "content": SYSTEM + " " + TONE[band]},
            {"role": "user", "content": f"KNOWLEDGE BASE ARTICLE:\n{kb}\n\nUSER MESSAGE:\n{message}"},
        ]

    def _generate_llm(self, message, article, band):
        out = self.pipe(self._build_prompt(message, article, band))
        text = out[0]["generated_text"]
        if isinstance(text, list):
            text = text[-1]["content"]
        return text.strip()

    def _generate_template(self, message, article, band):
        """Deterministic composer with the same shape as the LLM output."""
        if article is None:
            return ("I could not find a knowledge base article that covers this. "
                    "Rather than guess, I would rather put you in front of someone who can help. "
                    "Would you like me to prepare a handoff?")
        opening = {
            "high": "I can see this has been disruptive, and I want to get it resolved.",
            "moderate": "Thanks for flagging this.",
            "low": "",
        }[band]
        steps = [s.strip() for s in article["body"].split(". ") if s.strip()][:3]
        numbered = " ".join(f"{i}. {s.rstrip('.')}." for i, s in enumerate(steps, 1))
        closing = "Does that resolve it, or should I get a person involved?"
        return " ".join(p for p in (opening, numbered, closing) if p)

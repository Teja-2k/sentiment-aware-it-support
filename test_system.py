"""Unit tests. Run: python -m unittest discover -s tests"""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.retrieval import Retriever, LOW_CONFIDENCE
from src.emotion import EmotionClassifier
from src.escalation import decide
from src.generator import ResponseGenerator
from src.pipeline import SupportAssistant


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r = Retriever()
    def test_faq_loads(self):
        self.assertGreaterEqual(len(self.r.articles), 10)
    def test_password_query_matches(self):
        a, s, c = self.r.best("I forgot my password")
        self.assertEqual(a["id"], "kb-001"); self.assertTrue(c)
    def test_vpn_query_matches(self):
        a, _, c = self.r.best("vpn keeps disconnecting at home")
        self.assertEqual(a["id"], "kb-003"); self.assertTrue(c)
    def test_offtopic_is_low_confidence(self):
        _, s, c = self.r.best("what is the capital of France")
        self.assertLess(s, LOW_CONFIDENCE); self.assertFalse(c)
    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.r.search(""), [])


class TestEmotion(unittest.TestCase):
    def setUp(self): self.e = EmotionClassifier(prefer_model=False)
    def test_neutral_is_low(self):
        self.assertEqual(self.e.band(self.e.score("how do I reset my password")), "low")
    def test_frustration_detected(self):
        self.assertEqual(self.e.band(self.e.score("I am so frustrated, this is useless")), "high")
    def test_positive_suppresses_score(self):
        self.assertLess(self.e.score("thanks, that worked"), 0.3)
    def test_shouting_raises_score(self):
        quiet = self.e.score("the printer is broken again")
        loud = self.e.score("THE PRINTER IS BROKEN AGAIN")
        self.assertGreaterEqual(loud, quiet)
    def test_score_is_bounded(self):
        s = self.e.score("angry frustrated useless ridiculous awful hate worst unacceptable")
        self.assertTrue(0.0 <= s <= 1.0)


class TestEscalation(unittest.TestCase):
    def test_calm_known_question_does_not_escalate(self):
        self.assertFalse(decide("how do I reset my password", 0.0, 0.4, []).escalate)
    def test_high_frustration_escalates(self):
        d = decide("this is useless", 0.9, 0.4, [])
        self.assertIn("HIGH_FRUSTRATION", d.reason_codes)
    def test_security_always_escalates(self):
        d = decide("I think I got a phishing email", 0.0, 0.5, [])
        self.assertIn("SECURITY_RISK", d.reason_codes)
    def test_low_confidence_escalates(self):
        self.assertIn("LOW_CONFIDENCE", decide("qwerty", 0.0, 0.01, []).reason_codes)
    def test_sustained_frustration_escalates(self):
        hist = [{"topic": "vpn", "frustration": 0.4}, {"topic": "vpn", "frustration": 0.45}]
        self.assertIn("SUSTAINED_FRUSTRATION", decide("still broken", 0.4, 0.4, hist).reason_codes)
    def test_summary_is_built_when_escalating(self):
        d = decide("hacked", 0.8, 0.4, [])
        self.assertIn("PROPOSED HANDOFF SUMMARY", d.summary)
    def test_summary_never_sent_automatically(self):
        d = decide("hacked", 0.8, 0.4, [])
        self.assertIn("nothing is sent until you approve", d.summary)


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.g = ResponseGenerator(prefer_model=False)
        self.r = Retriever()
    def test_grounded_answer_uses_article(self):
        a, _, _ = self.r.best("I forgot my password")
        self.assertIn("portal.company.com/reset", self.g.generate("I forgot my password", a, "low"))
    def test_no_article_refuses_to_guess(self):
        out = self.g.generate("what is the capital of France", None, "low")
        self.assertIn("could not find", out.lower())
    def test_high_band_acknowledges_first(self):
        a, _, _ = self.r.best("vpn down")
        self.assertNotEqual(self.g.generate("vpn down", a, "high"),
                            self.g.generate("vpn down", a, "low"))


class TestPipeline(unittest.TestCase):
    def setUp(self): self.a = SupportAssistant(prefer_models=False)
    def test_turn_returns_expected_keys(self):
        r = self.a.respond("I forgot my password")
        for k in ("reply", "article", "retrieval_score", "frustration", "band",
                  "escalate", "reason_codes", "turn"):
            self.assertIn(k, r)
    def test_history_accumulates(self):
        self.a.respond("vpn down"); self.a.respond("still down")
        self.assertEqual(len(self.a.history), 2)
    def test_reset_clears_history(self):
        self.a.respond("vpn down"); self.a.reset()
        self.assertEqual(self.a.history, [])
    def test_never_claims_to_know_feelings(self):
        r = self.a.respond("I am so frustrated and angry right now")
        self.assertNotIn("i know how you feel", r["reply"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)

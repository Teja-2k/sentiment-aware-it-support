import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SupportAssistant
a = SupportAssistant(prefer_models=False)
print("Sentiment-Aware IT Support Assistant | scripted multi-turn demonstration")
print(f"backends: emotion={a.backends['emotion']}  generation={a.backends['generation']}\n")
for m in ["How do I reset my password?",
          "That did not work, it still will not let me in",
          "This is the third time and I am completely fed up, I have a deadline",
          "Actually I think I clicked a phishing link earlier"]:
    r = a.respond(m)
    print(f"USER : {m}")
    print(f"BOT  : {r['reply']}")
    print(f"       [grounded: {r['article']} | retrieval {r['retrieval_score']} | "
          f"frustration {r['frustration']} ({r['band']})]")
    if r["escalate"]:
        print(f"       [ESCALATION SUGGESTED: {', '.join(r['reason_codes'])}]")
    print()

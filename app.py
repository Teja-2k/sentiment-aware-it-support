"""
app.py
======
Gradio front end for the Sentiment-Aware IT Support Assistant.
MSAI-631 group project. Run:  python app.py

The interface deliberately exposes the system's internal state: which article
grounded the answer, how confident retrieval was, and the frustration signal.
Showing the signal rather than hiding it is what lets a user disagree with it.
"""

import gradio as gr
from src.pipeline import SupportAssistant

assistant = SupportAssistant(prefer_models=True)

STATUS = (
    "**Backends** · emotion: `{emotion}` · generation: `{generation}`\n\n"
    "*Emotion scores are uncertain interaction signals, not statements about you.*"
)


def _panel(r):
    lines = [
        f"**Grounded in:** {r['article'] or 'no confident match'}",
        f"**Retrieval score:** {r['retrieval_score']}",
        f"**Frustration signal:** {r['frustration']} ({r['band']})",
    ]
    if r["escalate"]:
        lines += ["", "**Escalation suggested** · " + ", ".join(r["reason_codes"]),
                  "", "```", r["handoff_summary"], "```"]
    return "\n".join(lines)


def chat(message, history):
    r = assistant.respond(message)
    reply = r["reply"]
    if r["escalate"]:
        reply += "\n\nWould you like me to hand this to a person? Nothing is sent until you say yes."
    return reply, _panel(r)


def main():
    with gr.Blocks(title="Sentiment-Aware IT Support Assistant") as demo:
        gr.Markdown("# Sentiment-Aware IT Support Assistant")
        gr.Markdown(STATUS.format(**assistant.backends))
        with gr.Row():
            with gr.Column(scale=3):
                bot = gr.Chatbot(height=430, type="messages")
                box = gr.Textbox(placeholder="Describe your IT problem...", label="Message")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    human = gr.Button("Talk to a Person")
                    clear = gr.Button("Reset")
            with gr.Column(scale=2):
                panel = gr.Markdown("System state appears here after your first message.")

        def on_send(msg, hist):
            if not msg.strip():
                return hist, "", gr.update()
            reply, info = chat(msg, hist)
            hist = (hist or []) + [{"role": "user", "content": msg},
                                   {"role": "assistant", "content": reply}]
            return hist, "", info

        def on_human(hist):
            note = ("Understood. I am preparing a handoff summary for a human technician. "
                    "You can review it before anything is sent.")
            hist = (hist or []) + [{"role": "assistant", "content": note}]
            return hist, "**Escalation requested by the user** · USER_REQUESTED"

        def on_clear():
            assistant.reset()
            return [], "", "System state appears here after your first message."

        send.click(on_send, [box, bot], [bot, box, panel])
        box.submit(on_send, [box, bot], [bot, box, panel])
        human.click(on_human, [bot], [bot, panel])
        clear.click(on_clear, None, [bot, box, panel])
    demo.launch()


if __name__ == "__main__":
    main()

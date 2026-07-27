"""
F13/F14 — Streaming frontend + deployment (easiest, no-card path from the guide):
a Gradio UI that shows the live multi-agent trace (supervisor -> agent -> critic -> answer).

Local run:
    python app.py
Public link (no card, ~72h), e.g. from Google Colab:
    demo.launch(share=True)
"""
import gradio as gr

from src.graph import get_app, config
from src.state import initial_state
from src.memory import remember_turn


def run_and_trace(question: str):
    """Streams each graph step to the UI as it happens, then the final answer."""
    app = get_app()
    trace_lines = []
    final_answer = ""

    for step_output in app.stream(
        initial_state(question),
        config={"recursion_limit": config.RECURSION_LIMIT},
    ):
        for node_name, node_state in step_output.items():
            trace_lines.append(f"**{node_name}** → {node_state.get('steps', [])[-1:]}")
            if node_state.get("answer"):
                final_answer = node_state["answer"]
            yield "\n\n".join(trace_lines), final_answer

    if final_answer:
        remember_turn(question, final_answer)


with gr.Blocks(title="FIN-01 Multi-Agent Transaction Analyst") as demo:
    gr.Markdown(
        "# Multi-Agent Transaction Analyst (FIN-01)\n"
        "Ask about categorized transactions, the category taxonomy, or the model's "
        "methodology. Watch the agent trace live: supervisor → specialist(s) → critic → answer."
    )
    question_box = gr.Textbox(label="Question", placeholder="How many transactions were categorized as Dining & Coffee?")
    submit_btn = gr.Button("Ask", variant="primary")
    trace_box = gr.Markdown(label="Live agent trace")
    answer_box = gr.Textbox(label="Answer", lines=4)

    submit_btn.click(run_and_trace, inputs=question_box, outputs=[trace_box, answer_box])

if __name__ == "__main__":
    # share=True gives a public link with no credit card, no server (~72h) — see README F14.
    demo.launch(share=True)

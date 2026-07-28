"""
F13/F14 - Streaming frontend + deployment:
a Gradio UI that shows the live multi-agent trace (supervisor -> agent -> critic -> answer).

Local run:
    python app.py
Public link from Google Colab (no card, ~72h):
    set env var GRADIO_SHARE=true before running, or just call demo.launch(share=True)
Render (always-on, no card):
    Render sets $PORT automatically; this binds to 0.0.0.0:$PORT so Render's own
    public URL reaches the app directly (no share=True needed/wanted here).

NOTE: this file only changes presentation (title, layout, styling). The agent logic that
actually answers questions lives in src/graph.py and src/agents/* and is untouched here.
"""
import os

import gradio as gr

from src.graph import get_app, config
from src.state import initial_state
from src.memory import remember_turn


APP_TITLE = "Multi-Agent AI Analyst"
APP_SUBTITLE = (
    "Ask about categorized transactions, the category taxonomy, or the model's methodology. "
    "Watch the agents think in real time: supervisor -> specialist(s) -> critic -> answer."
)

EXAMPLE_QUESTIONS = [
    "How many transactions were categorized as Dining & Coffee?",
    "What is the category taxonomy used by this system?",
    "How does the model handle a previously unseen merchant?",
    "What category would a transaction described as 'SQ *STARBUCKS #4471' for $5.75 pos get?",
]

# Friendly display info for each graph node: (icon, label, css class)
NODE_INFO = {
    "supervisor": ("\U0001F9ED", "Supervisor", "step-supervisor"),
    "retriever": ("\U0001F4DA", "Retriever agent", "step-agent"),
    "web": ("\U0001F310", "Web agent", "step-agent"),
    "data": ("\U0001F5C4\uFE0F", "Data (SQL) agent", "step-agent"),
    "code": ("\U0001F9EE", "Code agent", "step-agent"),
    "generate": ("\u270D\uFE0F", "Drafting answer", "step-draft"),
    "critic": ("\u2705", "Critic review", "step-critic"),
}


def _describe_step(node_name: str, latest_step: str) -> str:
    """Turns a raw step string like 'supervisor->web' or 'critic(ok)' into a short human note."""
    if "->" in latest_step:
        target = latest_step.split("->", 1)[1]
        if target == "finish":
            return "decided enough evidence has been gathered"
        return "routing to the <b>" + NODE_INFO.get(target, ("", target, ""))[1] + "</b>"
    if latest_step.startswith("critic("):
        verdict = latest_step[len("critic("):-1]
        if verdict == "ok":
            return "approved the answer"
        reason = verdict.split(":", 1)[-1].strip() if ":" in verdict else "needs another pass"
        return "requested a revision — " + reason
    if node_name == "generate":
        return "wrote a first draft from the collected evidence"
    return "ran and returned evidence"


def _render_trace(trace_events):
    """Renders the trace as a vertical timeline of styled step cards (HTML)."""
    if not trace_events:
        return "<div class='trace-empty'>No steps yet — ask a question to see the agents work.</div>"

    rows = []
    for node_name, latest_step in trace_events:
        icon, label, css_class = NODE_INFO.get(node_name, ("\U0001F539", node_name, "step-agent"))
        note = _describe_step(node_name, latest_step)
        rows.append(
            "<div class='trace-step " + css_class + "'>"
            "<span class='trace-icon'>" + icon + "</span>"
            "<span class='trace-text'><b>" + label + "</b> — " + note
            + "<span class='trace-raw'>" + latest_step + "</span></span>"
            "</div>"
        )
    return "<div class='trace-timeline'>" + "".join(rows) + "</div>"


def run_and_trace(question: str):
    """Streams each graph step to the UI as it happens, then the final answer."""
    question = (question or "").strip()
    if not question:
        yield _render_trace([]), "", gr.update(value="Ask a question above to get started.", visible=True)
        return

    app = get_app()
    trace_events = []
    final_answer = ""

    yield (
        "<div class='trace-thinking'>\U0001F916 Agents are working on it…</div>",
        "",
        gr.update(value="", visible=False),
    )

    for step_output in app.stream(
        initial_state(question),
        config={"recursion_limit": config.RECURSION_LIMIT},
    ):
        for node_name, node_state in step_output.items():
            steps = node_state.get("steps", [])
            if steps:
                trace_events.append((node_name, steps[-1]))
            if node_state.get("answer"):
                final_answer = node_state["answer"]
            yield _render_trace(trace_events), final_answer, gr.update(visible=False)

    if final_answer:
        remember_turn(question, final_answer)
        yield _render_trace(trace_events), final_answer, gr.update(visible=False)
    else:
        yield (
            _render_trace(trace_events),
            "",
            gr.update(value="No answer was produced — try rephrasing the question.", visible=True),
        )


CUSTOM_CSS = """
.app-header {
    text-align: center;
    padding: 8px 0 4px 0;
}
.app-header h1 {
    font-size: 1.9rem;
    margin-bottom: 0.25rem;
}
.app-header p {
    color: var(--body-text-color-subdued);
    max-width: 680px;
    margin: 0 auto;
}
.trace-timeline {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.trace-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
    animation: trace-fade-in 0.25s ease-out;
}
.trace-icon { font-size: 1.15rem; line-height: 1.4; }
.trace-text { font-size: 0.92rem; line-height: 1.4; }
.trace-raw {
    display: block;
    font-family: var(--font-mono, monospace);
    font-size: 0.75rem;
    color: var(--body-text-color-subdued);
    margin-top: 2px;
}
.step-supervisor { border-left: 3px solid #f0883e; }
.step-agent { border-left: 3px solid #58a6ff; }
.step-draft { border-left: 3px solid #a371f7; }
.step-critic { border-left: 3px solid #3fb950; }
.trace-empty, .trace-thinking {
    padding: 14px;
    border-radius: 10px;
    border: 1px dashed var(--border-color-primary);
    color: var(--body-text-color-subdued);
    font-size: 0.92rem;
}
#answer-card textarea {
    font-size: 1.02rem !important;
    line-height: 1.5 !important;
}
"""

theme = gr.themes.Soft(primary_hue="orange", secondary_hue="slate")

with gr.Blocks(title=APP_TITLE, theme=theme, css=CUSTOM_CSS) as demo:
    gr.HTML("<div class='app-header'><h1>\U0001F916 " + APP_TITLE + "</h1><p>" + APP_SUBTITLE + "</p></div>")

    with gr.Row():
        with gr.Column(scale=5):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. How many transactions were categorized as Dining & Coffee?",
                lines=2,
                autofocus=True,
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary", scale=3)
                clear_btn = gr.ClearButton([question_box], value="Clear", scale=1)
            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=question_box,
                label="Try one of these",
            )

    notice_box = gr.Markdown(visible=False)

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Answer")
            answer_box = gr.Textbox(
                label="",
                lines=6,
                interactive=False,
                elem_id="answer-card",
                placeholder="The answer will appear here once the agents finish.",
            )
        with gr.Column(scale=2):
            gr.Markdown("### Live agent trace")
            trace_box = gr.HTML(_render_trace([]))

    question_box.submit(
        run_and_trace, inputs=question_box, outputs=[trace_box, answer_box, notice_box]
    )
    submit_btn.click(
        run_and_trace, inputs=question_box, outputs=[trace_box, answer_box, notice_box]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    use_share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    print(f"[startup] launching Gradio on 0.0.0.0:{port} (share={use_share})", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=port, share=use_share, ssr_mode=False)

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
APP_SUBTITLE = "Ask a question about your transactions. Watch four specialist agents answer it live."

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
    """Renders the trace as a connected vertical timeline — nodes on a line, like a pipeline log."""
    if not trace_events:
        return "<div class='trace-empty'>No steps yet — ask a question to see the agents work.</div>"

    rows = []
    for node_name, latest_step in trace_events:
        icon, label, css_class = NODE_INFO.get(node_name, ("\U0001F539", node_name, "step-agent"))
        note = _describe_step(node_name, latest_step)
        rows.append(
            "<div class='trace-step " + css_class + "'>"
            "<span class='trace-node'>" + icon + "</span>"
            "<div class='trace-body'>"
            "<span class='trace-label'>" + label + "</span> <span class='trace-note'>— " + note + "</span>"
            "<span class='trace-raw'>" + latest_step + "</span>"
            "</div>"
            "</div>"
        )
    return "<div class='trace-timeline'>" + "".join(rows) + "</div>"


def run_and_trace(question: str):
    """Streams each graph step to the UI as it happens, then the final answer."""
    question = (question or "").strip()
    if not question:
        yield _render_trace([]), PLACEHOLDER_ANSWER, gr.update(value="Ask a question above to get started.", visible=True)
        return

    app = get_app()
    trace_events = []
    final_answer = ""
    working_note = "_Agents are working on it…_"

    yield (
        "<div class='trace-thinking'>\U0001F916 Agents are working on it…</div>",
        working_note,
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
            yield _render_trace(trace_events), (final_answer or working_note), gr.update(visible=False)

    if final_answer:
        remember_turn(question, final_answer)
        yield _render_trace(trace_events), final_answer, gr.update(visible=False)
    else:
        yield (
            _render_trace(trace_events),
            "_No answer was produced — try rephrasing the question._",
            gr.update(value="No answer was produced — try rephrasing the question.", visible=True),
        )


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&display=swap');

:root {
    --ink: #0B0F1A;
    --panel: #121826;
    --panel-2: #17202F;
    --line: #263047;
    --amber: #F5A623;
    --amber-hover: #FFBB4D;
    --green: #34D399;
    --blue: #58A6FF;
    --violet: #A371F7;
    --text: #E8ECF6;
    --text-dim: #8792AA;
}

.gradio-container { font-family: 'Inter', ui-sans-serif, sans-serif; }

/* ---- header: ticker eyebrow + display headline ---- */
.app-header { text-align: center; padding: 4px 0 2px 0; }
.eyebrow-bar {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--text-dim); margin-bottom: 10px;
}
.pulse-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--amber);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(245,166,35,0.55); }
    70% { box-shadow: 0 0 0 7px rgba(245,166,35,0); }
    100% { box-shadow: 0 0 0 0 rgba(245,166,35,0); }
}
@media (prefers-reduced-motion: reduce) { .pulse-dot { animation: none; } }
.app-header h1 {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-weight: 700; font-size: 1.85rem; letter-spacing: -0.01em;
    margin-bottom: 0.3rem; color: var(--text);
}
.app-header p { color: var(--text-dim); max-width: 520px; margin: 0 auto; font-size: 0.95rem; }

/* ---- section labels ---- */
.section-label {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-dim); margin: 4px 0 8px 2px;
}
.section-label::before { content: '// '; color: var(--amber); }

/* ---- terminal-style question input ---- */
.cmd-input textarea, .cmd-input input {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.92rem !important;
}

/* ---- answer card: receipt styling ---- */
#answer-card {
    border: none;
    border-top: 2px dashed var(--line);
    border-bottom: 2px dashed var(--line);
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    padding: 20px 22px;
    background: var(--panel);
    min-height: 90px;
    font-size: 1.04rem;
    line-height: 1.6;
}
#answer-card::before {
    content: '// ANALYST OUTPUT';
    display: block;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10px; letter-spacing: 0.12em;
    color: var(--amber); margin-bottom: 10px;
}
#answer-card p:first-of-type { margin-top: 0; }
#answer-card p:last-of-type { margin-bottom: 0; }

/* ---- connected-line agent trace ---- */
#trace-panel { max-height: 440px; overflow-y: auto; padding: 4px 6px 4px 0; }
.trace-timeline { position: relative; padding-left: 44px; }
.trace-timeline::before {
    content: ''; position: absolute; left: 16px; top: 2px; bottom: 2px;
    width: 2px; background: var(--line);
}
.trace-step { position: relative; margin-bottom: 16px; animation: step-in 0.2s ease-out; }
@media (prefers-reduced-motion: reduce) { .trace-step { animation: none; } }
@keyframes step-in { from { opacity: 0; transform: translateX(-4px); } to { opacity: 1; transform: translateX(0); } }
.trace-node {
    position: absolute; left: -44px; top: 0; width: 32px; height: 32px;
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    background: var(--panel); border: 2px solid var(--line); font-size: 15px; z-index: 1;
}
.trace-body { font-size: 0.88rem; line-height: 1.5; padding-top: 4px; }
.trace-label { font-weight: 600; color: var(--text); }
.trace-note { color: var(--text-dim); }
.trace-raw {
    display: block; font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.72rem; color: var(--text-dim); opacity: 0.75; margin-top: 2px;
}
.step-supervisor .trace-node { border-color: var(--amber); }
.step-agent .trace-node { border-color: var(--blue); }
.step-draft .trace-node { border-color: var(--violet); }
.step-critic .trace-node { border-color: var(--green); }
.trace-empty, .trace-thinking {
    padding: 16px; border-radius: 8px; border: 1px dashed var(--line);
    color: var(--text-dim); font-size: 0.9rem; text-align: center;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
}

@media (max-width: 768px) {
    #trace-panel { max-height: 260px; }
    .app-header h1 { font-size: 1.4rem; }
    .trace-timeline { padding-left: 38px; }
    .trace-node { left: -38px; width: 28px; height: 28px; font-size: 13px; }
    .trace-timeline::before { left: 14px; }
}
"""

theme = gr.themes.Base(
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0B0F1A",
    body_background_fill_dark="#0B0F1A",
    body_text_color="#E8ECF6",
    body_text_color_dark="#E8ECF6",
    body_text_color_subdued="#8792AA",
    body_text_color_subdued_dark="#8792AA",
    background_fill_primary="#121826",
    background_fill_primary_dark="#121826",
    background_fill_secondary="#17202F",
    background_fill_secondary_dark="#17202F",
    border_color_primary="#263047",
    border_color_primary_dark="#263047",
    border_color_accent="#F5A623",
    border_color_accent_dark="#F5A623",
    color_accent="#F5A623",
    color_accent_soft="rgba(245,166,35,0.12)",
    color_accent_soft_dark="rgba(245,166,35,0.16)",
    block_background_fill="#121826",
    block_background_fill_dark="#121826",
    block_border_color="#263047",
    block_border_color_dark="#263047",
    block_label_background_fill="#121826",
    block_label_background_fill_dark="#121826",
    block_label_text_color="#8792AA",
    block_label_text_color_dark="#8792AA",
    block_title_text_color="#E8ECF6",
    block_title_text_color_dark="#E8ECF6",
    input_background_fill="#17202F",
    input_background_fill_dark="#17202F",
    input_border_color="#263047",
    input_border_color_dark="#263047",
    input_border_color_focus="#F5A623",
    input_border_color_focus_dark="#F5A623",
    input_placeholder_color="#54607A",
    input_placeholder_color_dark="#54607A",
    button_primary_background_fill="#F5A623",
    button_primary_background_fill_dark="#F5A623",
    button_primary_background_fill_hover="#FFBB4D",
    button_primary_background_fill_hover_dark="#FFBB4D",
    button_primary_text_color="#0B0F1A",
    button_primary_text_color_dark="#0B0F1A",
    button_secondary_background_fill="#17202F",
    button_secondary_background_fill_dark="#17202F",
    button_secondary_background_fill_hover="#1D273A",
    button_secondary_background_fill_hover_dark="#1D273A",
    button_secondary_text_color="#E8ECF6",
    button_secondary_text_color_dark="#E8ECF6",
    button_secondary_border_color="#263047",
    button_secondary_border_color_dark="#263047",
    panel_background_fill="#121826",
    panel_background_fill_dark="#121826",
    panel_border_color="#263047",
    panel_border_color_dark="#263047",
    code_background_fill="#17202F",
    code_background_fill_dark="#17202F",
)

PLACEHOLDER_ANSWER = "_The answer will appear here once the agents finish — usually a few seconds._"

with gr.Blocks(title=APP_TITLE) as demo:
    gr.HTML(
        "<div class='app-header'>"
        "<div class='eyebrow-bar'><span class='pulse-dot'></span>Live &nbsp;·&nbsp; AI Analyst // Transaction Intelligence</div>"
        "<h1>" + APP_TITLE + "</h1>"
        "<p>" + APP_SUBTITLE + "</p>"
        "</div>"
    )

    question_box = gr.Textbox(
        label="$ your question",
        placeholder="e.g. How many transactions were categorized as Dining & Coffee?",
        lines=2,
        autofocus=True,
        elem_classes=["cmd-input"],
    )
    with gr.Row():
        submit_btn = gr.Button("Ask", variant="primary", scale=4)
        clear_btn = gr.ClearButton([question_box], value="Clear", scale=1)
    gr.Examples(
        examples=EXAMPLE_QUESTIONS,
        inputs=question_box,
        label="Try one of these",
    )

    notice_box = gr.Markdown(visible=False)

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("Answer", elem_classes=["section-label"])
            answer_box = gr.Markdown(
                value=PLACEHOLDER_ANSWER,
                elem_id="answer-card",
            )
        with gr.Column(scale=2):
            gr.Markdown("Live agent trace", elem_classes=["section-label"])
            with gr.Column(elem_id="trace-panel"):
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
    demo.launch(server_name="0.0.0.0", server_port=port, share=use_share, ssr_mode=False, css=CUSTOM_CSS)

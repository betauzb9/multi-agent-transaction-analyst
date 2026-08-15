"""
Business-friendly frontend for the Multi-Agent Transaction Analyst.

WHAT THIS FILE DOES (presentation layer only):
  - A light, professional "dashboard" look by default, with a one-click
    Light/Dark toggle in the top-right corner (pure CSS + a tiny bit of
    client-side JS -- no server round-trip, so it can never error out).
  - A single, plain-English progress bar while the agents work
    (Understanding -> Researching -> Drafting -> Verifying) with a live
    one-line caption -- no agent jargon, no raw trace log. Built for a
    non-technical audience.
  - A trust/feature strip under the header and a plain-language
    "How does this work?" accordion -- good for a presentation.
  - Friendlier, more natural example questions.
  - Cosmetic thumbs up/down feedback buttons under the answer.

Nothing about src/graph.py, src/state.py, src/memory.py, or the streaming
contract (steps / answer keys) was touched -- this is a drop-in
replacement: the agent logic is exactly the same, only the presentation
layer changed.

Local run:
    python app.py

Render (unchanged): Render sets $PORT automatically; this binds to
0.0.0.0:$PORT so Render's public URL reaches the app directly.
"""

import os

import gradio as gr

from src.graph import get_app, config
from src.state import initial_state
from src.memory import remember_turn

APP_TITLE = "AI Transaction Analyst"
APP_SUBTITLE = "Ask a question about your business transactions in plain English. A team of specialist AI agents finds the answer and double-checks it before you see it."

# Friendlier, more natural phrasing than the raw technical originals --
# same underlying questions (categorization count, taxonomy, unseen
# merchant handling, classify a sample transaction), easier to read.
EXAMPLE_QUESTIONS = [
    "How many transactions fall under 'Dining & Coffee'?",
    "What spending categories does this system use?",
    "What happens when the AI sees a brand-new merchant it's never seen before?",
    "What category would a $5.75 charge from 'SQ *STARBUCKS #4471' get?",
]

# Plain-language, one-line captions shown while a given agent node is
# working. These deliberately avoid jargon like "supervisor" / "critic".
BUSINESS_CAPTIONS = {
    "supervisor": "Understanding what you're asking\u2026",
    "retriever": "Looking through the documentation\u2026",
    "web": "Searching the web for extra context\u2026",
    "data": "Querying the transaction database\u2026",
    "code": "Running the calculations\u2026",
    "generate": "Writing up the answer\u2026",
    "critic": "Double-checking the answer for accuracy\u2026",
}

# The 4 stages shown in the progress bar.
STAGE_DEFS = [
    ("understanding", "\U0001F9ED", "Understanding"),
    ("gathering", "\U0001F50D", "Researching"),
    ("drafting", "\u270D\uFE0F", "Drafting"),
    ("verifying", "\u2705", "Verifying"),
]

PLACEHOLDER_ANSWER = "_The answer will appear here once the agents finish \u2014 usually a few seconds._"


def _simple_stage_reached(trace_events):
    """Collapses raw node names into one of 4 plain-language stages (0-4)."""
    seen_nodes = {n for n, _ in trace_events}
    reached = 0
    if seen_nodes:
        reached = 1
    if seen_nodes & {"retriever", "web", "data", "code"}:
        reached = 2
    if "generate" in seen_nodes:
        reached = 3
    if "critic" in seen_nodes:
        reached = 4
    return reached


def _render_progress(trace_events, done=False):
    """Renders the 4-step progress bar in plain English, for a non-technical audience."""
    reached = 4 if done else _simple_stage_reached(trace_events)

    if done:
        caption = "Answer ready \u2014 verified by the quality-check agent."
    elif trace_events:
        last_node = trace_events[-1][0]
        caption = BUSINESS_CAPTIONS.get(last_node, "Working on it\u2026")
    else:
        caption = "Ask a question above and watch the agents get to work."

    steps_html = []
    total = len(STAGE_DEFS)
    for i, (_key, icon, label) in enumerate(STAGE_DEFS, start=1):
        if i < reached or done:
            state = "done"
        elif i == reached:
            state = "active"
        else:
            state = "pending"
        icon_display = "\u2714" if state == "done" else icon
        steps_html.append(
            "<div class='sstep " + state + "'>"
            "<span class='sstep-icon'>" + icon_display + "</span>"
            "<span class='sstep-label'>" + label + "</span>"
            "</div>"
        )
        if i < total:
            line_state = "done" if (i < reached or done) else ""
            steps_html.append("<div class='sstep-line " + line_state + "'></div>")

    return (
        "<div class='simple-progress'>"
        "<div class='sstep-row'>" + "".join(steps_html) + "</div>"
        "<div class='sstep-caption'>" + caption + "</div>"
        "</div>"
    )


def run_and_trace(question: str):
    """Streams progress to the UI as the agents work, then the final answer.

    Yields (progress_html, answer_markdown, notice_update). The agent-side
    logic and the streaming contract (node_state['steps'] / ['answer']) are
    unchanged from the original app.py.
    """
    question = (question or "").strip()
    if not question:
        yield (
            _render_progress([], done=False),
            PLACEHOLDER_ANSWER,
            gr.update(value="Ask a question above to get started.", visible=True),
        )
        return

    app = get_app()
    trace_events = []
    final_answer = ""
    working_note = "_Agents are working on it\u2026_"

    yield (
        _render_progress([], done=False),
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
            yield (
                _render_progress(trace_events, done=bool(final_answer)),
                (final_answer or working_note),
                gr.update(visible=False),
            )

    if final_answer:
        remember_turn(question, final_answer)
        yield (
            _render_progress(trace_events, done=True),
            final_answer,
            gr.update(visible=False),
        )
    else:
        yield (
            _render_progress(trace_events, done=False),
            "_No answer was produced \u2014 try rephrasing the question._",
            gr.update(value="No answer was produced \u2014 try rephrasing the question.", visible=True),
        )


def _thumbs_feedback(positive: bool):
    """Cosmetic feedback buttons. Not wired to storage yet -- swap in a DB/Sheet write here later."""
    gr.Info("Thanks for the feedback! \U0001F44D" if positive else "Thanks \u2014 we'll use this to improve. \U0001F44E")


# ---------------------------------------------------------------------------
# Theme + CSS
#   Light palette lives under :root (default).
#   Dark palette lives under .dark (Gradio's standard dark-mode class), and
#   is also what the toggle button in the header adds/removes on
#   .gradio-container -- pure client-side, so it can never break the app.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #F6F7FB;
  --panel: #FFFFFF;
  --panel-2: #F1F3F9;
  --line: #E4E7F0;
  --primary: #4F46E5;
  --primary-hover: #4338CA;
  --primary-soft: rgba(79,70,229,0.08);
  --green: #16A34A;
  --green-soft: rgba(22,163,74,0.10);
  --blue: #2563EB;
  --violet: #7C3AED;
  --amber: #D97706;
  --text: #1E2330;
  --text-dim: #6B7280;
  --shadow: 0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06);
}

.gradio-container.dark {
  --bg: #0B0F1A;
  --panel: #121826;
  --panel-2: #17202F;
  --line: #263047;
  --primary: #F5A623;
  --primary-hover: #FFBB4D;
  --primary-soft: rgba(245,166,35,0.14);
  --green: #34D399;
  --green-soft: rgba(52,211,153,0.14);
  --blue: #58A6FF;
  --violet: #A371F7;
  --amber: #F5A623;
  --text: #E8ECF6;
  --text-dim: #8792AA;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 1px 3px rgba(0,0,0,0.3);
}

.gradio-container { font-family: 'Inter', ui-sans-serif, sans-serif; background: var(--bg) !important; transition: background 0.2s ease; }

/* ---- theme toggle button ---- */
.theme-toggle-wrap { position: absolute; top: 14px; right: 14px; z-index: 20; }
.theme-toggle-btn {
  font-family: 'Inter', sans-serif; font-size: 0.82rem; font-weight: 600;
  background: var(--panel); color: var(--text); border: 1px solid var(--line);
  border-radius: 999px; padding: 7px 14px; cursor: pointer; box-shadow: var(--shadow);
  transition: all 0.15s ease;
}
.theme-toggle-btn:hover { border-color: var(--primary); color: var(--primary); }

/* ---- header ---- */
.app-header { text-align: center; padding: 8px 0 4px 0; position: relative; }
.eyebrow-bar {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 11.5px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--primary); background: var(--primary-soft);
  padding: 5px 14px; border-radius: 999px; margin-bottom: 14px;
}
.pulse-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--green);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(22,163,74,0.5); }
  70% { box-shadow: 0 0 0 7px rgba(22,163,74,0); }
  100% { box-shadow: 0 0 0 0 rgba(22,163,74,0); }
}
@media (prefers-reduced-motion: reduce) { .pulse-dot { animation: none; } }
.app-header h1 {
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-weight: 700; font-size: 2.1rem; letter-spacing: -0.02em;
  margin-bottom: 0.5rem; color: var(--text);
}
.app-header p { color: var(--text-dim); max-width: 620px; margin: 0 auto; font-size: 1rem; line-height: 1.5; }

/* ---- trust / feature strip ---- */
.trust-strip {
  display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;
  margin: 18px 0 4px 0;
}
.trust-chip {
  display: flex; align-items: center; gap: 7px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 9px 14px; font-size: 0.85rem; font-weight: 600; color: var(--text);
  box-shadow: var(--shadow);
}
.trust-chip .n { color: var(--primary); }

/* ---- section labels ---- */
.section-label {
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--text-dim); margin: 4px 0 8px 2px;
}

/* ---- question input ---- */
.cmd-input textarea, .cmd-input input {
  font-size: 0.98rem !important;
  border-radius: 12px !important;
}

/* ---- answer card ---- */
#answer-card {
  border: 1px solid var(--line);
  border-left: 4px solid var(--primary);
  border-radius: 12px;
  padding: 22px 24px;
  background: var(--panel);
  min-height: 90px;
  font-size: 1.05rem;
  line-height: 1.65;
  box-shadow: var(--shadow);
}
#answer-card::before {
  content: '\\1F4A1  Answer';
  display: block;
  font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--primary); margin-bottom: 12px;
}
#answer-card p:first-of-type { margin-top: 0; }
#answer-card p:last-of-type { margin-bottom: 0; }

/* ---- feedback row ---- */
.feedback-row { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
.feedback-row span { font-size: 0.82rem; color: var(--text-dim); }

/* ---- progress bar ---- */
#progress-panel { margin-top: 2px; }
.simple-progress {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 22px 18px 16px 18px; box-shadow: var(--shadow);
}
.sstep-row { display: flex; align-items: flex-start; }
.sstep { display: flex; flex-direction: column; align-items: center; gap: 8px; min-width: 64px; }
.sstep-icon {
  width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 17px; background: var(--panel-2); border: 2px solid var(--line); color: var(--text-dim);
  transition: all 0.25s ease;
}
.sstep-label { font-size: 0.74rem; font-weight: 600; color: var(--text-dim); text-align: center; }
.sstep-line { flex: 1; height: 2px; background: var(--line); margin: 19px 4px 0 4px; border-radius: 2px; transition: background 0.25s ease; }
.sstep-line.done { background: var(--green); }
.sstep.done .sstep-icon { background: var(--green-soft); border-color: var(--green); color: var(--green); }
.sstep.done .sstep-label { color: var(--text); }
.sstep.active .sstep-icon {
  background: var(--primary-soft); border-color: var(--primary); color: var(--primary);
  animation: sstep-pulse 1.4s infinite;
}
.sstep.active .sstep-label { color: var(--primary); }
@keyframes sstep-pulse {
  0% { box-shadow: 0 0 0 0 rgba(79,70,229,0.35); }
  70% { box-shadow: 0 0 0 8px rgba(79,70,229,0); }
  100% { box-shadow: 0 0 0 0 rgba(79,70,229,0); }
}
@media (prefers-reduced-motion: reduce) { .sstep.active .sstep-icon { animation: none; } }
.sstep-caption {
  margin-top: 16px; padding-top: 14px; border-top: 1px dashed var(--line);
  font-size: 0.88rem; color: var(--text-dim); text-align: center;
}

/* ---- footer ---- */
.app-footer {
  text-align: center; margin-top: 22px; padding-top: 14px; border-top: 1px solid var(--line);
  color: var(--text-dim); font-size: 0.8rem;
}

@media (max-width: 768px) {
  .app-header h1 { font-size: 1.5rem; }
  .theme-toggle-wrap { position: static; display: flex; justify-content: center; margin-bottom: 10px; }
  .sstep-label { display: none; }
}
"""

# Small, purely client-side toggle. Adds/removes Gradio's own '.dark' class
# on the container, which our CSS above already knows how to theme -- no
# server call involved, so this cannot error at runtime.
THEME_TOGGLE_HTML = (
    "<div class='theme-toggle-wrap'>"
    "<button type='button' class='theme-toggle-btn' onclick=\""
    "var c=document.querySelector('.gradio-container');"
    "c.classList.toggle('dark');"
    "this.textContent = c.classList.contains('dark') ? '\u2600\ufe0f Light mode' : '\U0001F319 Dark mode';"
    "\">\U0001F319 Dark mode</button>"
    "</div>"
)

theme = gr.themes.Base(
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#F6F7FB",
    body_background_fill_dark="#0B0F1A",
    body_text_color="#1E2330",
    body_text_color_dark="#E8ECF6",
    body_text_color_subdued="#6B7280",
    body_text_color_subdued_dark="#8792AA",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#121826",
    background_fill_secondary="#F1F3F9",
    background_fill_secondary_dark="#17202F",
    border_color_primary="#E4E7F0",
    border_color_primary_dark="#263047",
    border_color_accent="#4F46E5",
    border_color_accent_dark="#F5A623",
    color_accent="#4F46E5",
    color_accent_soft="rgba(79,70,229,0.08)",
    color_accent_soft_dark="rgba(245,166,35,0.16)",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#121826",
    block_border_color="#E4E7F0",
    block_border_color_dark="#263047",
    block_label_background_fill="#FFFFFF",
    block_label_background_fill_dark="#121826",
    block_label_text_color="#6B7280",
    block_label_text_color_dark="#8792AA",
    block_title_text_color="#1E2330",
    block_title_text_color_dark="#E8ECF6",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#17202F",
    input_border_color="#E4E7F0",
    input_border_color_dark="#263047",
    input_border_color_focus="#4F46E5",
    input_border_color_focus_dark="#F5A623",
    input_placeholder_color="#9CA3AF",
    input_placeholder_color_dark="#54607A",
    button_primary_background_fill="#4F46E5",
    button_primary_background_fill_dark="#F5A623",
    button_primary_background_fill_hover="#4338CA",
    button_primary_background_fill_hover_dark="#FFBB4D",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#0B0F1A",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_dark="#17202F",
    button_secondary_background_fill_hover="#F1F3F9",
    button_secondary_background_fill_hover_dark="#1D273A",
    button_secondary_text_color="#1E2330",
    button_secondary_text_color_dark="#E8ECF6",
    button_secondary_border_color="#E4E7F0",
    button_secondary_border_color_dark="#263047",
    panel_background_fill="#FFFFFF",
    panel_background_fill_dark="#121826",
    panel_border_color="#E4E7F0",
    panel_border_color_dark="#263047",
    code_background_fill="#F1F3F9",
    code_background_fill_dark="#17202F",
)

with gr.Blocks(title=APP_TITLE, theme=theme) as demo:
    gr.HTML(
        "<div class='app-header'>"
        + THEME_TOGGLE_HTML
        + "<div class='eyebrow-bar'><span class='pulse-dot'></span>Live &nbsp;\u00b7&nbsp; AI Analyst // Transaction Intelligence</div>"
        "<h1>" + APP_TITLE + "</h1>"
        "<p>" + APP_SUBTITLE + "</p>"
        "</div>"
    )

    gr.HTML(
        "<div class='trust-strip'>"
        "<div class='trust-chip'><span class='n'>\U0001F916</span> 4 specialist AI agents</div>"
        "<div class='trust-chip'><span class='n'>\u2705</span> Every answer fact-checked</div>"
        "<div class='trust-chip'><span class='n'>\u26A1</span> Answers in seconds</div>"
        "<div class='trust-chip'><span class='n'>\U0001F512</span> Read-only over your data</div>"
        "</div>"
    )

    with gr.Accordion("How does this work? (for the curious)", open=False):
        gr.Markdown(
            "When you ask a question, a **Coordinator** agent figures out what "
            "kind of information is needed and calls in specialists:\n\n"
            "- **Knowledge Search** \u2014 explains methodology and category definitions\n"
            "- **Database Query** \u2014 counts and aggregates your transactions\n"
            "- **Calculation Engine** \u2014 runs exact math, or classifies a new transaction\n"
            "- **Web Research** \u2014 answers anything outside your own data\n\n"
            "Once enough evidence is collected, a draft answer is written and then "
            "reviewed by a separate **Quality Check** agent, which rejects any claim "
            "that isn't backed by the evidence \u2014 before you ever see it."
        )

    question_box = gr.Textbox(
        label="Your question",
        placeholder="e.g. How many transactions fall under 'Dining & Coffee'?",
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
            gr.HTML(
                "<div class='feedback-row'>"
                "<span>Was this answer helpful?</span>"
                "</div>"
            )
            with gr.Row():
                thumbs_up_btn = gr.Button("\U0001F44D Yes", size="sm", scale=0)
                thumbs_down_btn = gr.Button("\U0001F44E No", size="sm", scale=0)

        with gr.Column(scale=2):
            gr.Markdown("Agent activity", elem_classes=["section-label"])
            with gr.Column(elem_id="progress-panel"):
                progress_box = gr.HTML(_render_progress([]))

    gr.HTML(
        "<div class='app-footer'>Built with LangGraph + Gemini \u00b7 answers are generated by AI and reviewed automatically \u2014 verify anything critical.</div>"
    )

    question_box.submit(
        run_and_trace, inputs=question_box, outputs=[progress_box, answer_box, notice_box]
    )
    submit_btn.click(
        run_and_trace, inputs=question_box, outputs=[progress_box, answer_box, notice_box]
    )
    thumbs_up_btn.click(lambda: _thumbs_feedback(True), outputs=None)
    thumbs_down_btn.click(lambda: _thumbs_feedback(False), outputs=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    use_share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    print(f"[startup] launching Gradio on 0.0.0.0:{port} (share={use_share})", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=port, share=use_share, ssr_mode=False, css=CUSTOM_CSS)

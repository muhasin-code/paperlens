"""PaperLens Gradio UI: web interface for the RAG query endpoint.

Claude-inspired design: clean chat bubbles, inline citation chips, latency waterfall,
model badge, health pill, keyboard-first interaction.
"""

import asyncio

import gradio as gr
import httpx

from src.paperlens.settings import get_settings
from src.paperlens.ui.formatters import format_citations, format_health, format_latency

settings = get_settings()
API_BASE = f"http://{settings.api_host}:{settings.api_port}"


# Claude-inspired Custom CSS
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* General Container Styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    background-color: #fbfaf7 !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 1.5rem !important;
}

/* Header Styling */
.header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #e2dfd9;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}

.logo-text {
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 1.625rem;
    font-weight: 700;
    color: #191919;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.status-pill {
    padding: 0.35rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8125rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

.status-green {
    background-color: #ecfdf5;
    color: #047857;
    border-color: #a7f3d0;
}

.status-amber {
    background-color: #fffbeb;
    color: #b45309;
    border-color: #fde68a;
}

.status-red {
    background-color: #fef2f2;
    color: #b91c1c;
    border-color: #fecaca;
}

/* Chatbot Styling (Claude-inspired bubbles) */
.paperlens-chatbot {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.paperlens-chatbot .message.user {
    background-color: #f3f0ec !important;
    color: #191919 !important;
    border: 1px solid #d4d1cb !important;
    border-radius: 1rem 1rem 0.25rem 1rem !important;
    padding: 1rem 1.25rem !important;
    max-width: 80% !important;
    margin-left: auto !important;
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 1.0625rem !important;
    line-height: 1.6 !important;
}

.paperlens-chatbot .message.bot {
    background-color: #ffffff !important;
    color: #191919 !important;
    border-radius: 1rem 1rem 1rem 0.25rem !important;
    padding: 1.25rem 1.5rem !important;
    max-width: 85% !important;
    border: 1px solid #e2dfd9 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03) !important;
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 1.0625rem !important;
    line-height: 1.6 !important;
}

/* Citation Chip Styling (inline in answer bubbles) */
.cite-chip {
    display: inline-flex;
    align-items: center;
    background-color: #f5ebd8;
    color: #c2410c;
    border: 1px solid #f1c385;
    border-radius: 0.375rem;
    padding: 0.125rem 0.375rem;
    font-size: 0.75rem;
    font-weight: 600;
    cursor: pointer;
    margin: 0 0.25rem;
    transition: all 0.2s ease;
    font-family: 'Inter', sans-serif !important;
}

.cite-chip:hover {
    background-color: #ebdcb9;
    transform: translateY(-1px);
}

/* Sample Question Cards */
.sample-card {
    background-color: #ffffff !important;
    border: 1px solid #e2dfd9 !important;
    border-radius: 0.75rem !important;
    padding: 1rem !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-align: left !important;
    height: 100% !important;
}

.sample-card:hover {
    border-color: #c2410c !important;
    box-shadow: 0 6px 12px -3px rgba(194, 65, 12, 0.05) !important;
    transform: translateY(-2px) !important;
}

.sample-card-title {
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 600 !important;
    font-size: 0.9375rem !important;
    color: #191919 !important;
    margin-bottom: 0.25rem !important;
}

.sample-card-desc {
    font-size: 0.8125rem !important;
    color: #5c5b57 !important;
}

/* Timing waterfall box */
.latency-waterfall {
    background-color: #f3f2ee;
    border: 1px solid #e2dfd9;
    border-radius: 0.5rem;
    padding: 0.75rem;
    font-size: 0.8125rem;
    color: #5c5b57;
    margin-top: 0.75rem;
    font-family: 'Inter', sans-serif !important;
}

/* Active Model Badge */
.model-badge {
    background-color: #f3f2ee;
    border: 1px solid #e2dfd9;
    color: #5c5b57;
    padding: 0.25rem 0.625rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
}

/* Input layout adjustments */
.input-container {
    background: #ffffff;
    border: 1px solid #e2dfd9;
    border-radius: 1rem;
    padding: 0.5rem;
    box-shadow: 0 4px 12px -3px rgba(0, 0, 0, 0.02) !important;
}

.input-container:focus-within {
    border-color: #c2410c;
    box-shadow: 0 0 0 2px rgba(194, 65, 12, 0.1) !important;
}

.char-counter {
    font-size: 0.75rem;
    color: #8c8a82;
    text-align: right;
    margin-top: 0.25rem;
}
"""


async def query_backend(prompt: str, top_k: int, model: str) -> dict:
    """
    Asynchronously POST a query to the FastAPI backend /query endpoint.
    """
    url = f"{API_BASE}/query"
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            url,
            json={"query": prompt, "top_k": top_k, "model": model},
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_health() -> dict:
    """
    Fetch the system status and health indicators from /health endpoint.
    """
    url = f"{API_BASE}/health"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    # Return mock offline status
    return {
        "status": "degraded",
        "chroma_collection": "paperlens_chunks",
        "chroma_vector_count": 0,
        "ollama_reachable": False,
        "ollama_model": settings.ollama_model,
        "ollama_fallback_model": "llama3.2:1b",
    }


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(title="PaperLens - Research Intelligence") as demo:
        # State variables
        citations_data = gr.State([])

        # Header Bar
        with gr.Row(elem_classes="header-bar"):
            with gr.Column(scale=8):
                gr.HTML(
                    '<div class="logo-text">\U0001f4da PaperLens '
                    '<span style="font-size: 0.875rem; font-weight: 500; '
                    "color: #6b7280; padding: 0.25rem 0.5rem; "
                    'background-color: #f3f4f6; border-radius: 0.375rem;">'
                    "v0.1.0 (Phase 1)</span></div>"
                )
            with gr.Column(scale=4, min_width=200):
                health_pill = gr.HTML(
                    value="<div class='status-pill status-amber'>\U0001f7e1 Fetching health...</div>"
                )

        # Hidden timer/refresh for health (polls every 10s)
        health_trigger = gr.Timer(10.0, active=True)

        async def update_health_display():
            health_data = await fetch_health()
            status_text, color = format_health(health_data)
            status_class = f"status-{color}"
            pill_html = f'<div class="status-pill {status_class}">{status_text}</div>'
            return pill_html

        health_trigger.tick(update_health_display, outputs=health_pill)
        demo.load(update_health_display, outputs=health_pill)

        # Main Workspace: 2-column layout (Chat side, Citation side)
        with gr.Row():
            # Left side: Chat area & inputs
            with gr.Column(scale=8):
                # Chat display viewport
                chatbot = gr.Chatbot(
                    label="Research Intelligence Chatbot",
                    elem_classes="paperlens-chatbot",
                    layout="bubble",
                    height=500,
                )

                # Empty State Cards (only visible before starting chat)
                with gr.Group() as empty_state:
                    gr.Markdown("### \U0001f50d Sample Questions to Begin")
                    with gr.Row():
                        with gr.Column():
                            q1_btn = gr.Button(
                                "\U0001f4a1 **Learning Rate Schedules**\nHow do cosine annealing schedules compare to linear decay for transformers?",
                                elem_classes="sample-card",
                                size="sm",
                            )
                        with gr.Column():
                            q2_btn = gr.Button(
                                "\U0001f4a1 **LoRA & Parameter-Efficient Tuning**\nWhat is the mathematical formulation of Low-Rank Adaptation (LoRA)?",
                                elem_classes="sample-card",
                                size="sm",
                            )
                        with gr.Column():
                            q3_btn = gr.Button(
                                "\U0001f4a1 **Transformer Attention Variants**\nHow does FlashAttention optimize the softmax computation step?",
                                elem_classes="sample-card",
                                size="sm",
                            )

                # Active message stats and latency bar
                latency_view = gr.Markdown(value="", elem_classes="latency-waterfall")

                # Input container
                with gr.Row(elem_classes="input-container"):
                    with gr.Column(scale=10):
                        user_input = gr.Textbox(
                            show_label=False,
                            placeholder="Ask PaperLens about ML literature (e.g., 'Compare LoRA and QLoRA')... [Enter to send]",
                            lines=2,
                            max_lines=6,
                            elem_id="query-textarea",
                        )
                    with gr.Column(scale=2, min_width=100):
                        send_btn = gr.Button("Send \u26a1", variant="primary", size="sm")
                        new_chat_btn = gr.Button(
                            "New Chat \U0001f9f9", variant="secondary", size="sm"
                        )

                # Input configuration bar
                with gr.Row():
                    with gr.Column(scale=6):
                        # Model Selector
                        model_selector = gr.Radio(
                            choices=[settings.ollama_model, "llama3.2:1b"],
                            value=settings.ollama_model,
                            label="Ollama LLM Engine Selection",
                            interactive=True,
                        )
                    with gr.Column(scale=6):
                        with gr.Accordion("Advanced Retrieval Parameters (Top-K)", open=False):
                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=settings.retrieval_top_k,
                                step=1,
                                label="Chunks Retrieve Count",
                            )

            # Right side: Citation drawer
            with gr.Column(scale=4):
                gr.Markdown("### \U0001f4d1 Cited Source Chunks")
                citation_search = gr.Textbox(
                    show_label=False,
                    placeholder="Search cited chunks / author / title...",
                    lines=1,
                    max_lines=1,
                )
                citation_panel = gr.Markdown(
                    value="*Citations from your queries will appear here automatically.*",
                    elem_id="citation-viewer",
                )

        # Chat and citation updates
        async def handle_submit(
            prompt: str, chat_history: list, active_cites: list, top_k: int, selected_model: str
        ):
            if not prompt.strip():
                yield "", chat_history, active_cites, "", gr.update()
                return

            # Append user message to history
            chat_history.append({"role": "user", "content": prompt})

            # Temporary bot response for loading cursor
            chat_history.append({"role": "assistant", "content": "\u258c"})
            yield (
                "",
                chat_history,
                active_cites,
                "\u23f1\ufe0f Retrieving sources...",
                gr.update(visible=False),
            )

            # Query the backend
            try:
                result = await query_backend(prompt, top_k, selected_model)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 502:
                    # Fallback handled by backend; still show error gracefully
                    error_msg = f"\u26a0\ufe0f **Backend Error (502)**: {exc.response.text}"
                    chat_history[-1]["content"] = error_msg
                    yield "", chat_history, active_cites, "", gr.update(visible=False)
                    return
                raise
            except httpx.RequestError:
                error_msg = f"\u274c **Connection Error**: Could not reach API at `{API_BASE}`. Is the server running? (`make run`)"
                chat_history[-1]["content"] = error_msg
                yield "", chat_history, active_cites, "", gr.update(visible=False)
                return

            # Format answers and latency
            answer = result.get("answer", "No response.")
            citations = result.get("citations", [])
            ret_time = result.get("retrieval_time_ms", 0)
            gen_time = result.get("generation_time_ms", 0)
            tot_time = result.get("total_time_ms", 0)

            latency_html = format_latency(ret_time, gen_time, tot_time)

            # Save citations to state
            updated_cites = active_cites + citations

            # Token-by-token simulated streaming of response (Phase 4.2 Stream readiness)
            words = answer.split(" ")
            current_answer = ""
            for word in words:
                current_answer += word + " "
                chat_history[-1]["content"] = current_answer + "\u258c"
                await asyncio.sleep(0.01)  # Simulated smooth token rendering
                yield "", chat_history, updated_cites, latency_html, gr.update(visible=False)

            # Finish text streaming with finalized answer
            chat_history[-1]["content"] = current_answer.strip()
            yield "", chat_history, updated_cites, latency_html, gr.update(visible=False)

        # Trigger on Submit / Clicks
        send_btn.click(
            handle_submit,
            inputs=[user_input, chatbot, citations_data, top_k_slider, model_selector],
            outputs=[user_input, chatbot, citations_data, latency_view, empty_state],
        ).then(fn=lambda c: format_citations(c), inputs=[citations_data], outputs=[citation_panel])

        user_input.submit(
            handle_submit,
            inputs=[user_input, chatbot, citations_data, top_k_slider, model_selector],
            outputs=[user_input, chatbot, citations_data, latency_view, empty_state],
        ).then(fn=lambda c: format_citations(c), inputs=[citations_data], outputs=[citation_panel])

        # New Chat reset
        new_chat_btn.click(
            fn=lambda: ([], [], ""), inputs=[], outputs=[chatbot, citations_data, latency_view]
        ).then(
            fn=lambda: (
                gr.update(value=""),
                gr.update(visible=True),
                gr.update(value="*Citations from your queries will appear here automatically.*"),
            ),
            outputs=[user_input, empty_state, citation_panel],
        )

        # Sample question click flows
        def load_q1():
            return "How do cosine annealing schedules compare to linear decay for transformers?"

        def load_q2():
            return "What is the mathematical formulation of Low-Rank Adaptation (LoRA)?"

        def load_q3():
            return "How does FlashAttention optimize the softmax computation step?"

        q1_btn.click(fn=load_q1, outputs=user_input)
        q2_btn.click(fn=load_q2, outputs=user_input)
        q3_btn.click(fn=load_q3, outputs=user_input)

        # Citation search filter
        def filter_citations(search_text: str, current_cites: list) -> str:
            if not search_text.strip():
                return format_citations(current_cites)
            filtered = [
                c
                for c in current_cites
                if search_text.lower() in c.get("title", "").lower()
                or search_text.lower()
                in (
                    ", ".join(c.get("authors", ""))
                    if isinstance(c.get("authors"), list)
                    else c.get("authors", "")
                ).lower()
                or search_text.lower() in c.get("chunk_id", "").lower()
            ]
            return format_citations(filtered)

        citation_search.change(
            fn=filter_citations, inputs=[citation_search, citations_data], outputs=[citation_panel]
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_port,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="slate", secondary_hue="zinc"),
        css=CUSTOM_CSS,
    )

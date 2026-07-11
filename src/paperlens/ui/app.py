"""Gradio web UI for PaperLens.

Calls the FastAPI /query and /health endpoints over HTTP.
Displays answer, expandable citations, and latency breakdown.
"""

import gradio as gr
import httpx

from src.paperlens.settings import get_settings

settings = get_settings()
API_BASE = f"http://{settings.api_host}:{settings.api_port}"


async def query_api(question: str, top_k: int | None, model: str | None) -> dict:
    """Call FastAPI /query endpoint and return parsed JSON."""
    payload = {"query": question}
    if top_k is not None:
        payload["top_k"] = top_k
    if model is not None:
        payload["model"] = model

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{API_BASE}/query", json=payload)
        resp.raise_for_status()
        return resp.json()


async def health_api() -> dict:
    """Call FastAPI /health endpoint."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{API_BASE}/health")
        resp.raise_for_status()
        return resp.json()


def format_citations(citations: list[dict]) -> str:
    """Render citations as markdown with expandable details."""
    if not citations:
        return "_No sources retrieved._"

    lines = []
    for c in citations:
        header = f"**[{c['rank']}] {c['title']}** (`{c['chunk_id']}`)"
        details = (
            f"<details><summary>{header}</summary>\n\n"
            f"**arXiv ID:** {c['arxiv_id']}  \n"
            f"**Authors:** {', '.join(c['authors'])}  \n"
            f"**Section:** {c['section_label']}  \n"
            f"**Pages:** {c['page_start']}–{c['page_end']}  \n"
            f"**Score:** {c['score']:.4f}  \n"
            f"**Rank:** {c['rank']}\n\n"
            f"</details>"
        )
        lines.append(details)
    return "\n\n".join(lines)


def format_latency(
    retrieval_ms: float, generation_ms: float, total_ms: float, confidence: float
) -> str:
    """Render latency breakdown as markdown."""
    return (
        f"**Retrieval:** {retrieval_ms:.1f} ms  \n"
        f"**Generation:** {generation_ms:.1f} ms  \n"
        f"**Total:** {total_ms:.1f} ms  \n"
        f"**Confidence:** {confidence:.3f}"
    )


def format_health(health: dict) -> str:
    """Render health status as markdown badge + details."""
    status = health.get("status", "unknown")
    badge_color = "green" if status == "ok" else "orange" if status == "degraded" else "red"
    badge = f"![status](https://img.shields.io/badge/status-{status}-{badge_color})"

    lines = [
        badge,
        f"**ChromaDB:** {health.get('chroma_collection', '?')} ({health.get('chroma_vector_count', 0)} vectors)",
        f"**Ollama:** {'✓ reachable' if health.get('ollama_reachable') else '✗ unreachable'}",
        f"**Model:** {health.get('ollama_model', '?')} (fallback: {health.get('ollama_fallback_model', '?')})",
    ]
    return "\n\n".join(lines)


async def handle_query(question: str, top_k: int, model: str) -> tuple[str, str, str]:
    """Gradio event handler: query API, return (answer_md, citations_md, latency_md)."""
    if not question.strip():
        return "Please enter a question.", "", ""

    try:
        data = await query_api(question, top_k if top_k > 0 else None, model if model else None)
        answer_md = data["answer"]
        citations_md = format_citations(data["citations"])
        latency_md = format_latency(
            data["retrieval_time_ms"],
            data["generation_time_ms"],
            data["total_time_ms"],
            data["confidence"],
        )
        return answer_md, citations_md, latency_md
    except httpx.HTTPStatusError as exc:
        return f"**API Error {exc.response.status_code}:** {exc.response.text}", "", ""
    except httpx.RequestError:
        return (
            f"**Connection Error:** Could not reach API at `{API_BASE}`. Is the server running? (`make run`)",
            "",
            "",
        )
    except Exception as exc:  # noqa: BLE001
        return f"**Unexpected Error:** {exc}", "", ""


async def handle_health() -> str:
    """Gradio event handler: poll /health and return formatted markdown."""
    try:
        health = await health_api()
        return format_health(health)
    except Exception as exc:  # noqa: BLE001
        return f"**Health Check Failed:** {exc}"


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(title="PaperLens", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# PaperLens — Research Intelligence over arXiv ML Papers")
        gr.Markdown(
            "Ask a question about ML/AI papers. The system retrieves relevant chunks, "
            "generates a cited answer via Ollama (`phi4-mini` primary, `llama3.2:1b` fallback), "
            "and shows source provenance."
        )

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Question",
                    placeholder="e.g. How do recent papers approach learning rate scheduling?",
                    lines=2,
                )
                with gr.Row():
                    top_k = gr.Slider(
                        1, 20, value=settings.retrieval_top_k, step=1, label="Top-K chunks"
                    )
                    model = gr.Dropdown(
                        choices=[settings.ollama_model, "llama3.2:1b"],
                        value=settings.ollama_model,
                        label="Ollama Model",
                    )
                submit = gr.Button("Query", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### System Health")
                health_md = gr.Markdown("Loading…")
                refresh_health = gr.Button("Refresh Health", size="sm")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Answer")
                answer_md = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Citations (click to expand)")
                citations_md = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Latency Breakdown")
                latency_md = gr.Markdown("")

        # Event wiring
        submit.click(
            handle_query,
            inputs=[question, top_k, model],
            outputs=[answer_md, citations_md, latency_md],
        )
        question.submit(
            handle_query,
            inputs=[question, top_k, model],
            outputs=[answer_md, citations_md, latency_md],
        )
        refresh_health.click(handle_health, inputs=None, outputs=health_md)
        demo.load(handle_health, inputs=None, outputs=health_md)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=settings.gradio_port, show_error=True)

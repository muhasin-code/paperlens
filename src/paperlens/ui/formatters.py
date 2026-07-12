"""Formatters for PaperLens Gradio UI: citations, latency, health."""

import typing as t


def format_citations(citations: list[dict[str, t.Any]] | None) -> str:
    """
    Format a list of citation dictionaries into a clean, professional markdown representation.
    Each citation dictionary has the structure:
    {
        'chunk_id': str,
        'arxiv_id': str,
        'title': str,
        'authors': list[str] | str,
        'section_label': str,
        'page_start': int,
        'page_end': int,
        'score': float,
        'rank': int
    }
    """
    if not citations:
        return "*No sources cited.*"

    md_lines = []
    # Sort by rank or score
    sorted_cites = sorted(citations, key=lambda x: x.get("rank", 999))

    for item in sorted_cites:
        chunk_id = item.get("chunk_id", "N/A")
        arxiv_id = item.get("arxiv_id", "")
        title = item.get("title", "Untitled Paper")
        authors = item.get("authors", "Unknown Authors")
        if isinstance(authors, list):
            authors = ", ".join(authors)
        section = item.get("section_label", "Main")
        p_start = item.get("page_start", 0)
        p_end = item.get("page_end", 0)
        score = item.get("score", 0.0)

        arxiv_link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "#"
        page_str = f"p. {p_start}" if p_start == p_end else f"pp. {p_start}-{p_end}"
        if not p_start and not p_end:
            page_str = "N/A"

        md_lines.append(
            f"### [{chunk_id}] {title}\n"
            f"- **Authors:** {authors}\n"
            f"- **arXiv:** [{arxiv_id}]({arxiv_link}) | **Section:** *{section}* | **Pages:** {page_str}\n"
            f"- **Retrieval Score:** `{score:.3f}`\n"
        )

    return "\n".join(md_lines)


def format_latency(
    retrieval_time_ms: int | float | None,
    generation_time_ms: int | float | None,
    total_time_ms: int | float | None,
) -> str:
    """
    Formats the retrieval and generation latency metrics into a clean markdown badge
    with a visual waterfall representation.
    """
    ret = retrieval_time_ms or 0.0
    gen = generation_time_ms or 0.0
    tot = total_time_ms or (ret + gen)

    # Calculate percentages for waterfall layout
    ret_pct = (ret / tot * 100) if tot > 0 else 0
    gen_pct = (gen / tot * 100) if tot > 0 else 0

    # Simple ASCII waterfall bar
    bar_width = 20
    ret_chars = int(round((ret_pct / 100) * bar_width))
    gen_chars = bar_width - ret_chars

    waterfall_bar = "\u2588" * ret_chars + "\u2591" * gen_chars

    md_output = (
        f"\u23f1\ufe0f **Latency Waterfall:** `{tot:.1f}ms` total\n"
        f"  * \U0001f50d **Retrieval:** `{ret:.1f}ms` ({ret_pct:.1f}%)\n"
        f"  * \U0001f9e0 **Generation:** `{gen:.1f}ms` ({gen_pct:.1f}%)\n"
        f"  * `[{waterfall_bar}]` (\u2588 Retrieval | \u2591 Generation)"
    )
    return md_output


def format_health(health_data: dict[str, t.Any] | None) -> tuple[str, str]:
    """
    Formats the system health dictionary into (status_markdown, status_color).
    health_data structure:
    {
        'status': str,
        'chroma_collection': str,
        'chroma_vector_count': int,
        'ollama_reachable': bool,
        'ollama_model': str,
        'ollama_fallback_model': str
    }

    Returns:
        (status_text, css_color_class_or_color)
    """
    if not health_data:
        return "\U0001f534 **Offline** (Backend Unreachable)", "red"

    status = health_data.get("status", "error")
    vector_count = health_data.get("chroma_vector_count", 0)
    ollama_reachable = health_data.get("ollama_reachable", False)
    ollama_model = health_data.get("ollama_model", "N/A")
    ollama_fallback = health_data.get("ollama_fallback_model", "")

    if status == "ok" and ollama_reachable:
        status_text = f"\U0001f7e2 **Healthy** | \U0001f4ca {vector_count:,} Vectors | \U0001f9e0 Model: `{ollama_model}`"
        color = "green"
    elif status == "ok" or ollama_reachable:
        model_info = f"`{ollama_model}`" if ollama_reachable else f"Fallback `{ollama_fallback}`"
        status_text = f"\U0001f7e1 **Degraded** | \U0001f4ca {vector_count:,} Vectors | \U0001f9e0 Model: {model_info}"
        color = "amber"
    else:
        status_text = "\U0001f534 **Unhealthy** | Connection or Ollama failed"
        color = "red"

    return status_text, color

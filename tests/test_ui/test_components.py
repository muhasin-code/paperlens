"""Tests for PaperLens UI component structure."""

import gradio as gr

from src.paperlens.ui.app import build_ui


def test_ui_structure_and_components():
    """
    Test that the gr.Blocks interface initializes successfully
    and contains all the essential elements defined in our layout.
    """
    demo = build_ui()

    # Assert it is indeed a gr.Blocks app
    assert isinstance(demo, gr.Blocks)

    # Traverse children components to check for existence of required modules
    all_components = []

    def traverse(comp):
        all_components.append(comp)
        if hasattr(comp, "children"):
            for child in comp.children:
                traverse(child)

    traverse(demo)

    # Extract names/types of elements we want to guarantee are in the app
    has_chatbot = any(isinstance(c, gr.Chatbot) for c in all_components)
    has_textbox = any(isinstance(c, gr.Textbox) for c in all_components)
    has_slider = any(isinstance(c, gr.Slider) for c in all_components)
    has_radio = any(isinstance(c, gr.Radio) for c in all_components)
    has_button = any(isinstance(c, gr.Button) for c in all_components)
    has_html = any(isinstance(c, gr.HTML) for c in all_components)

    assert has_chatbot, "Gradio App is missing gr.Chatbot component"
    assert has_textbox, "Gradio App is missing gr.Textbox input component"
    assert has_slider, "Gradio App is missing gr.Slider component for Top-K"
    assert has_radio, "Gradio App is missing gr.Radio model selection component"
    assert has_button, "Gradio App is missing interactive submit or query trigger buttons"
    assert has_html, "Gradio App is missing styling/health gr.HTML component"

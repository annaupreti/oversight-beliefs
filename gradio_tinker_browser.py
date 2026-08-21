#!/usr/bin/env python3
"""Password-protected Tinker chat and qualitative-probe browser.

The Tinker API key remains only in the server environment. Do not publish an
unauthenticated Gradio share link: visitors could otherwise spend your Tinker
inference credits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import gradio as gr
import tinker

from build_mentor_portal import MODEL_A, MODEL_B, ROOT, read_records


MODELS = {MODEL_A["short_label"]: MODEL_A, MODEL_B["short_label"]: MODEL_B}
PROBES = ROOT / "artifacts" / "tinker_probes"


def clean_control_tokens(text: str) -> str:
    return re.sub(r"<\|[^>]+\|>", "", text).strip()


def split_completion(raw: str, clean: str) -> tuple[str, str]:
    reasoning_match = re.search(r"<\|channel\|>analysis(?:<\|message\|>)?(.*?)(?=<\|channel\|>final|\Z)", raw, re.DOTALL)
    final_match = re.search(r"<\|channel\|>final(?:<\|message\|>)?(.*)", raw, re.DOTALL)
    reasoning = clean_control_tokens(reasoning_match.group(1)) if reasoning_match else ""
    answer = clean_control_tokens(final_match.group(1)) if final_match else clean
    return reasoning, answer


class TinkerSampler:
    def __init__(self) -> None:
        self.service = tinker.ServiceClient(user_metadata={"experiment": "contrastive-sdf-browser"})
        self.clients: dict[str, Any] = {}
        self.tokenizers: dict[str, Any] = {}

    def get(self, model: dict[str, str]) -> tuple[Any, Any]:
        key = model["sampler"]
        if key not in self.clients:
            client = self.service.create_sampling_client(model_path=key)
            self.clients[key] = client
            self.tokenizers[key] = client.get_tokenizer()
        return self.clients[key], self.tokenizers[key]

    def chat(self, model: dict[str, str], messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int) -> tuple[str, str, str, str]:
        client, tokenizer = self.get(model)
        rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        ids = getattr(rendered, "input_ids", None)
        if ids is None:
            try:
                ids = rendered["input_ids"]
            except (KeyError, TypeError):
                ids = rendered
        response = client.sample(
            tinker.ModelInput.from_ints(list(ids)),
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed),
        ).result()
        sequence = response.sequences[0]
        raw = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
        clean = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        reasoning, answer = split_completion(raw, clean)
        return answer, reasoning, raw, str(sequence.stop_reason)


SAMPLER: TinkerSampler | None = None


def get_sampler() -> TinkerSampler:
    global SAMPLER
    if SAMPLER is None:
        SAMPLER = TinkerSampler()
    return SAMPLER


def load_records() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(PROBES.glob("*/responses.jsonl")):
        rows.extend(read_records(path))
    return rows


RECORDS = load_records()
RECORD_OPTIONS = {
    f"{record['model']['short_label']} | {record['task_id']} | {Path(record['source_file']).parent.name}:{record['source_line']}": record
    for record in RECORDS
}


def send(message: str, history: list[dict[str, str]], model_name: str, temperature: float, max_tokens: int, seed: int):
    history = history or []
    if not message.strip():
        return history, "", "", "", ""
    messages = [*history, {"role": "user", "content": message}]
    answer, reasoning, raw, stop_reason = get_sampler().chat(MODELS[model_name], messages, temperature, int(max_tokens), int(seed))
    updated = [*history, {"role": "user", "content": message}, {"role": "assistant", "content": answer}]
    return updated, "", reasoning or "(No analysis channel captured.)", raw, f"Stop reason: `{stop_reason}`"


def clear_chat():
    return [], "", "", "", ""


def show_record(option: str):
    record = RECORD_OPTIONS[option]
    messages = "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in record["messages"])
    metadata = (
        f"**Model:** {record['model']['label']}\n\n"
        f"**SDF direction:** {record['model']['sdf_direction']}\n\n"
        f"**Source:** `{record['source_file']}:{record['source_line']}`  \n"
        f"**Category:** `{record['category']}` · **style:** `{record.get('style_label')}` · **stop:** `{record.get('stop_reason')}`"
    )
    return metadata, messages, record.get("answer") or "", record.get("reasoning") or "", record.get("raw_completion") or ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Contrastive SDF model browser", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Contrastive SDF model browser\nChat with the two Tinker samplers and inspect every saved qualitative completion. The server holds the Tinker key; the browser never receives it.")
        with gr.Tab("Chat with models"):
            with gr.Row():
                model = gr.Dropdown(list(MODELS), value=MODEL_A["short_label"], label="SDF model")
                temperature = gr.Slider(0, 1.2, value=0, step=0.1, label="Temperature")
                max_tokens = gr.Slider(128, 4096, value=1200, step=128, label="Maximum generated tokens")
                seed = gr.Number(value=0, precision=0, label="Seed")
            # Gradio 6's Chatbot uses message dictionaries by default.
            chatbot = gr.Chatbot(height=460, label="Conversation")
            with gr.Row():
                message = gr.Textbox(label="Message", placeholder="Ask either model a question…", scale=5)
                send_button = gr.Button("Send", variant="primary", scale=1)
                clear_button = gr.Button("Clear", scale=1)
            stop = gr.Markdown()
            with gr.Accordion("Last completion reasoning", open=False):
                reasoning = gr.Textbox(lines=14, label="Reasoning", interactive=False)
            with gr.Accordion("Raw completion", open=False):
                raw = gr.Textbox(lines=12, label="Raw", interactive=False)
            inputs = [message, chatbot, model, temperature, max_tokens, seed]
            outputs = [chatbot, message, reasoning, raw, stop]
            send_button.click(send, inputs, outputs)
            message.submit(send, inputs, outputs)
            clear_button.click(clear_chat, outputs=[chatbot, message, reasoning, raw, stop])
        with gr.Tab("Saved completions"):
            gr.Markdown(f"{len(RECORD_OPTIONS)} saved records are currently indexed. Select one to inspect its exact prompt, final answer, reasoning, and raw output.")
            record_picker = gr.Dropdown(list(RECORD_OPTIONS), label="Saved record", filterable=True)
            metadata = gr.Markdown()
            prompt = gr.Textbox(lines=8, label="Prompt/messages", interactive=False)
            answer = gr.Textbox(lines=12, label="Final answer", interactive=False)
            reasoning_saved = gr.Textbox(lines=16, label="Reasoning trace", interactive=False)
            raw_saved = gr.Textbox(lines=10, label="Raw completion", interactive=False)
            record_picker.change(show_record, record_picker, [metadata, prompt, answer, reasoning_saved, raw_saved])
        with gr.Tab("Evidence and charts"):
            gr.Markdown("## Current interpretation\nThe initial paired probe showed crossed default style behavior. The v2 source-visible versus outcome-only manipulation did **not** yield the predicted interaction, so it is a negative result for the current visibility hypothesis—not evidence of reward-seeking.")
            with gr.Row():
                gr.Image(str(ROOT / "artifacts/mentor_update/graphs/initial_paired_style_difference.png"), label="Initial crossed style difference")
                gr.Image(str(ROOT / "artifacts/mentor_update/graphs/v2_visibility_result.png"), label="v2 visibility result")
            gr.Markdown("Read the complete analysis in `artifacts/mentor_update/combined_mentor_update.md`.")
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio public URL.")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("Set TINKER_API_KEY in the server environment before launching.")
    auth = None
    if args.share:
        username, password = os.environ.get("GRADIO_USERNAME"), os.environ.get("GRADIO_PASSWORD")
        if not username or not password:
            raise RuntimeError("For --share, set GRADIO_USERNAME and GRADIO_PASSWORD to protect your Tinker credits.")
        auth = (username, password)
    build_ui().launch(server_name="0.0.0.0", server_port=args.port, share=args.share, auth=auth)


if __name__ == "__main__":
    main()

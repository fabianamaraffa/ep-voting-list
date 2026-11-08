"""Gradio app for Hugging Face Spaces (ZeroGPU).

Set EP_USE_STUBS=1 to run the interface with fake models (no downloads, no GPU).
"""

from __future__ import annotations

# `spaces` must be imported before torch (and before anything that could import torch),
# otherwise ZeroGPU can fail to attach a GPU. Keep this as the first import.
try:
    import spaces
except ImportError:  # not running on Hugging Face
    spaces = None

import os

import gradio as gr
import pandas as pd

from ep_voting.export import COLUMNS, write_excel
from ep_voting.llm import PREFILL
from ep_voting.pipeline import run_pipeline

# ------------------------------------------------------------------------------
# Configuration (change these as Space variables, no code edit needed)
# ------------------------------------------------------------------------------
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
USE_STUBS = os.getenv("EP_USE_STUBS") == "1"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "24"))  # amendments per GPU call
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))  # prompts generated at once on the GPU
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))
GPU_SECONDS = int(os.getenv("GPU_SECONDS", "110"))  # time limit of one GPU call

if spaces is not None:
    gpu = spaces.GPU
else:  # elsewhere the decorator does nothing

    def gpu(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn


# ------------------------------------------------------------------------------
# Models are loaded once, at start-up, outside any GPU-decorated function.
# On ZeroGPU, `.to("cuda")` at module level is allowed: the GPU is attached only
# while a function decorated with @spaces.GPU is running.
# ------------------------------------------------------------------------------
if USE_STUBS:
    from ep_voting.stubs import StubEmbedder, stub_generate

    embedder = StubEmbedder()
    _generate_chunk = stub_generate
else:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[INFO] Loading embedding model {EMBED_MODEL_ID} (CPU)")
    embedder = SentenceTransformer(EMBED_MODEL_ID, device="cpu")

    print(f"[INFO] Loading language model {LLM_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    tokenizer.padding_side = "left"  # required for batched generation with decoder models
    llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, dtype=torch.bfloat16)
    if torch.cuda.is_available():
        llm.to("cuda")

    @gpu(duration=GPU_SECONDS)
    def _generate_chunk(prompts: list[list[dict]]) -> list[str]:
        """Run the language model on a list of chat prompts and return the continuations."""
        outputs: list[str] = []
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i : i + BATCH_SIZE]
            texts = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) + PREFILL
                for m in batch
            ]
            enc = tokenizer(texts, return_tensors="pt", padding=True).to(llm.device)
            with torch.inference_mode():
                generated = llm.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs


# ------------------------------------------------------------------------------
# Gradio callbacks
# ------------------------------------------------------------------------------
DEFAULT_PRIORITIES = """Safeguard a dedicated budget civil society organisations and keep NGOs able to access grants directly and simply.
Oppose transfers that divert social cohesion resources."""


def analyse(pdf_file, priorities: str, threshold: float, max_items: int, progress=gr.Progress()):
    path = pdf_file if isinstance(pdf_file, str) or pdf_file is None else getattr(pdf_file, "name", str(pdf_file))
    out = run_pipeline(
        path,
        priorities,
        float(threshold),
        int(max_items),
        embedder=embedder,
        generate=_generate_chunk,
        chunk_size=CHUNK_SIZE,
        progress=lambda frac, desc: progress(frac, desc=desc),
    )
    return out.table, out.status


def export_table(df: pd.DataFrame | None):
    if df is None or len(df) == 0:
        raise gr.Error("There is nothing to export yet. Run the analysis first.")
    return write_excel(pd.DataFrame(df, columns=COLUMNS))


PRIVACY_NOTE = (
    "Hosted demo: your PDF and priorities are processed on Hugging Face servers. "
    "For full privacy, run this app on your own computer (see the README)."
    if os.getenv("SPACE_ID")
    else "Running locally: your files and priorities stay on this computer."
)

THEME = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1280px !important; margin: auto; }
.hero h1 { font-size: 1.9rem; margin: 0 0 .25rem 0; }
.hero p { margin: .15rem 0; color: var(--body-text-color-subdued); font-size: 1rem; }
.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; margin: .75rem 0 1rem 0; }
.step { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: .7rem .9rem;
        background: var(--background-fill-secondary); font-size: .92rem; line-height: 1.35; }
.step b { display: block; margin-bottom: .15rem; }
.step .n { display: inline-block; width: 1.5rem; height: 1.5rem; line-height: 1.5rem; text-align: center;
           border-radius: 50%; background: #1f4e79; color: white; font-size: .8rem; margin-right: .4rem; }
.panel-title { font-weight: 600; font-size: 1.05rem; margin: .2rem 0 .1rem 0; }
.legend { display: flex; flex-wrap: wrap; gap: .5rem; margin: .25rem 0 .5rem 0; font-size: .88rem; }
.chip { padding: .15rem .6rem; border-radius: 999px; border: 1px solid var(--border-color-primary); }
.chip.plus { background: #e3f4e8; color: #1b6b35; }
.chip.minus { background: #fbe4e4; color: #9b1c1c; }
.chip.split { background: #fff3d6; color: #8a5a00; }
.chip.abst { background: #e8ecf1; color: #3d4a5c; }
.chip.review { background: #f3e5f7; color: #6b2a80; }
.detail { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: .8rem 1rem;
          background: var(--background-fill-secondary); }
.privacy { font-size: .85rem; color: var(--body-text-color-subdued); }
@media (max-width: 800px) { .steps { grid-template-columns: 1fr 1fr; } }
"""

HERO = f"""
<div class="hero">
<h1>EP Voting List Generator</h1>
<p>Turn a European Parliament amendments PDF into a draft voting list, based on your own priorities.</p>
<p class="privacy">{PRIVACY_NOTE}</p>
</div>
<div class="steps">
<div class="step"><b><span class="n">1</span>Upload</b>The committee's amendments PDF. Every amendment is read exactly as printed.</div>
<div class="step"><b><span class="n">2</span>Say what matters</b>Write your priorities and red lines, one per line.</div>
<div class="step"><b><span class="n">3</span>Analyse</b>Irrelevant amendments are filtered out, then an AI model suggests a vote for the rest.</div>
<div class="step"><b><span class="n">4</span>Check and export</b>Click a row to read the full text, edit any cell, then download Excel.</div>
</div>
"""

LEGEND = """
<div class="legend">
<span class="chip plus"><b>+</b> support</span>
<span class="chip minus"><b>-</b> oppose</span>
<span class="chip split"><b>SPLIT</b> two separable changes, vote separately</span>
<span class="chip abst"><b>ABSTAIN</b> unclear effect</span>
<span class="chip review"><b>REVIEW</b> the model failed, decide yourself</span>
</div>
"""

COLUMN_WIDTHS = [110, 170, 190, 80, 420, 320, 320, 100, 90]
EMPTY_DETAIL = "*Click any row of the table to read the full text of that amendment here.*"


def show_detail(df: pd.DataFrame, evt: gr.SelectData):
    """Show the full text of the selected row below the table."""
    try:
        row = df.iloc[evt.index[0]]
    except Exception:
        return EMPTY_DETAIL
    current = str(row["Current text"]) or "(none)"
    return (
        f"### {row['Amendment']}  |  suggested vote: **{row['Vote']}**\n"
        f"*{row['Authors']}*  \n{row['Location']}\n\n"
        f"**Why:** {row['Rationale']}\n\n"
        f"**Current text**\n\n> {current}\n\n"
        f"**Amendment text**\n\n> {row['Amendment text']}"
    )


with gr.Blocks(title="EP Voting List Generator") as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            gr.Markdown("<div class='panel-title'>1. Your document</div>")
            pdf_input = gr.File(label="Amendments PDF (EP committee document)", file_types=[".pdf"])
            gr.Markdown("<div class='panel-title'>2. Your priorities</div>")
            priorities_input = gr.Textbox(
                label="Priorities and red lines",
                info="One per line. Short, specific sentences work best.",
                lines=7,
                value=DEFAULT_PRIORITIES,
            )
            gr.Markdown("<div class='panel-title'>3. How many to analyse</div>")
            threshold = gr.Slider(
                0.05, 0.60, value=0.25, step=0.01,
                label="Relevance filter",
                info="Slide right to keep fewer, more closely related amendments. "
                "Slide left to keep more. Around 0.25 is a good start.",
            )
            max_items = gr.Slider(
                5, 150, value=40, step=5,
                label="Maximum amendments",
                info="Upper limit after filtering. More amendments take longer and use more GPU time.",
            )
            run_btn = gr.Button("Analyse amendments", variant="primary", size="lg")
        with gr.Column(scale=8, min_width=480):
            status = gr.Markdown("Waiting for a document.")
            gr.Markdown("<div class='panel-title'>Suggested votes</div>")
            gr.HTML(LEGEND)
            table = gr.Dataframe(
                headers=COLUMNS,
                label="Voting list (cells can be edited)",
                interactive=True,
                wrap=False,
                max_height=460,
                column_widths=COLUMN_WIDTHS,
                show_row_numbers=False,
            )
            gr.Markdown("<div class='panel-title'>Selected amendment</div>")
            detail = gr.Markdown(EMPTY_DETAIL)
            with gr.Row():
                export_btn = gr.Button("Export to Excel (.xlsx)", variant="secondary")
                download = gr.File(label="Excel file (appears after export)", height=90)

    run_btn.click(analyse, [pdf_input, priorities_input, threshold, max_items], [table, status])
    table.select(show_detail, [table], [detail])
    export_btn.click(export_table, [table], [download])

if __name__ == "__main__":
    demo.queue().launch(
        theme=THEME,
        css=CSS,
    )

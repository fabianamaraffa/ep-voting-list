---
title: EP Voting List Generator
**[Try the live demo on Hugging Face](https://huggingface.co/spaces/Fabianamaraffa/ep-voting-list)**
emoji: 🗳️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.10.13"
app_file: app.py
pinned: false
---

# EP Voting List Generator

Upload the amendments PDF of a European Parliament committee, write your organisation's
priorities and red lines, and get a draft **voting list**: for each relevant amendment the
authors, the location in the text, the current and amended wording, a recommended vote
(`+`, `-`, `SPLIT`, `ABSTAIN`) and a one-sentence rationale. The table can be edited in the
browser and exported to Excel.

> The output is a draft to support a decision. A person must check every recommendation.
> Rows the model could not handle are marked `REVIEW` instead of getting a guessed vote.

## How it works

```
PDF ──> parser ──> screening ──> language model ──> table ──> Excel
        (code)     (embeddings)   (GPU, ZeroGPU)     (pandas)
```

1. **Parse** (`ep_voting/parser.py`). Reads the PDF with PyMuPDF and extracts every amendment
   exactly as printed: number, authors, document type, location, current text, amended text,
   justification and original language. No AI is involved, so these fields cannot be altered.
   Words printed in bold italic (the EP convention for changes) are wrapped in `**` so the
   model can see what changed.
2. **Screen** (`ep_voting/screening.py`). A small embedding model (`all-MiniLM-L6-v2`, CPU)
   scores each amendment against each priority line and keeps the best match. Amendments
   below the threshold are dropped, and the rest are capped at the "maximum" slider, so GPU
   time is only spent where it matters.
3. **Evaluate** (`ep_voting/llm.py`, `app.py`). A language model (default
   `Qwen/Qwen2.5-7B-Instruct`) gets the priorities plus one amendment and returns only a vote
   and a rationale as JSON. The code tells the model exactly what the amendment removes and
   adds (read from the bold-italic marks of the EP document), and asks for the rationale
   before the vote so the model reasons first. The reply is forced to start as JSON, and
   anything unreadable becomes `REVIEW` with the start of the raw answer shown. Work is sent to the GPU in chunks, so a quota error halfway keeps the
   results already obtained.
4. **Table and export** (`ep_voting/export.py`). One row per amendment in voting order, with
   the `**` markers removed, then a formatted `.xlsx`.

### Files

| File | Role |
| --- | --- |
| `app.py` | Gradio interface, model loading, the `@spaces.GPU` function |
| `ep_voting/parser.py` | PDF to structured amendments (state machine over text lines with coordinates) |
| `ep_voting/screening.py` | Priority splitting, embedding scores, threshold and cap |
| `ep_voting/llm.py` | Prompt, safe parsing of model output, chunked evaluation |
| `ep_voting/pipeline.py` | The whole workflow as one function, independent of Gradio |
| `ep_voting/export.py` | Table building and Excel writing |
| `ep_voting/schemas.py` | Pydantic models |
| `ep_voting/stubs.py` | Fake models for tests (`EP_USE_STUBS=1`) |
| `tests/` | Unit tests, a synthetic EP-style PDF, and an end-to-end Gradio test |

### Why the parser reads coordinates

EP amendments are two-column tables and often continue across a page break; the heading
"Amendment 86" can even sit at the bottom of one page with its authors at the top of the next.
Flattening the text and splitting with a regex mixes the columns and splits on in-text
mentions such as "see Amendment 5". Here each line is read with its position: the left
column is the current text, the right column the amendment, and a small state machine
(`idle`, `header`, `table`, `justification`) carries over from page to page until the
`Or. xx` end marker. The cover page states the range ("AMENDMENTS 1 - 317"), and the app
warns if any number is missing.

## Hosted demo and private local use

The hosted demo on Hugging Face Spaces is the full application: same code, same models,
nothing cut down. Anyone can open the link and use it.

Because it runs on Hugging Face servers, the uploaded PDF and the priorities typed into the
box are processed there. Amendments are public documents, but an organisation's priorities
and red lines may not be. Anyone who wants more privacy can run exactly the same app on
their own computer, where nothing leaves the machine.

### Run it locally (full version, private)

```bash
git clone <this-repository>
cd <this-repository>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt gradio==6.31.0
python app.py
```

Open http://127.0.0.1:7860. Needs Python 3.10 or newer.

* The first start downloads the models (several GB for the default 7B model). After that,
  set `HF_HUB_OFFLINE=1` to run with no network connection at all.
* **With an NVIDIA GPU** (about 16 GB of memory for the 7B model) it runs as on the Space.
* **Without a GPU** it still works but is slow. Use a smaller model and keep the "Maximum
  amendments" slider low:
  `LLM_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct python app.py`
  (Windows PowerShell: `$env:LLM_MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"; python app.py`).
* The `spaces` package is not needed locally, the GPU decorator is skipped.

Settings are the same environment variables listed in the table below.

### For developers: quick interface check without models

`EP_USE_STUBS=1 python app.py` starts the app with fake models (no download, no GPU). PDF
reading, screening, table and Excel export are real, but the votes come from placeholder
rules, not AI. This is what the automated tests use. It is not meant for real work.

## Deploy on Hugging Face Spaces

1. Create a Space: SDK **Gradio**, any name. Hardware: **ZeroGPU** (free accounts can host
   up to 2 such Spaces) or a paid GPU. In my own test ZeroGPU failed at start-up with
   `No CUDA GPUs are available`, while a dedicated **Nvidia L4 (24 GB)** worked at once, so
   the L4 is the tested configuration. On paid hardware, set a short *sleep time* (for
   example 15 minutes) and use *Settings > Pause Space* when done: billing is per minute
   while the Space is Starting or Running.
2. Put these files in the Space repository, either by dragging them into
   *Files > Add file > Upload files*, or with git:
   ```bash
   git clone https://huggingface.co/spaces/<user>/<space-name>
   cp -r <this-folder>/* <this-folder>/.gitignore <space-name>/
   cd <space-name> && git add . && git commit -m "Add app" && git push
   ```
3. Wait for the build. The first start downloads the models and can take several minutes.

### GPU quota

ZeroGPU time is charged to the person who uses the app, per day: about 2 minutes for visitors
who are not logged in, 5 minutes for free accounts, 40 minutes for PRO accounts. A full run
on the 7B model can use a good part of a free quota, so for free accounts lower the
"Maximum amendments" slider (10 to 20) and, if runs get cut off, set `CHUNK_SIZE=12` and
`GPU_SECONDS=60`. If the quota runs out halfway, the app keeps the rows already analysed and
marks the others `REVIEW`. Alternatively, pick a paid GPU for the Space (for example an L4
with 24 GB) and set a sleep time so it only costs money while in use.

### Settings (Space *Settings > Variables*)

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | Language model. Use `Qwen/Qwen2.5-3B-Instruct` if the Space runs out of memory or GPU quota. |
| `EMBED_MODEL_ID` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for screening |
| `CHUNK_SIZE` | `24` | Amendments per GPU call |
| `BATCH_SIZE` | `8` | Prompts generated at once inside a GPU call |
| `MAX_NEW_TOKENS` | `160` | Length limit of each answer |
| `GPU_SECONDS` | `110` | Time limit of one GPU call |

Model size is a trade-off. "7B" means about 7 billion parameters: larger models follow the
JSON format and judge policy text better, but are slower and use more of the daily ZeroGPU
quota. A 1.5B model is fast but gives noticeably weaker recommendations.

## License

MIT, see `LICENSE`.

## Run the tests

```bash
pip install -r requirements.txt pytest reportlab gradio gradio_client
pytest                                    # synthetic PDF, fake models
EP_SAMPLE_PDF=path/to/amendments.pdf pytest   # also tests a real document
```

## What was tested, honestly

Tested on a real document (EMPL committee, ESF+ post-2027, 317 amendments) with
`Qwen/Qwen2.5-7B-Instruct` on an L4 GPU, 40 amendments per run, three successive runs while
the prompt was being improved.

| What | Result |
| --- | --- |
| PDF parsing | 317 of 317 amendments found, including one whose header and authors are split across pages. No AI involved. |
| Screening | Works as intended: the amendments kept are the ones that touch the priorities. |
| Run 1 | 34 of 40 votes were `SPLIT`. The model voted before reasoning and could not tell what the amendment changed. |
| Run 2 | After computing "removed / added" from the EP bold-italic marks in code and asking for the rationale before the vote, votes became varied. Some had a rationale that contradicted the vote or reversed the direction of the change. |
| Run 3 | Vote words (`SUPPORT`, `OPPOSE`, ...) instead of symbols, and an `ABSTAIN` rule for speculative effects. Clear cases are good; borderline cases are still unstable. |

Conclusions:

* Extraction, screening and export are reliable. The vote recommendation from a 7B model is
  a **first draft**: good on clear-cut amendments, weak on subtle ones. Every row needs a
  human check, which is why the app labels it a draft and never invents a vote (`REVIEW`).
* I did not measure agreement against a human's own votes, so no accuracy figure is claimed.
  The next step would be to vote on 20 amendments by hand, compare, and try a larger model
  (for example `Qwen/Qwen2.5-14B-Instruct`, about 28 GB, needs a 48 GB GPU) through
  `LLM_MODEL_ID`.
* A frontier model such as Claude will usually reason better on one document. This tool's
  value is different: a repeatable pipeline, exact text extraction, and an option to run
  fully private on your own machine.

## Known limits

* Built for the standard English EP amendment layout. Other languages work if the layout is
  the same. Scanned PDFs without a text layer are not supported.
* Relevance screening uses an English embedding model.
* Votes depend on the model and on how clearly the priorities are written. Specific, one-line
  priorities give the best results.

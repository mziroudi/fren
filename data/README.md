# Data

Built by `python3 data/build_dataset.py`. The committed JSONL files are the dataset of record.

| File | Task | Inputs | Source | License |
|---|---|---|---|---|
| `code_explain.jsonl` | Task A: explain a chunk of Python code | 80 | CPython 3.11 standard library (`Lib/*.py`, files shipped with the local Anaconda Python). One chunk per module, starting at the first top-level `def`/`class`, up to 2,400 characters. Modules picked in a seeded random order (seed 1234). | PSF License 2.0 |
| `news_summary.jsonl` | Task B: summarize a news article | 80 | English Wikinews, `Category:Published`, newest first (the site-closure announcement skipped), plain text from the MediaWiki TextExtracts API, trimmed to 2,400 characters. Each row links its article. | CC BY 2.5 (Wikinews contributors) |

Each row: `task`, `input_id`, `source`, `prompt` (the user message, before the chat template).

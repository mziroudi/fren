"""Build the two week-1 datasets.

Task A (code_explain): explain a chunk of a Python standard-library module.
  Source: the CPython standard library shipped with the local Python (PSF License).
Task B (news_summary): summarize a news article.
  Source: English Wikinews, Category:Published (CC BY 2.5), via the MediaWiki API.

Writes data/code_explain.jsonl and data/news_summary.jsonl with fields:
  task, input_id, source, prompt

The committed JSONL files are the dataset of record. Re-running this script needs network
access for task B and the same Python version for task A to give identical files.
"""
import json
import random
import sys
import sysconfig
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

N_PER_TASK = 80
MAX_CHARS = 2400  # keeps prompts around 500-700 tokens
SEED = 1234
OUT = Path(__file__).resolve().parent

CODE_PROMPT = (
    "Explain what the following Python code does. Describe the purpose of each function "
    "or class and how they work together.\n\n```python\n{code}\n```"
)
NEWS_PROMPT = (
    "Summarize the following news article in one short paragraph. Mention who, what, "
    "when and where.\n\n{text}"
)


def code_chunks():
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in stdlib.glob("*.py") if not p.name.startswith("_"))
    rng = random.Random(SEED)
    rng.shuffle(files)
    rows = []
    for path in files:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        # start at the first top-level def/class, skipping license headers and docstrings
        start = next(
            (i for i, l in enumerate(lines) if l.startswith(("def ", "class "))), None
        )
        if start is None:
            continue
        chunk, size = [], 0
        for line in lines[start:]:
            if size + len(line) + 1 > MAX_CHARS:
                break
            chunk.append(line)
            size += len(line) + 1
        code = "\n".join(chunk).rstrip()
        if len(code) < 1200:  # too short to be a meaningful input
            continue
        rows.append(
            {
                "task": "code_explain",
                "input_id": len(rows),
                "source": f"cpython-{sys.version_info.major}.{sys.version_info.minor}/Lib/{path.name}",
                "prompt": CODE_PROMPT.format(code=code),
            }
        )
        if len(rows) == N_PER_TASK:
            break
    return rows


def api(params):
    url = "https://en.wikinews.org/w/api.php?" + urllib.parse.urlencode(
        {"format": "json", **params}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "fren-dataset-builder/0.1"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(5 * 2**attempt)  # rate limited: back off
    raise RuntimeError("Wikinews API kept rate-limiting")


def news_articles():
    titles, cont = [], {}
    while len(titles) < N_PER_TASK * 2:
        d = api(
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": "Category:Published",
                "cmtype": "page",
                "cmsort": "timestamp",
                "cmdir": "desc",
                "cmlimit": 100,
                **cont,
            }
        )
        titles += [m["title"] for m in d["query"]["categorymembers"]]
        cont = d.get("continue", {})
        if not cont:
            break
    titles = titles[1:]  # skip the site-closure announcement, it is not a normal article
    rows = []
    for title in titles:
        d = api(
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": title,
            }
        )
        page = next(iter(d["query"]["pages"].values()))
        text = page.get("extract", "")
        # drop the trailing Sources / Related news sections
        for marker in ("\n\n\n== Sources ==", "\n\n\n== Related news ==", "\n\n\n== Sister links =="):
            text = text.split(marker)[0]
        text = text.strip()
        if len(text) < 800:
            continue
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS].rsplit("\n", 1)[0].strip()
        rows.append(
            {
                "task": "news_summary",
                "input_id": len(rows),
                "source": "https://en.wikinews.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                "prompt": NEWS_PROMPT.format(text=f"Title: {title}\n\n{text}"),
            }
        )
        if len(rows) == N_PER_TASK:
            break
        time.sleep(1.5)
    return rows


def write(rows, name):
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} inputs")


if __name__ == "__main__":
    write(code_chunks(), "code_explain.jsonl")
    write(news_articles(), "news_summary.jsonl")

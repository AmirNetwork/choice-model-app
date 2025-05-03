#!/usr/bin/env python3
"""
generate_training_dataset.py

Generate high-quality instruction–response pairs by calling Ollama,
filter them, and write them out as JSONL for fine-tuning.

Each reference JSON in ./references must contain:
  - title
  - paper_text
  - code_snippet

Output is written to ./training_data/dataset_<timestamp>.jsonl
with lines of the form:
{"prompt": "...", "completion": "..."}
"""

import os
import json
import hashlib
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

# === Configuration ===
REFS_DIR     = Path("references")     # where your ref_*.json files live
OUTPUT_DIR   = Path("training_data")  # where to save dataset_*.jsonl
OLLAMA_MODEL = "gpt-3.5-turbo"        # Ollama model name
N_GEN        = 5                      # how many pairs per reference
MIN_Q_LEN    = 10                     # min instruction length
MIN_A_LEN    = 20                     # min response length

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# === Utility functions ===

def load_references(refs_dir: Path):
    """Load all JSON reference files."""
    refs = []
    for path in sorted(refs_dir.glob("*.json")):
        try:
            refs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning(f"Failed to load {path.name}: {e}")
    return refs

def build_prompt_header(ref: dict) -> str:
    """
    Create the static part of the prompt (up to the instruction placeholder).
    We'll later substitute the actual instruction into it.
    """
    title = ref.get("title", "<no title>")
    paper = ref.get("paper_text", "").strip()
    code  = ref.get("code_snippet", "").strip() or "# (no code snippet provided)"

    return (
        f"### Task: {title}\n"
        "### Context ###\n"
        "```python\n"
        f"{code}\n"
        "```\n"
        "### Instruction ###\n"
        # placeholder:
        "{instruction}\n"
        "### Response ###\n"
        "```python\n"
    )

def call_ollama(full_prompt: str) -> str:
    """
    Pipe the full prompt to Ollama and capture stdout.
    """
    proc = subprocess.Popen(
        ["ollama", "run", OLLAMA_MODEL],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = proc.communicate(full_prompt)
    if proc.returncode != 0:
        raise RuntimeError(f"Ollama error (code {proc.returncode}): {err.strip()}")
    return out.strip()

def parse_json_or_fallback(raw: str) -> list:
    """
    Try to parse raw as a JSON array; otherwise split lines on first ':'.
    """
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return arr
    except json.JSONDecodeError:
        pass

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if ":" in ln:
            q, a = ln.split(":", 1)
            out.append({"instruction": q.strip(), "response": a.strip()})
    return out

def quality_filter(pairs: list) -> list:
    """
    Remove too-short or duplicate pairs.
    """
    seen = set()
    good = []
    for p in pairs:
        q = p.get("instruction","")
        a = p.get("response","")
        if len(q) < MIN_Q_LEN or len(a) < MIN_A_LEN:
            continue
        key = hashlib.md5((q + a).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        good.append(p)
    return good

def save_jsonl(entries: list, out_dir: Path):
    """
    Write entries (dicts with 'prompt' & 'completion') to a timestamped JSONL.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = out_dir / f"dataset_{ts}.jsonl"
    with open(fpath, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(entries)} examples to {fpath}")

def main():
    logger.info("Loading references…")
    refs = load_references(REFS_DIR)
    logger.info(f"Found {len(refs)} reference file(s).")

    all_entries = []
    for ref in refs:
        title = ref.get("title","<untitled>")
        logger.info(f"→ Generating for: {title}")

        header = build_prompt_header(ref)
        ask_json = (
            f"Please output a JSON array of {N_GEN} objects,
each with keys \"instruction\" and \"response\".\n"
        )

        try:
            raw = call_ollama(header.replace("{instruction}", ask_json))
        except Exception as e:
            logger.error(f"Ollama failed on {title}: {e}")
            continue

        candidates = parse_json_or_fallback(raw)
        logger.info(f"  - Raw candidates: {len(candidates)}")
        filtered = quality_filter(candidates)
        logger.info(f"  - After filtering: {len(filtered)}")

        for p in filtered:
            inst = p["instruction"]
            resp = p["response"]
            prompt = header.replace("{instruction}", inst)
            all_entries.append({"prompt": prompt, "completion": resp})

        time.sleep(1)  # avoid rapid-fire calls

    if all_entries:
        save_jsonl(all_entries, OUTPUT_DIR)
    else:
        logger.warning("No examples generated; nothing to save.")

if __name__ == "__main__":
    main()

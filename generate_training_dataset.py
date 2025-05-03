#!/usr/bin/env python3
"""
generate_training_dataset.py
────────────────────────────
Create high-quality instruction–response pairs for fine-tuning GPT-Neo (or
any LLM) on your choice-modelling repository – especially the *ResLogit* code.

Key ideas
──────────
*   Reads reference JSON files stored in storage/references/.
    Each JSON looks like:
        {
          "title": "Reslogit",
          "paper_text": "...",       # optional
          "code_snippet": "import …" # REQUIRED (full source!)
        }

*   For every reference it fabricates N_GEN diverse tasks that mirror the
    examples you supplied (training, cross-validation, plotting …).

*   If USE_MODEL == True it calls a local LLM (DeepSeek via Ollama, GPT-4, …)
    to auto-generate the *response* portion.  
    The prompt format is **identical** to your working samples.

*   All pairs are written to ./training_data/dataset_<timestamp>.jsonl
    – ready for HuggingFace fine-tuning.

The script is Windows-robust: it reads the model’s stdout/stderr as **bytes**
and decodes with UTF-8, so CP-1252 consoles no longer explode with
UnicodeDecodeError (0x8F, 0x90 …).

Requires Python ≥3.8.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ── Configuration ────────────────────────────────────────────────────
REFS_DIR   = Path("storage/references")        # where *.json live
OUTPUT_DIR = Path("training_data")             # output folder
N_GEN      = 8                                 # tasks per reference

USE_MODEL  = True                              # False ⇒ leave completions blank
MODEL_CMD  = ["ollama", "run", "deepseek-r1:7b"]
TIMEOUT_SEC = 420                              # seconds for each model call
SLEEP_SEC   = 1.0                              # pause between calls

MIN_Q_LEN  = 10                                # basic filtering
MIN_A_LEN  = 20
# ──────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Task templates (mirrors your examples) ───────────────────────────
TASK_TEMPLATES: list[tuple[str, bool]] = [
    ("Show how to plot histogram of residual layer outputs from {class_name} on random input.", True),
    ("Provide code: load `df`, train {class_name}, and plot the training loss.", False),
    ("Wrap {class_name} training into a function `train_{lname}_{i}(df)` that returns model and history.", False),
    ("Instantiate a {class_name} model with random hyperparameters and print summary.", False),
    ("Show how to save and reload a trained {class_name} model.", False),
    ("Add early stopping callback to {class_name} training and print number of epochs run.", False),
    ("Perform 4-fold cross-validation with {class_name} on `df` and report average loss.", False),
    ("Generate code to print all trainable weight names and shapes after instantiating {class_name}.", False),
]


# ── Helpers ──────────────────────────────────────────────────────────
def pick_tasks(class_name: str) -> List[str]:
    """Return N_GEN unique instructions for a given class."""
    out, seen = [], set()
    i = 0
    while len(out) < N_GEN:
        tpl, _ = TASK_TEMPLATES[i % len(TASK_TEMPLATES)]
        instr = tpl.format(class_name=class_name,
                           lname=class_name.lower(),
                           i=len(out))
        if instr not in seen:
            out.append(instr)
            seen.add(instr)
        i += 1
    return out


def build_prompt(title: str, code: str, instruction: str) -> str:
    """Exact prompt format that matches your manual examples."""
    return textwrap.dedent(f"""\
        ### Task: {title}
        ### Context ###
        ```python
        {code.strip()}
        ```
        ### Instruction ###
        {instruction}
        ### Response ###
        ```python
        """)


# ── Unicode-safe subprocess call ────────────────────────────────────
def call_model(prompt: str) -> str:
    """
    Send *prompt* to the local CLI LLM (DeepSeek via Ollama, etc.)
    and return its raw textual answer (decoded as UTF-8, errors→�).

    * No `text=True`: we work with BYTES to dodge locale issues.
    * Timeout and errors are bubbled up to caller.
    """
    proc = subprocess.Popen(
        MODEL_CMD,
        stdin=subprocess.PIPE,  # binary pipes
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout_b, stderr_b = proc.communicate(
            prompt.encode("utf-8"), timeout=TIMEOUT_SEC
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("LLM call timed out")

    if proc.returncode != 0:
        msg = stderr_b.decode("utf-8", "replace") or f"Model exited {proc.returncode}"
        raise RuntimeError(msg)

    return stdout_b.decode("utf-8", "replace").strip()


# ── Reference loading ────────────────────────────────────────────────
def load_references() -> List[Dict]:
    refs: list[Dict] = []
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    for fp in sorted(REFS_DIR.glob("*.json")):
        try:
            refs.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            log.warning("Skip %s – %s", fp.name, e)
    return refs


# ── Main workflow ────────────────────────────────────────────────────
def main() -> None:
    refs = load_references()
    if not refs:
        log.error("No reference JSON files found in %s", REFS_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"dataset_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    written = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for ref in refs:
            title = ref.get("title", "Untitled")
            code  = ref.get("code_snippet", "").strip()
            if not code:
                log.warning("Reference '%s' has no code_snippet – skipped", title)
                continue

            # crude class-name extraction
            m = re.search(r"class\s+(\w+)\(", code)
            class_name = m.group(1) if m else title

            for instr in pick_tasks(class_name):
                prompt = build_prompt(title, code, instr)
                if USE_MODEL:
                    try:
                        raw_out = call_model(prompt)
                        # keep only what follows the first ```python block
                        response = raw_out.split("```python", 1)[-1].rstrip("` \n")
                    except Exception as e:  # noqa: BLE001
                        log.error("LLM failed: %s", e)
                        continue
                else:
                    response = ""  # you can fill in later by hand

                # basic filtering
                if len(instr) < MIN_Q_LEN or len(response) < MIN_A_LEN:
                    continue

                fout.write(json.dumps({"prompt": prompt, "completion": response},
                                      ensure_ascii=False) + "\n")
                written += 1
                time.sleep(SLEEP_SEC)

    log.info("Wrote %d samples → %s", written, out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
data_generation_general.py

Generates JSONL training data for GPT‑Neo fine‑tuning so it can learn to
produce runnable Python code for fitting arbitrary models described by
reference JSONs. Each prompt now always includes the reference code snippet
(if available), ensuring DeepSeek sees it when generating instruction–response pairs.
"""

import os
import json
import hashlib
import time
import traceback
import re
import subprocess
from datetime import datetime
from pathlib import Path

# === Configuration ===
REFS_DIR       = Path("C:/choice-model-app/storage/references")
OUTPUT_DIR     = Path("training_data")
DEEPSEEK_MODEL = "deepseek-r1:7b"
OLLAMA_PATH    = r"C:/Users/amirg/AppData/Local/Programs/Ollama"
TIMEOUT_SECS   = 300
MIN_Q_LEN      = 10
MIN_A_LEN      = 80

PHRASE_TEMPLATES = [
    "Write Python code to instantiate and fit a {model} on the pandas DataFrame `df` and print the learned parameters.",
    "Given DataFrame `df`, generate runnable Python that sets up a {model}, trains it, and returns the trained model.",
    "Produce only Python code that loads `df`, defines a {model}, fits it, and displays the model summary.",
    "Using the reference below, write complete Python to train a {model} on `df` and output its coefficients.",
    "Generate a Python script that takes `df`, builds a {model}, fits it for 100 epochs, and prints the results."
]

# prepare environment
ENV = os.environ.copy()
ENV["PATH"] = OLLAMA_PATH + os.pathsep + ENV.get("PATH", "")

def warm_up():
    print("Warming up DeepSeek model…", end="", flush=True)
    try:
        subprocess.run(
            ["ollama", "run", DEEPSEEK_MODEL],
            input="print('warmup')",
            text=True,
            env=ENV,
            timeout=TIMEOUT_SECS//2,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except:
        pass
    print(" done.\n")

def deepseek_call(prompt: str) -> str:
    try:
        proc = subprocess.Popen(
            ["ollama", "run", DEEPSEEK_MODEL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=ENV
        )
        out, err = proc.communicate(input=prompt+"\n", timeout=TIMEOUT_SECS)
        if proc.returncode != 0:
            return f"Error: {err.strip() or 'exit code '+str(proc.returncode)}"
        return re.sub(r'\x1b\[[0-9;]*m','', out).strip()
    except subprocess.TimeoutExpired:
        proc.kill()
        return f"Error: timed out after {TIMEOUT_SECS}s"
    except FileNotFoundError:
        return "Error: Ollama CLI not found on PATH"
    except Exception as e:
        return f"Unexpected error: {e}\n{traceback.format_exc()}"

def load_references(refs_dir: Path):
    refs_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for file in sorted(refs_dir.glob("*.json")):
        try:
            refs.append(json.loads(file.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"Warning: failed to load {file.name}: {e}")
    return refs

def build_prompts(ref: dict):
    title = ref.get("title","<untitled>").strip()
    desc  = (ref.get("paper_text","") or "").replace("\n"," ").strip()[:300]
    code  = ref.get("code_snippet","").strip()
    prompts = []
    for template in PHRASE_TEMPLATES:
        instr = template.format(model=title)
        lines = [
            "You are a Python expert in choice modelling.",
            "Generate *only* runnable Python code (no prose).",
            "Assume the user's pandas DataFrame is named `df`.",
            f"Reference Model: {title}",
            f"Description: {desc}",
        ]
        # always include the code snippet if available
        if code:
            lines.append("Reference Code Snippet:\n```python\n" + code + "\n```")
        lines += [
            "Wrap your code in triple backticks ```python ... ```.",
            "Return a JSON array of objects with keys 'instruction' and 'response'."
        ]
        prompt = "\n".join(lines)
        prompts.append((prompt, instr))
    return prompts

def parse_candidates(raw: str):
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return arr
    except:
        pass
    blocks = re.findall(r"```python\s*(.*?)```", raw, re.S)
    return [{"instruction":"Generate code","response":"```python\n"+b.strip()+"\n```"} for b in blocks]

def quality_filter(pairs):
    seen, out = set(), []
    for p in pairs:
        q = p.get("instruction","").strip()
        a = p.get("response","").strip()
        if len(q)<MIN_Q_LEN or len(a)<MIN_A_LEN:
            continue
        key = hashlib.md5((q+a).encode("utf-8")).hexdigest()
        if key in seen: continue
        seen.add(key)
        out.append({"instruction":q,"response":a})
    return out

def save_jsonl(records, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"dataset_{ts}.jsonl"
    with path.open("w",encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path

def main():
    start = time.time()
    warm_up()

    refs = load_references(REFS_DIR)
    print(f"Loaded {len(refs)} references in {time.time()-start:.2f}s\n")

    all_samples = []
    for ref in refs:
        for prompt, instr in build_prompts(ref):
            print(f"→ Prompting: {instr}")
            t1 = time.time()
            raw = deepseek_call(prompt)
            print(f"  DeepSeek call: {time.time()-t1:.2f}s, output len={len(raw)}")

            cand = parse_candidates(raw)
            print(f"  Parsed {len(cand)} candidate(s)")

            filt = quality_filter(cand)
            print(f"  Kept {len(filt)} after filter\n")

            for ex in filt:
                ex["instruction"] = instr
            all_samples.extend(filt)
            time.sleep(1)

    if not all_samples:
        print("No samples generated; exiting.")
        return

    out_path = save_jsonl(all_samples, OUTPUT_DIR)
    print(f"Saved {len(all_samples)} records to {out_path}")
    print(f"Total run time: {time.time()-start:.2f}s")

if __name__=="__main__":
    main()

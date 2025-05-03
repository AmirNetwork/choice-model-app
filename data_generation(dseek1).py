#!/usr/bin/env python3
"""
generate_training_dataset.py

Process reference JSONs into a HuggingFace Dataset and optional JSONL of
prompt/completion pairs for GPT-Neo fine-tuning. Uses embedding-based retrieval
and DeepSeek for generation, with robust decoding and quality checks.
"""
import os
import json
import re
import ast
import hashlib
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import faiss
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import argparse

# --- Configuration ---
BASE_DIR = Path(__file__).parent
REFS_DIR = BASE_DIR / "storage" / "references"
OUT_DIR = BASE_DIR / "storage" / "training_dataset"
EXPORT_JSONL = True  # also export JSONL for inspection

EMBED_MODEL = "microsoft/codebert-base"
TOP_K = 2
MIN_SNIPPET_LEN = 80  # minimum characters per code snippet

INSTRUCTION_TEMPLATES = [
    "Instantiate a {title} model on a DataFrame `df` and print a summary.",
    "Wrap `{title}` training into a function `train_model(df)` that returns model and history.",
    "Generate a script: load `df`, define {title}, train it, and plot predictions.",
    "Explain each constructor parameter of {title} in code comments, then fit on `df`.",
]

OLLAMA_MODEL = "deepseek-r1:7b"
OLLAMA_CMD = ["ollama", "run", OLLAMA_MODEL]
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def call_deepseek(prompt: str, timeout: int = 300) -> str:
    """Invoke DeepSeek via Ollama CLI, decode UTF-8 ignoring errors."""
    try:
        proc = subprocess.Popen(
            OLLAMA_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out_bytes, err_bytes = proc.communicate(input=prompt.encode('utf-8'), timeout=timeout)
        if proc.returncode != 0:
            err = err_bytes.decode('utf-8', errors='ignore').strip()
            logging.warning(f"DeepSeek exit {proc.returncode}: {err}")
            return ""
        return re.sub(r"\x1b\[[0-9;]*m", "", out_bytes.decode('utf-8', errors='ignore')).strip()
    except subprocess.TimeoutExpired:
        proc.kill()
        logging.error("DeepSeek timeout")
        return ""
    except Exception as e:
        logging.exception("DeepSeek exception")
        return ""


def parse_blocks(raw: str):
    return re.findall(r"```python\s*(.*?)```", raw, flags=re.S)


def quality_check(code: str) -> bool:
    if len(code.strip()) < MIN_SNIPPET_LEN:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def load_references():
    for path in sorted(REFS_DIR.glob("ref_*.json")):
        data = json.loads(path.read_text(encoding='utf-8'))
        yield path.stem, data


def chunk_reference(ref: dict, min_chars: int = 100):
    combined = (ref.get('paper_text','') + '\n\n' + ref.get('code_snippet','')).strip()
    parts = [blk.strip() for blk in combined.split('\n\n') if len(blk.strip()) >= min_chars]
    return parts


# cache the embedder to avoid repeated loading per reference
_EMBEDDER = None
def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        logging.info(f"Loading embedding model '{EMBED_MODEL}'")
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


def build_index(chunks):
    embedder = get_embedder()
    vecs = embedder.encode(chunks, convert_to_numpy=True)
    dim = vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    return embedder, index


def generate_examples_for_ref(title: str, ref: dict):
    chunks = chunk_reference(ref)
    if not chunks:
        return []
    embedder, index = build_index(chunks)
    examples = []
    for tmpl in INSTRUCTION_TEMPLATES:
        instr = tmpl.format(title=title)
        qv = embedder.encode([instr], convert_to_numpy=True)
        _, ids = index.search(qv, TOP_K)
        context = "\n\n".join(chunks[i] for i in ids[0])
        prompt = (
            "### Reference Code ###\n```python\n" + context + "\n```\n\n"
            f"### Instruction ###\n{instr}\n\n"
            "### Response ###\n```python\n"
        )
        raw = call_deepseek(prompt)
        for blk in parse_blocks(raw):
            code = blk.strip()
            if quality_check(code):
                examples.append({"prompt": prompt, "completion": code})
    return examples


def save_jsonl(examples, path: Path):
    logging.info(f"Writing JSONL to {path}")
    with path.open('w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')


def main(export_jsonl: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_ex = []
    for stem, ref in load_references():
        title = ref.get('title', stem)
        logging.info(f"Processing '{title}'")
        exs = generate_examples_for_ref(title, ref)
        logging.info(f" → {len(exs)} examples generated")
        all_ex.extend(exs)

    if not all_ex:
        logging.warning("No examples generated. Exiting.")
        return
    # save as HuggingFace dataset
    ds = Dataset.from_list(all_ex)
    ds_path = OUT_DIR / 'rag_training_dataset'
    ds.save_to_disk(str(ds_path))
    logging.info(f"Dataset saved ({len(all_ex)} examples) to {ds_path}")

    if export_jsonl:
        jl_path = OUT_DIR / 'rag_training_dataset.jsonl'
        save_jsonl(all_ex, jl_path)
        logging.info(f"Also exported JSONL for human inspection: {jl_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-jsonl', action='store_true', help='Skip JSONL export')
    args = parser.parse_args()
    main(export_jsonl=not args.no_jsonl)

#!/usr/bin/env python3
# app_llama_adalora.py
# ─────────────────────────────────────────────────────────────
"""
Flask chatbot demo

• user signup / reference upload
• on-the-fly instruction-tuning of Llama-2-7B via **AdaLoRA** (PEFT)
• **REALM-style dense retrieval** over user-uploaded refs
  – Sentence-Transformers (MPNet) embeddings
  – FAISS inner-product index with async refresh
• second-pass answer polish with DeepSeek (via Ollama)

Quick install
─────────────
pip install flask pandas torch --upgrade
pip install transformers accelerate peft datasets
pip install sentence-transformers faiss-cpu --upgrade
pip install --upgrade "huggingface_hub[cli]"
#  (install Ollama separately for DeepSeek)
"""
# ─── stdlib ──────────────────────────────────────────────────
import os, json, hashlib, subprocess, re, time, traceback, threading
from datetime import datetime
from functools import wraps

# ─── third-party ─────────────────────────────────────────────
import pandas as pd
import torch
from flask import Flask, request, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments)
from peft import AdaLoraConfig, get_peft_model
from datasets import Dataset

from sentence_transformers import SentenceTransformer
import faiss, numpy as np

# ═════════════════════════════════════════════════════════════
#  CONSTANTS  &  FOLDERS
# ═════════════════════════════════════════════════════════════
BASE_DIR          = os.path.abspath(os.path.dirname(__file__))
STORAGE_DIR       = os.path.join(BASE_DIR, "storage")
UPLOAD_FOLDER     = os.path.join(STORAGE_DIR, "uploads")
LOG_FOLDER        = os.path.join(STORAGE_DIR, "logs")
REFERENCES_FOLDER = os.path.join(STORAGE_DIR, "references")
CACHE_DIR         = os.path.join(BASE_DIR, "hf_cache")
USER_DB_FILE      = os.path.join(STORAGE_DIR, "users.json")

for p in (UPLOAD_FOLDER, LOG_FOLDER, REFERENCES_FOLDER, CACHE_DIR):
    os.makedirs(p, exist_ok=True)

ALLOWED_EXT       = {"csv", "xlsx"}
LLAMA_MODEL_NAME  = "meta-llama/Llama-2-7b-hf"   # accept licence on HF
MAX_INPUT_TOKENS  = 4096 - 512   # head-room for generation
MAX_NEW_TOKENS    = 256
LLM_TIMEOUT_SECS  = 680

# DeepSeek via Ollama ----------------------------------------
OLLAMA_PATH = r"C:\Users\<YOU>\AppData\Local\Programs\Ollama"
DEEPSEEK_CMD = "ollama run deepseek-r1:7b"
os.environ["PATH"] = OLLAMA_PATH + os.pathsep + os.environ.get("PATH", "")

# ═════════════════════════════════════════════════════════════
#  FLASK  APP
# ═════════════════════════════════════════════════════════════
app            = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
user_data: dict = {}   # per-session RAM cache only

# ═════════════════════════════════════════════════════════════
#  MINI  HELPERS
# ═════════════════════════════════════════════════════════════
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf8")).hexdigest()

def load_users() -> dict:
    if not os.path.exists(USER_DB_FILE):
        json.dump({}, open(USER_DB_FILE, "w", encoding="utf8"))
    return json.load(open(USER_DB_FILE, encoding="utf8"))

def save_users(d: dict):
    json.dump(d, open(USER_DB_FILE, "w", encoding="utf8"), indent=2)

def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[-1].lower() in ALLOWED_EXT

def log_line(txt: str):
    with open(os.path.join(LOG_FOLDER, "chat.log"), "a", encoding="utf8") as f:
        f.write(txt + "\n")

# ═════════════════════════════════════════════════════════════
#  DEEPSEEK second-pass
# ═════════════════════════════════════════════════════════════
def deepseek(prompt: str) -> str:
    """Call DeepSeek 7B through Ollama CLI for refinement."""
    try:
        proc = subprocess.Popen(DEEPSEEK_CMD,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True, shell=True)
        out, err = proc.communicate(prompt + "\n", timeout=LLM_TIMEOUT_SECS)
        if proc.returncode != 0:
            return f"[DeepSeek error] {err.strip()}"
        return re.sub(r'\x1b\\[[0-9;]*m', '', out).strip()
    except subprocess.TimeoutExpired:
        return "[DeepSeek] timeout"
    except Exception as e:
        return f"[DeepSeek exception] {e}"

# ═════════════════════════════════════════════════════════════
#  LLAMA-2  +  AdaLoRA
# ═════════════════════════════════════════════════════════════
g_tok   = None   # tokenizer
g_model = None   # fine-tunable PEFT model

def init_llama_adalora() -> None:
    """Load base Llama-2 and wrap it with an AdaLoRA adapter."""
    global g_tok, g_model
    if g_model is not None:
        return

    device_map  = "auto"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # tokenizer
    g_tok = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME,
                                          cache_dir=CACHE_DIR, use_fast=True)
    g_tok.pad_token = g_tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME, cache_dir=CACHE_DIR,
        device_map=device_map, torch_dtype=torch_dtype
    )

    ada_cfg = AdaLoraConfig(
        r=8, target_r=2, init_r=8,
        tinit=0, tfinal=8000, delta_t=10,
        beta1=0.85, beta2=0.85, orth_reg_weight=0.5,
        lora_alpha=32, lora_dropout=0.05,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ]
    )
    g_model = get_peft_model(base, ada_cfg)
    g_model.print_trainable_parameters()

# ═════════════════════════════════════════════════════════════
#  GENERATION
# ═════════════════════════════════════════════════════════════
def llama_generate(prompt: str,
                   temperature: float = 0.7,
                   max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    if g_model is None:
        return "Model not initialised."

    enc = g_tok(prompt, return_tensors="pt",
                truncation=True, max_length=MAX_INPUT_TOKENS).to(g_model.device)

    with torch.no_grad():
        out_ids = g_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=50,
            pad_token_id=g_tok.eos_token_id
        )
    full = g_tok.decode(out_ids[0], skip_special_tokens=True)
    return full[len(prompt):].strip()

# ═════════════════════════════════════════════════════════════
#  ONE-SHOT  ADA-LoRA  FINE-TUNE
# ═════════════════════════════════════════════════════════════
def fine_tune_once(user_id: str, ref_text: str):
    if g_model is None:
        init_llama_adalora()

    ds = Dataset.from_dict({
        "text": [f"### Instruction\n{ref_text}\n### Response\n"]
    })

    def tokenize(batch):
        tok = g_tok(batch["text"], truncation=True,
                    max_length=512, padding="max_length")
        tok["labels"] = tok["input_ids"].copy()
        return tok

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=os.path.join(STORAGE_DIR, "adalora_ft"),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1, optim="adamw_torch"
    )
    Trainer(model=g_model, train_dataset=ds_tok, args=args).train()

    g_model.base_model.save_pretrained(
        os.path.join(STORAGE_DIR, f"adalora_user_{user_id}_{int(time.time())}")
    )

# ═════════════════════════════════════════════════════════════
#  REALM-STYLE  DENSE  RETRIEVER
# ═════════════════════════════════════════════════════════════
EMB_MODEL_NAME      = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
g_emb_model         = SentenceTransformer(EMB_MODEL_NAME)
EMB_DIM             = g_emb_model.get_sentence_embedding_dimension()

faiss_index         = None        # IndexFlatIP
ref_store: list[dict] = []        # mirrors FAISS order
index_lock          = threading.Lock()

def _encode(txts: list[str]) -> np.ndarray:
    vec = g_emb_model.encode(txts, convert_to_numpy=True, show_progress_bar=False)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-10
    return vec.astype("float32")

def load_refs():
    res = []
    for f in sorted(os.listdir(REFERENCES_FOLDER)):
        if f.endswith(".json"):
            res.append(json.load(open(os.path.join(REFERENCES_FOLDER, f),
                                      encoding="utf8")))
    return res

def _rebuild_faiss():
    """Initial load of FAISS index from disk refs."""
    global faiss_index, ref_store
    with index_lock:
        ref_store = load_refs()
        faiss_index = faiss.IndexFlatIP(EMB_DIM)
        if not ref_store:
            return
        vecs = _encode([r.get("paper_text","") + " " + r.get("code_snippet","")
                        for r in ref_store])
        faiss_index.add(vecs)

def _add_ref_to_index(ref: dict):
    """Embed & append a single ref to FAISS (async)."""
    global faiss_index
    vec = _encode([ref.get("paper_text","") + " " + ref.get("code_snippet","")])
    with index_lock:
        faiss_index.add(vec)
        ref_store.append(ref)

def retrieve_refs(q: str, k: int = 3):
    """Dense max-inner-product retrieval."""
    if faiss_index is None or faiss_index.ntotal == 0:
        return []
    vec_q = _encode([q])
    with index_lock:
        sims, idxs = faiss_index.search(vec_q, min(k, faiss_index.ntotal))
    hits = []
    for score, idx in zip(sims[0], idxs[0]):
        if idx == -1:
            continue
        ref = ref_store[idx].copy()
        ref["score"] = float(score)
        hits.append(ref)
    return hits

# build FAISS once at startup
_rebuild_faiss()

# ═════════════════════════════════════════════════════════════
#  CHAT  PIPELINE
# ═════════════════════════════════════════════════════════════
def produce_answer(user_msg: str, user_id: str) -> str:
    ctx = ""
    for r in retrieve_refs(user_msg):
        ctx += (f"# REF {r.get('title','')} (sim={r['score']:.3f})\n"
                f"{r.get('paper_text','')}\n")
        if r.get("code_snippet"):
            ctx += f"```python\n{r['code_snippet']}\n```\n"

    prompt = (f"{ctx}\n### Instruction\n{user_msg}\n"
              "### Response (include runnable Python in ```python```):\n")

    llama_out = llama_generate(prompt)
    refined   = deepseek(f"Refine the following answer and code:\n\n{llama_out}")

    return f"[Llama-AdaLoRA]\n{llama_out}\n\n[DeepSeek]\n{refined}"

# ═════════════════════════════════════════════════════════════
#  AUTH  DECORATOR  &  ROUTES
# ═════════════════════════════════════════════════════════════
def login_required(fn):
    @wraps(fn)
    def wrap(*a, **kw):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return wrap

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        u, pw = request.form["username"].lower(), request.form["password"]
        users = load_users()
        if u in users:
            return render_template("signup3.html", error="User exists.")
        users[u] = {"user_id": str(len(users)+1), "pw": hash_pw(pw)}
        save_users(users)
        return redirect(url_for("login"))
    return render_template("signup3.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u, pw = request.form["username"].lower(), request.form["password"]
        users = load_users(); info = users.get(u)
        if info and info["pw"] == hash_pw(pw):
            session["user_id"], session["username"] = info["user_id"], u
            return redirect(url_for("app_home"))
        return render_template("login3.html", error="Bad credentials")
    return render_template("login3.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/", methods=["GET"])
def root(): return redirect(url_for("login"))

@app.route("/app", methods=["GET","POST"])
@login_required
def app_home():
    uid = session["user_id"]
    feedback = file_msg = None
    udata = user_data.setdefault(uid, {"history":[]})

    if request.method == "POST":
        if request.form.get("action") == "send":
            msg = request.form.get("msg","").strip()
            if msg:
                feedback = produce_answer(msg, uid)
                udata["history"].extend([("user", msg), ("bot", feedback)])
                log_line(f"{datetime.now()} {uid} {msg} >> {feedback}")

        elif request.form.get("action") == "upload_ref":
            title = request.form.get("ref_title","").strip()
            paper = request.form.get("ref_paper","").strip()
            code  = request.form.get("ref_code","").strip()
            if title or paper or code:
                fn = f"ref_{int(time.time())}.json"
                path = os.path.join(REFERENCES_FOLDER, fn)
                ref  = {"title":title,"paper_text":paper,"code_snippet":code}
                json.dump(ref, open(path,"w",encoding="utf8"), indent=2)
                file_msg = "Reference saved • quick fine-tune running…"
                fine_tune_once(uid, paper + "\n" + code)
                # async add to FAISS
                threading.Thread(target=_add_ref_to_index, args=(ref,),
                                 daemon=True).start()

    return render_template("index3.html",
                           chat=udata["history"],
                           feedback=feedback,
                           file_feedback=file_msg,
                           username=session.get("username",""))

# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[Init] Loading Llama-2 & attaching AdaLoRA…")
    init_llama_adalora()
    print("[Init] FAISS index size:", faiss_index.ntotal)
    print("[Ready] http://127.0.0.1:5000")
    app.run("127.0.0.1", 5000, debug=False)

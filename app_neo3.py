#!/usr/bin/env python3
# app_neo3.py
# -*- coding: utf-8 -*-
#
# Multi-step chatbot pipeline with:
#  - User login (signup, login, logout)
#  - Data and reference uploads for immediate incremental LoRA fine-tuning (GPT-Neo)
#  - A streamlined generation process:
#      1. GPT-Neo generates a domain-specific answer (including a Python code snippet)
#      2. DeepSeek refines that answer
#
# Dependencies: flask, transformers, peft, datasets, pandas, torch, subprocess, re, traceback, time
#
# IMPORTANT: When a reference is uploaded, fine-tuning is triggered immediately for that user.
# When starting the application, you will be prompted whether to run a global auto fine-tuning
# over all references (which you can choose to skip).  

import os
import json
import traceback
import re
import subprocess
import time
from datetime import datetime
import hashlib
from functools import wraps

import pandas as pd
import torch
from flask import Flask, request, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename

# Transformers + LoRA
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

###############################################################################
# Configuration & Constants
###############################################################################
BASE_DIR                 = os.path.abspath(os.path.dirname(__file__))
STORAGE_DIR              = os.path.join(BASE_DIR, "storage")
UPLOAD_FOLDER            = os.path.join(STORAGE_DIR, "uploads")
LOG_FOLDER               = os.path.join(STORAGE_DIR, "logs")
COMPLETED_MODELS_FOLDER  = os.path.join(STORAGE_DIR, "completed_models")
REFERENCES_FOLDER        = os.path.join(STORAGE_DIR, "references")
TRAINING_DATA_FOLDER     = os.path.join(STORAGE_DIR, "training_data")
USER_DB_FILE             = os.path.join(STORAGE_DIR, "users.json")

# Ensure storage directories exist
for folder in [UPLOAD_FOLDER, LOG_FOLDER, COMPLETED_MODELS_FOLDER, REFERENCES_FOLDER, TRAINING_DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS   = {"csv", "xlsx"}
CACHE_DIR           = "./hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

GPTNEO_MODEL_NAME   = "EleutherAI/gpt-neo-1.3B"
LLM_TIMEOUT_SECS    = 680
MAX_INPUT_TOKENS    = 1792
MAX_NEW_TOKENS      = 256

# Ollama path for DeepSeek
OLLAMA_PATH         = r"C:\Users\amirg\AppData\Local\Programs\Ollama"
DEEPSEEK_CMD        = "ollama run deepseek-r1:7b"
os.environ["PATH"]  = OLLAMA_PATH + os.pathsep + os.environ.get("PATH", "")

###############################################################################
# Flask App Setup
###############################################################################
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# In-memory user/session data
user_data = {}

###############################################################################
# Helpers: user management, logging, data preview
###############################################################################
def load_users():
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(USER_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users_dict):
    with open(USER_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(users_dict, f, indent=4)

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def log_interaction(user_msg: str, bot_msg: str, user_id=None):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(LOG_FOLDER, "chat.log"), "a", encoding="utf-8") as f:
        f.write(f"{stamp} - User({user_id}): {user_msg} | Bot: {bot_msg}\n")

def get_data_preview(user_id: str, num_rows=20) -> str:
    udata = user_data.get(user_id, {})
    df = udata.get("data_info", {}).get("df")
    if isinstance(df, pd.DataFrame):
        return df.head(num_rows).to_csv(index=False)
    return ""

###############################################################################
# DeepSeek integration
###############################################################################
def deepseek_call(prompt_text: str) -> str:
    try:
        proc = subprocess.Popen(
            DEEPSEEK_CMD,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, shell=True, env=os.environ
        )
        stdout, stderr = proc.communicate(input=prompt_text + "\n", timeout=LLM_TIMEOUT_SECS)
        if proc.returncode != 0:
            err = stderr.strip() or f"Error code {proc.returncode}"
            return f"Error calling DeepSeek: {err}"
        # strip ANSI
        return re.sub(r'\x1b\[[0-9;]*m', '', stdout).strip()
    except subprocess.TimeoutExpired:
        return f"Error: DeepSeek timed out after {LLM_TIMEOUT_SECS}s"
    except Exception as e:
        return f"DeepSeek exception: {e}"

###############################################################################
# GPT-Neo + LoRA setup
###############################################################################
g_neo_tokenizer = None
g_neo_model     = None

def initialize_gpt_neo_model():
    global g_neo_tokenizer, g_neo_model
    g_neo_tokenizer = AutoTokenizer.from_pretrained(GPTNEO_MODEL_NAME, cache_dir=CACHE_DIR)
    if g_neo_tokenizer.pad_token is None:
        g_neo_tokenizer.pad_token = g_neo_tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        GPTNEO_MODEL_NAME, cache_dir=CACHE_DIR,
        device_map="auto", torch_dtype=torch.float16
    )
    lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    g_neo_model = get_peft_model(base_model, lora_config)
    if g_neo_tokenizer.pad_token_id is None:
        g_neo_model.resize_token_embeddings(len(g_neo_tokenizer))

def generate_gpt_neo_text(prompt: str, temperature=0.7) -> str:
    global g_neo_model, g_neo_tokenizer
    if g_neo_model is None or g_neo_tokenizer is None:
        return "Error: GPT-Neo not initialized."
    enc = g_neo_tokenizer.encode(prompt, add_special_tokens=False)
    if len(enc) > MAX_INPUT_TOKENS:
        enc = enc[-MAX_INPUT_TOKENS:]
    inputs = torch.tensor([enc], dtype=torch.long).to(g_neo_model.device)
    with torch.no_grad():
        out_ids = g_neo_model.generate(
            input_ids=inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=temperature,
            top_k=50, pad_token_id=g_neo_tokenizer.eos_token_id
        )
    text = g_neo_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    # strip the prompt part
    return text[len(g_neo_tokenizer.decode(enc, skip_special_tokens=True)):].strip()

###############################################################################
# Reference retrieval and RAG pipeline
###############################################################################
def load_references():
    refs = []
    for f in sorted(os.listdir(REFERENCES_FOLDER)):
        if f.endswith(".json"):
            try:
                with open(os.path.join(REFERENCES_FOLDER, f), "r", encoding="utf-8") as rf:
                    refs.append(json.load(rf))
            except:
                pass
    return refs

def retrieve_refs(query: str):
    all_refs = load_references()
    hits = [r for r in all_refs if r.get("title","").lower() in query.lower()]
    return hits if hits else all_refs[:1]

def multi_step_generation(user_msg: str, user_id: str) -> str:
    # build reference context
    refs = retrieve_refs(user_msg)
    ref_ctx = ""
    for r in refs:
        ref_ctx += f"# REF: {r.get('title','')}\n{r.get('paper_text','')}\n"
        if cs := r.get("code_snippet",""):
            ref_ctx += f"```python\n{cs}\n```\n"
    data_preview = get_data_preview(user_id)
    prompt1 = (
        f"### Data Preview\n{data_preview}\n"
        f"### References\n{ref_ctx}\n"
        f"### Instruction\n{user_msg}\n"
        "### Response (include runnable Python in ```python ...```):"
    )
    neo_out = generate_gpt_neo_text(prompt1)
    refine_prompt = f"Refine this answer (including code):\n\n{neo_out}\n\nReturn final runnable Python in ```python ...```."
    final_out   = deepseek_call(refine_prompt)
    return f"[GPT‑Neo]\n{neo_out}\n\n[DeepSeek]\n{final_out}"

###############################################################################
# Chat flow
###############################################################################
def process_chat(user_msg: str, user_id: str) -> str:
    udata = user_data.setdefault(user_id, {"history": [], "data_info": {}, "learning_points": [], "current_code": "", "original_query": "", "conv_state": None})
    resp = multi_step_generation(user_msg, user_id)
    udata["history"].append({"role":"user","content":user_msg})
    udata["history"].append({"role":"assistant","content":resp})
    return resp

###############################################################################
# Decorator: fix name collision
###############################################################################
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login2"))
        return f(*args, **kwargs)
    return wrapper

###############################################################################
# Flask routes
###############################################################################
@app.route("/signup", methods=["GET","POST"])
def signup2():
    if request.method == "POST":
        username = request.form.get("username","").strip().lower()
        password = request.form.get("password","").strip()
        if not username or not password:
            return render_template("signup3.html", error="Please provide both username and password.")
        users = load_users()
        if username in users:
            return render_template("signup3.html", error="Username already exists.")
        users[username] = {"user_id": str(len(users)+1), "password": hash_password(password)}
        save_users(users)
        return redirect(url_for("login2"))
    return render_template("signup3.html")

@app.route("/login", methods=["GET","POST"])
def login2():
    if request.method == "POST":
        username = request.form.get("username","").strip().lower()
        password = request.form.get("password","").strip()
        users = load_users()
        info = users.get(username)
        if info and info["password"] == hash_password(password):
            session["user_id"] = info["user_id"]
            session["username"] = username
            return redirect(url_for("index2"))
        return render_template("login3.html", error="Invalid username or password.")
    return render_template("login3.html")

@app.route("/logout")
def logout2():
    session.clear()
    return redirect(url_for("login2"))

@app.route("/", methods=["GET"])
def home_redirect():
    return redirect(url_for("login2"))

@app.route("/app", methods=["GET","POST"])
@login_required
def index2():
    user_id = session["user_id"]
    username = session.get("username","")
    feedback = None
    file_feedback = None
    desc_feedback = None
    columns = []
    description = ""
    current_filename = None

    udata = user_data.setdefault(user_id, {"history": [], "data_info": {}, "learning_points": [], "current_code": "", "original_query": "", "conv_state": None})
    data_info = udata.get("data_info", {})

    # if data already uploaded
    if isinstance(data_info.get("df"), pd.DataFrame):
        columns = list(data_info["df"].columns)
        description = data_info.get("description","")
        current_filename = data_info.get("filename","")

    if request.method == "POST":
        action = request.form.get("action")
        if action == "upload_file":
            f = request.files.get("file")
            if f and allowed_file(f.filename):
                fn = secure_filename(f.filename)
                path = os.path.join(UPLOAD_FOLDER, fn)
                f.save(path)
                try:
                    df = pd.read_csv(path) if fn.lower().endswith(".csv") else pd.read_excel(path)
                    udata["data_info"] = {"df": df, "description": "", "filename": fn}
                    columns = list(df.columns)
                    file_feedback = f"File '{fn}' loaded: {df.shape[0]} rows × {df.shape[1]} cols."
                except Exception as e:
                    file_feedback = f"Error loading '{fn}': {e}"
            else:
                file_feedback = "Please upload a .csv or .xlsx file."
        elif action == "save_description":
            desc = request.form.get("data_description","").strip()
            if "df" in data_info:
                data_info["description"] = desc
                desc_feedback = "Description saved."
            else:
                desc_feedback = "Upload data first."
        elif action == "send_message":
            msg = request.form.get("message","").strip()
            if msg:
                feedback = process_chat(msg, user_id)
                log_interaction(msg, feedback, user_id)
            else:
                feedback = "Please enter a message."

    return render_template("index3.html",
        feedback=feedback,
        file_feedback=file_feedback,
        desc_feedback=desc_feedback,
        columns=columns,
        description=description,
        current_filename=current_filename,
        username=username,
        chat_history=udata["history"]
    )

@app.route("/references", methods=["GET","POST"])
@login_required
def references2():
    user_id = session["user_id"]
    msg = None
    if request.method == "POST":
        title = request.form.get("ref_title","").strip()
        paper = request.form.get("ref_paper","").strip()
        code  = request.form.get("ref_code","").strip()
        if not any((title,paper,code)):
            msg = "Provide title, paper text or code snippet."
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = f"ref_{stamp}.json"
            with open(os.path.join(REFERENCES_FOLDER, fn), "w", encoding="utf-8") as f:
                json.dump({"title":title,"paper_text":paper,"code_snippet":code}, f, indent=4)
            msg = "Reference saved."
            # trigger immediate fine‑tune if you like:
            ft = fine_tune_model(user_id)
            msg += f" Fine-tune: {ft}"
    return render_template("references3.html", msg=msg)

###############################################################################
#  Main
###############################################################################
if __name__ == "__main__":
    print("[MAIN] Initializing GPT‑Neo + LoRA model ...")
    initialize_gpt_neo_model()

    choice = input("Run global auto fine‑tune on ALL refs now? (yes/no): ").strip().lower()
    if choice.startswith("y"):
        print("[MAIN] Global fine‑tune triggered.")
        print(global_fine_tune_model())

    print("[MAIN] Starting Flask on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

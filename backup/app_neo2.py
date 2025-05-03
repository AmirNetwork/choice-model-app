# app_deepseek_neo_refined.py
# -*- coding: utf-8 -*-
#
# Multi-step chatbot pipeline with:
#  - User login (signup, login, logout)
#  - Data upload and reference upload for incremental LoRA fine-tuning (GPT-Neo)
#  - A multi-step generation process:
#      1. DeepSeek baseline response
#      2. DeepSeek refines the prompt for GPT-Neo
#      3. GPT-Neo (with LoRA and references) generates a specialized answer
#      4. DeepSeek merges the baseline and specialized responses
#  - Revised deepseek_call with improved error handling for Ollama commands
#
# Dependencies: flask, transformers, peft, datasets, pandas, torch, subprocess, re, traceback

import os
import json
import traceback
import re
import subprocess
from datetime import datetime
import hashlib

import pandas as pd
import torch
from flask import Flask, request, render_template, session, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
print("Flask Process PATH:", os.environ.get("PATH"))
env = os.environ.copy()
env["PATH"] = r"C:\Program Files\Ollama;" + env.get("PATH", "")


# Transformers + LoRA
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

###############################################################################
# Configuration & Constants
###############################################################################
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

UPLOAD_FOLDER = os.path.join(STORAGE_DIR, "uploads")
LOG_FOLDER = os.path.join(STORAGE_DIR, "logs")
COMPLETED_MODELS_FOLDER = os.path.join(STORAGE_DIR, "completed_models")
REFERENCES_FOLDER = os.path.join(STORAGE_DIR, "references")
TRAINING_DATA_FOLDER = os.path.join(STORAGE_DIR, "training_data")

for folder in [UPLOAD_FOLDER, LOG_FOLDER, COMPLETED_MODELS_FOLDER, REFERENCES_FOLDER, TRAINING_DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx"}
CACHE_DIR = "./hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# GPT-Neo base model name
GPTNEO_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"

# Flask app config
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# For user credentials
USER_DB_FILE = os.path.join(STORAGE_DIR, "users.json")

# Per-user data in memory: key=user_id => value=dict
user_data = {}

# LLM timeout (in seconds)
LLM_TIMEOUT_SECS = 180

###############################################################################
# 1) DeepSeek Integration with Improved Error Handling
###############################################################################
def deepseek_call(prompt_text: str) -> str:
    command = "ollama run deepseek-r1:7b"
    try:
        env = os.environ.copy()
        # Ensure the directory for Ollama is included.
        ollama_dir = r"C:\Users\amirg\AppData\Local\Programs\Ollama"
        env["PATH"] = ollama_dir + ";" + env.get("PATH", "")

        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        stdout, stderr = proc.communicate(input=prompt_text + "\n", timeout=LLM_TIMEOUT_SECS)

        if proc.returncode != 0:
            error_message = stderr.strip() or f"Ollama command failed (code {proc.returncode})"
            if "connection refused" in error_message.lower():
                return "Error: Cannot connect to Ollama service."
            return f"Error calling DeepSeek (Ollama): {error_message}"

        resp = re.sub(r'\x1b\[\d{1,3}m', '', stdout.strip())
        return resp

    except subprocess.TimeoutExpired:
        return f"Error: The request to the language model timed out after {LLM_TIMEOUT_SECS} seconds."
    except FileNotFoundError:
        return ("Error: Ollama command not found. "
                "Please ensure Ollama is installed and its location is added to your system's PATH.")
    except Exception as e:
        return f"Error running LLM: {e}\n{traceback.format_exc()}"


def deepseek_baseline_pass(user_msg: str) -> str:
    """
    Pass 1: Use DeepSeek to get a baseline (internal) response.
    """
    prompt = (
        "You are DeepSeek+Olama providing a baseline internal thought. "
        "Do NOT simply repeat the user's text verbatim. Summarize or rephrase if needed.\n\n"
        f"User request: {user_msg}\n\n"
        "Give your best general knowledge answer."
    )
    return deepseek_call(prompt)

def deepseek_refine_prompt_for_gptneo(user_msg: str) -> str:
    """
    Pass 2: Let DeepSeek produce a refined GPT-Neo prompt based on the user's query.
    """
    prompt = (
        "You are DeepSeek+Olama. The user gave the following request:\n\n"
        f"'{user_msg}'\n\n"
        "Rewrite or refine this in a concise, clear way for GPT-Neo to generate a specialized reference-based answer. "
        "Focus on clarity, avoid repeating text verbatim, but keep all essential details."
    )
    refined = deepseek_call(prompt)
    return refined

def deepseek_supervise_merge(baseline_resp: str, specialized_resp: str, user_msg: str) -> str:
    """
    Pass 4: Let DeepSeek merge the baseline and specialized GPT-Neo responses into a final answer.
    """
    merge_prompt = (
        "You are DeepSeek+Olama. You have two partial answers:\n\n"
        f"[Baseline]\n{baseline_resp}\n\n"
        f"[Specialized]\n{specialized_resp}\n\n"
        f"User request:\n'{user_msg}'\n\n"
        "Combine these into a single, coherent final answer. Avoid repeating large blocks verbatim and focus on clarity."
    )
    merged = deepseek_call(merge_prompt)
    return merged

###############################################################################
# 2) GPT-Neo + LoRA Setup
###############################################################################
MAX_INPUT_TOKENS = 1792
MAX_NEW_TOKENS = 256

g_neo_tokenizer = None
g_neo_model = None

def initialize_gpt_neo_model():
    """Load GPT-Neo (with LoRA) into memory."""
    global g_neo_tokenizer, g_neo_model
    print(f"[INIT] Loading GPT-Neo model {GPTNEO_MODEL_NAME} from cache_dir={CACHE_DIR} ...")
    g_neo_tokenizer = AutoTokenizer.from_pretrained(
        GPTNEO_MODEL_NAME,
        cache_dir=CACHE_DIR
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        GPTNEO_MODEL_NAME,
        cache_dir=CACHE_DIR,
        device_map="auto",
        torch_dtype=torch.float16
    )
    lora_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM"
    )
    g_neo_model = get_peft_model(base_model, lora_config)
    print("[INIT] GPT-Neo + LoRA loaded successfully.")

def generate_gpt_neo_text(prompt: str, temperature=0.7) -> str:
    """
    Generate text from GPT-Neo with LoRA.
    The prompt is encoded (up to MAX_INPUT_TOKENS) and then generation is done for up to MAX_NEW_TOKENS.
    """
    global g_neo_model, g_neo_tokenizer
    if g_neo_model is None or g_neo_tokenizer is None:
        return "Error: GPT-Neo model not initialized."

    enc = g_neo_tokenizer.encode(prompt, add_special_tokens=False)
    if len(enc) > MAX_INPUT_TOKENS:
        enc = enc[:MAX_INPUT_TOKENS]

    inputs_dict = {
        "input_ids": torch.tensor([enc], dtype=torch.long).to(g_neo_model.device)
    }
    with torch.no_grad():
        out_ids = g_neo_model.generate(
            **inputs_dict,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            pad_token_id=g_neo_tokenizer.eos_token_id
        )
    out_text = g_neo_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    generated = out_text[len(g_neo_tokenizer.decode(enc, skip_special_tokens=True)):].strip()
    return generated

###############################################################################
# 3) Basic Utilities
###############################################################################
def load_users():
    """Load user credentials from a JSON file."""
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(USER_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users_dict):
    """Save user credentials back to the JSON file."""
    with open(USER_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(users_dict, f, indent=4)

def hash_password(password):
    """Simple SHA-256 hashing for passwords (consider using a salted approach in production)."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def allowed_file(filename):
    """Check if uploaded file is a CSV or XLSX file."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def log_interaction(user_msg, bot_msg, user_id=None):
    """Append conversation details to the chat log."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} - User({user_id}): {user_msg} | Bot: {bot_msg}\n"
    try:
        with open(os.path.join(LOG_FOLDER, "chat.log"), "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"Logging failed: {e}")

def interpret_feedback(user_msg: str) -> str:
    """
    Interpret user feedback.
    If the user says 'looks good' or 'finish', we interpret it as 'satisfied';
    if they say 'modify', 'change', or 'improve', it is interpreted as 'modify';
    otherwise, 'unclear'.
    """
    lw = user_msg.lower()
    if "looks good" in lw or "finish" in lw or "satisfied" in lw:
        return "satisfied"
    elif "modify" in lw or "change" in lw or "improve" in lw:
        return "modify"
    return "unclear"

###############################################################################
# 4) Save Artifacts
###############################################################################
def save_artifacts(user_id):
    """Save the final code and details as a JSON artifact for record-keeping."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    udata = user_data.get(user_id, {})
    final_code = udata.get("current_code", "No code.")
    original_query = udata.get("original_query", "Unknown query.")
    data_info = udata.get("data_info", {})
    user_lessons = udata.get("learning_points", [])

    artifact = {
        "timestamp": datetime.now().isoformat(),
        "original_query": original_query,
        "final_code": final_code,
        "data_info": data_info,
        "learning_points": user_lessons
    }
    out_fname = f"completion_{stamp}.json"
    out_path = os.path.join(COMPLETED_MODELS_FOLDER, out_fname)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=4)
        return f"Process details saved as {out_fname}."
    except Exception as e:
        print(f"Error saving artifacts: {e}")
        return "Error saving artifacts."

###############################################################################
# 5) GPT-Neo Fine-Tuning with References + Feedback
###############################################################################
def build_training_dataset(user_id) -> str:
    """
    Create a .jsonl dataset from references & user feedback.
    This dataset is later used for quick in-memory LoRA fine-tuning for GPT-Neo.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = f"training_{stamp}.jsonl"
    out_path = os.path.join(TRAINING_DATA_FOLDER, out_fname)

    instruct_samples = []

    # Gather references from the REFERENCES_FOLDER
    for fname in os.listdir(REFERENCES_FOLDER):
        if fname.endswith(".json"):
            fpath = os.path.join(REFERENCES_FOLDER, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as rf:
                    ref_data = json.load(rf)
                title = ref_data.get("title", "")
                paper_text = ref_data.get("paper_text", "")
                code_snip = ref_data.get("code_snippet", "")
                instruction = (
                    f"Title: {title}\n"
                    f"Paper:\n{paper_text}\n"
                    f"Code:\n{code_snip}"
                )
                output = "Understood. Incorporate these references in future answers."
                instruct_samples.append({"instruction": instruction, "output": output})
            except Exception as e:
                print(f"[WARN] reading ref {fname}: {e}")

    # Gather user lessons/feedback
    udata = user_data.get(user_id, {})
    user_lessons = udata.get("learning_points", [])
    for lesson in user_lessons:
        instruct_samples.append({
            "instruction": f"User feedback: {lesson}",
            "output": "We incorporate this feedback in future responses."
        })

    # If no samples, create an empty file
    if not instruct_samples:
        with open(out_path, "w", encoding="utf-8") as bf:
            pass
        return out_path

    with open(out_path, "w", encoding="utf-8") as ds_f:
        for s in instruct_samples:
            ds_f.write(json.dumps(s) + "\n")
    return out_path

def fine_tune_model(user_id) -> str:
    """
    Perform a quick in-memory LoRA fine-tune of GPT-Neo on the references and user feedback.
    """
    global g_neo_model, g_neo_tokenizer
    ds_file = build_training_dataset(user_id)
    if not os.path.exists(ds_file):
        return "No dataset file found."
    if os.path.getsize(ds_file) < 10:
        return "Dataset empty; skipping fine-tune."

    try:
        raw_ds = load_dataset("json", data_files=ds_file, split="train")

        def preprocess(ex):
            return {
                "prompt": f"Instruction: {ex['instruction']}\nResponse:",
                "label": ex["output"]
            }
        raw_ds = raw_ds.map(preprocess)

        def tokenize_fn(ex):
            full_text = ex["prompt"] + " " + ex["label"]
            tok = g_neo_tokenizer(full_text, truncation=True, max_length=512)
            tok["labels"] = tok["input_ids"].copy()
            return tok

        ds_mapped = raw_ds.map(tokenize_fn, batched=False)

        def data_collator(features):
            keys = ["input_ids", "attention_mask", "labels"]
            batch = {k: [f[k] for f in features] for k in keys}
            out = g_neo_tokenizer.pad({"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}, return_tensors="pt")
            labs = g_neo_tokenizer.pad({"input_ids": batch["labels"]}, return_tensors="pt")
            out["labels"] = labs["input_ids"]
            return out

        train_args = TrainingArguments(
            output_dir=os.path.join(TRAINING_DATA_FOLDER, "lora_out"),
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            fp16=True,
            logging_steps=5,
            save_steps=50
        )
        trainer = Trainer(
            model=g_neo_model,
            args=train_args,
            train_dataset=ds_mapped,
            data_collator=data_collator
        )
        trainer.train()
        rows = raw_ds.num_rows
        return f"LoRA fine-tune done. {rows} samples used. (In memory)"
    except Exception as e:
        print(f"[FT] error: {e}")
        return f"Fine-tune error: {e}"

###############################################################################
# 6) Multi-Step Generation Pipeline
###############################################################################
def read_filtered_references(user_msg: str) -> str:
    """
    Return an excerpt of references relevant to the user's query (to avoid too large a prompt).
    """
    lw = user_msg.lower()
    keywords = re.findall(r"\b\w+\b", lw)
    lines = []
    for fn in os.listdir(REFERENCES_FOLDER):
        if fn.endswith(".json"):
            path = os.path.join(REFERENCES_FOLDER, fn)
            try:
                with open(path, "r", encoding="utf-8") as rr:
                    ref_data = json.load(rr)
                text_block = json.dumps(ref_data)
                if any(k in text_block.lower() for k in keywords):
                    short_text = text_block[:1000]  # limit excerpt length
                    lines.append(f"--- {fn} (excerpt) ---\n{short_text}\n------------------\n")
            except Exception as e:
                print(f"[WARN] reading {fn}: {e}")
    return "\n".join(lines) if lines else "No relevant references found for the user's query."

def gptneo_reference_pass(refined_prompt: str, user_msg: str) -> str:
    """
    Pass 3: Use GPT-Neo (with references) to generate a specialized answer.
    We combine the filtered references with the DeepSeek-refined prompt.
    """
    refs_text = read_filtered_references(user_msg)
    final_prompt = (
        "You are GPT-Neo with domain references. "
        "IMPORTANT: Do NOT simply repeat references or user text verbatim; summarize or rephrase as needed.\n\n"
        f"=== Filtered References ===\n{refs_text}\n=== End of References ===\n\n"
        "Below is the refined prompt from DeepSeek based on the user's request:\n\n"
        f"{refined_prompt}\n\n"
        "Use these references only if they add domain-specific insight."
    )
    return generate_gpt_neo_text(final_prompt)

def multi_step_generation(user_msg: str) -> str:
    """
    Master pipeline:
       1) Get baseline answer using DeepSeek
       2) Refine the GPT-Neo prompt using DeepSeek
       3) Generate a specialized answer using GPT-Neo with references
       4) Merge the baseline and specialized answers using DeepSeek supervision
    """
    baseline_answer = deepseek_baseline_pass(user_msg)
    refined_prompt = deepseek_refine_prompt_for_gptneo(user_msg)
    specialized_answer = gptneo_reference_pass(refined_prompt, user_msg)
    final_answer = deepseek_supervise_merge(baseline_answer, specialized_answer, user_msg)
    return final_answer

###############################################################################
# 7) Chat Flow
###############################################################################
def process_chat(user_msg: str, user_id: str) -> str:
    """
    Main function for processing each user message.
    If in feedback mode, interpret feedback and either finalize or refine code.
    Otherwise, process as a new query using the multi-step generation pipeline.
    """
    if user_id not in user_data:
        user_data[user_id] = {
            "history": [],
            "learning_points": [],
            "data_info": {},
            "current_code": "",
            "original_query": "",
            "conv_state": None
        }
    udata = user_data[user_id]
    conversation = udata["history"]
    conversation.append({"role": "user", "content": user_msg})
    udata["history"] = conversation

    if udata.get("conv_state") == "AWAITING_FEEDBACK":
        decision = interpret_feedback(user_msg)
        if decision == "satisfied":
            artifact_msg = save_artifacts(user_id)
            ft_msg = fine_tune_model(user_id)
            udata["conv_state"] = None
            udata["current_code"] = ""
            udata["original_query"] = ""
            bot_resp = f"Great! {artifact_msg}\n{ft_msg}\nYou can start a new request."
            conversation.append({"role": "assistant", "content": bot_resp})
            user_data[user_id] = udata
            return bot_resp
        elif decision == "modify":
            orig_code = udata.get("current_code", "")
            refine_prompt = (
                f"Original code:\n```python\n{orig_code}\n```\n\n"
                f"User feedback: {user_msg}\n"
                "Please refine or improve this code. Provide the final code in triple backticks."
            )
            new_code = generate_gpt_neo_text(refine_prompt)
            udata["current_code"] = new_code
            ft_msg = fine_tune_model(user_id)
            bot_resp = (
                f"Refined code:\n\n{new_code}\n\n"
                f"Fine-tune result: {ft_msg}\n"
                "Anything else? (Say 'Looks good' or describe further changes.)"
            )
            conversation.append({"role": "assistant", "content": bot_resp})
            user_data[user_id] = udata
            return bot_resp
        else:
            bot_resp = "Not sure if you're done or want changes. Please clarify with 'Looks good' or specify modifications."
            conversation.append({"role": "assistant", "content": bot_resp})
            user_data[user_id] = udata
            return bot_resp

    udata["original_query"] = user_msg
    final_answer = multi_step_generation(user_msg)
    conversation.append({"role": "assistant", "content": final_answer})
    user_data[user_id] = udata
    return final_answer

###############################################################################
# 8) Flask Routes
###############################################################################
def login_required(f):
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login2"))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route("/signup", methods=["GET", "POST"])
def signup2():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()
        if not username or not password:
            return render_template("signup2.html", error="Please provide a username and password.")
        all_users = load_users()
        if username in all_users:
            return render_template("signup2.html", error="Username already exists. Please pick another.")
        hashed = hash_password(password)
        new_user_id = str(len(all_users) + 1)
        all_users[username] = {"user_id": new_user_id, "password": hashed}
        save_users(all_users)
        return redirect(url_for("login2"))
    return render_template("signup2.html")

@app.route("/login", methods=["GET", "POST"])
def login2():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()
        if not username or not password:
            return render_template("login2.html", error="Please provide username and password.")
        all_users = load_users()
        user_info = all_users.get(username)
        if not user_info:
            return render_template("login2.html", error="No such username found.")
        hashed = hash_password(password)
        if user_info["password"] != hashed:
            return render_template("login2.html", error="Incorrect password.")
        session["user_id"] = user_info["user_id"]
        session["username"] = username
        return redirect(url_for("index2"))
    return render_template("login2.html")

@app.route("/logout")
def logout2():
    session.clear()
    return redirect(url_for("login2"))

@app.route("/", methods=["GET"])
def home_redirect():
    return redirect(url_for("login2"))

@app.route("/home", methods=["GET"])
@login_required
def home2():
    return redirect(url_for("index2"))

@app.route("/app", methods=["GET", "POST"])
@login_required
def index2():
    user_id = session["user_id"]
    username = session.get("username", "UnknownUser")
    feedback = None
    file_feedback = None
    desc_feedback = None
    columns_list = []
    current_description = ""
    current_filename = None

    if user_id not in user_data:
        user_data[user_id] = {
            "history": [],
            "learning_points": [],
            "data_info": {},
            "current_code": "",
            "original_query": "",
            "conv_state": None
        }
    udata = user_data[user_id]
    data_info = udata.get("data_info", {})

    if "df" in data_info and isinstance(data_info["df"], pd.DataFrame):
        columns_list = data_info["df"].columns.tolist()
        current_description = data_info.get("description", "")
        current_filename = data_info.get("filename", "")

    if request.method == "POST":
        action = request.form.get("action")
        if action == "upload_file":
            f = request.files.get("file")
            if f and f.filename and allowed_file(f.filename):
                user_data[user_id]["history"] = []
                user_data[user_id]["learning_points"] = []
                user_data[user_id]["current_code"] = ""
                user_data[user_id]["conv_state"] = None
                fname = secure_filename(f.filename)
                fpath = os.path.join(UPLOAD_FOLDER, fname)
                try:
                    f.save(fpath)
                    if fname.lower().endswith(".csv"):
                        try:
                            df = pd.read_csv(fpath, encoding='utf-8')
                        except UnicodeDecodeError:
                            df = pd.read_csv(fpath, encoding='latin-1')
                    else:
                        df = pd.read_excel(fpath)
                    if df.empty or df.shape[0] < 2:
                        file_feedback = f"Error: '{fname}' has insufficient rows."
                    else:
                        data_info["filename"] = fname
                        data_info["df"] = df
                        data_info["description"] = ""
                        user_data[user_id]["data_info"] = data_info
                        columns_list = df.columns.tolist()
                        current_filename = fname
                        file_feedback = f"File '{fname}' uploaded. Rows={df.shape[0]}, Cols={df.shape[1]}"
                except Exception as e:
                    file_feedback = f"Error reading '{fname}': {e}"
            else:
                file_feedback = "No valid CSV/XLSX file provided."
        elif action == "save_description":
            dtext = request.form.get("data_description", "").strip()
            if "df" in data_info:
                data_info["description"] = dtext
                current_description = dtext
                desc_feedback = "Data description saved."
            else:
                desc_feedback = "Please upload data first."
            user_data[user_id]["data_info"] = data_info
        elif action == "send_message":
            user_msg = request.form.get("message", "").strip()
            if not user_msg:
                feedback = "Please enter a message."
            else:
                bot_resp = process_chat(user_msg, user_id)
                feedback = bot_resp
                log_interaction(user_msg, bot_resp, user_id=user_id)

    return render_template(
        "index2.html",
        feedback=feedback,
        file_feedback=file_feedback,
        desc_feedback=desc_feedback,
        columns=columns_list,
        description=current_description,
        current_filename=current_filename,
        username=username
    )

@app.route("/references", methods=["GET", "POST"])
@login_required
def references2():
    user_id = session["user_id"]
    msg = None
    if request.method == "POST":
        ref_title = request.form.get("ref_title", "").strip()
        paper_txt = request.form.get("ref_paper", "").strip()
        code_txt = request.form.get("ref_code", "").strip()
        if not ref_title and not paper_txt and not code_txt:
            msg = "Please provide some text or code."
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            f_name = f"ref_{stamp}.json"
            r_path = os.path.join(REFERENCES_FOLDER, f_name)
            ref_obj = {"title": ref_title, "paper_text": paper_txt, "code_snippet": code_txt}
            try:
                with open(r_path, "w", encoding="utf-8") as r_f:
                    json.dump(ref_obj, r_f, indent=4)
                ft_res = fine_tune_model(user_id)
                msg = f"Thank you! Your reference was saved. Fine-tune result: {ft_res}"
            except Exception as e:
                msg = f"Error saving reference: {e}"
    return render_template("references2.html", msg=msg)

@app.route("/chat", methods=["POST"])
@login_required
def chat_api():
    user_id = session["user_id"]
    data = request.get_json() or {}
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"response": "Please enter a message."})
    bot_resp = process_chat(user_msg, user_id)
    log_interaction(f"[AJAX] {user_msg}", bot_resp, user_id=user_id)
    return jsonify({"response": bot_resp})

###############################################################################
# 9) Main
###############################################################################
if __name__ == "__main__":
    print("[MAIN] Initializing GPT-Neo + LoRA model ...")
    initialize_gpt_neo_model()
    print("[MAIN] Starting Flask on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)

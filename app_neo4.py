# app_neo3.py
# -*- coding: utf-8 -*-
"""
Revised multi-step chatbot pipeline that:
  - Supports user login (signup, login, logout)
  - Accepts data uploads and incremental fine-tuning with GPT-Neo (using LoRA)
  - Implements a streamlined generation process:
       1. GPT-Neo generates a domain-specific answer (with uploaded data context and Python code snippet)
       2. DeepSeek synthesizes a final answer by merging its knowledge with the GPT-Neo output
  - Logs progress with timestamps and elapsed times for transparency
  - Incorporates suggestions for modularization, error handling, and security improvements

Dependencies:
  - flask, transformers, peft, datasets, pandas, torch, subprocess, re, traceback, time, logging
  - (Celery integration can be added later for asynchronous task processing)
"""

import os
import json
import traceback
import re
import subprocess
import time
from datetime import datetime
import hashlib
import logging

import pandas as pd
import torch
from flask import Flask, request, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename

# -------------------------
# Configuration & Logging
# -------------------------
# Configuration constants (could be moved to a separate config file or environment variables)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
UPLOAD_FOLDER = os.path.join(STORAGE_DIR, "uploads")
LOG_FOLDER = os.path.join(STORAGE_DIR, "logs")
COMPLETED_MODELS_FOLDER = os.path.join(STORAGE_DIR, "completed_models")
REFERENCES_FOLDER = os.path.join(STORAGE_DIR, "references")
TRAINING_DATA_FOLDER = os.path.join(STORAGE_DIR, "training_data")
TRAINED_REFERENCES_FOLDER = os.path.join(STORAGE_DIR, "trained_references")
ALLOWED_EXTENSIONS = {"csv", "xlsx"}
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
GPTNEO_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
USER_DB_FILE = os.path.join(STORAGE_DIR, "users.json")
LLM_TIMEOUT_SECS = 180  # seconds for LLM calls
MAX_INPUT_TOKENS = 1792
MAX_NEW_TOKENS = 256

# Ensure necessary directories exist
for folder in [UPLOAD_FOLDER, LOG_FOLDER, COMPLETED_MODELS_FOLDER,
               REFERENCES_FOLDER, TRAINING_DATA_FOLDER, TRAINED_REFERENCES_FOLDER, CACHE_DIR]:
    os.makedirs(folder, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join(LOG_FOLDER, "app.log"),
                    filemode="a")
logger = logging.getLogger(__name__)
logger.info("Application starting...")

# -------------------------
# Flask App Setup
# -------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# In-memory per-user data storage (for quick prototyping; consider using a persistent DB for production)
user_data = {}

# -------------------------
# Helper Functions and Utilities
# -------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_password(password):
    # Consider using bcrypt or passlib for stronger hashing in production.
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def load_users():
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(USER_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users_dict):
    with open(USER_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(users_dict, f, indent=4)

def log_interaction(user_msg, bot_msg, user_id=None):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} - User({user_id}): {user_msg} | Bot: {bot_msg}\n"
    try:
        with open(os.path.join(LOG_FOLDER, "chat.log"), "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        logger.error(f"Logging failed: {e}")

def get_data_preview(user_id, num_rows=20) -> str:
    """Return CSV preview of uploaded data if available."""
    udata = user_data.get(user_id, {})
    data_info = udata.get("data_info", {})
    if "df" in data_info and isinstance(data_info["df"], pd.DataFrame):
        preview = data_info["df"].head(num_rows).to_csv(index=False)
        return f"Uploaded Data (first {num_rows} rows):\n{preview}\n"
    else:
        return "No uploaded data available.\n"

def interpret_feedback(user_msg: str) -> str:
    lw = user_msg.lower()
    if "looks good" in lw or "finish" in lw or "satisfied" in lw:
        return "satisfied"
    elif "modify" in lw or "change" in lw or "improve" in lw:
        return "modify"
    return "unclear"

def save_artifacts(user_id):
    """Save user conversation and domain context artifacts."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    udata = user_data.get(user_id, {})
    artifact = {
        "timestamp": datetime.now().isoformat(),
        "original_query": udata.get("original_query", "Unknown query."),
        "final_code": udata.get("current_code", "No code."),
        "data_info": udata.get("data_info", {}),
        "learning_points": udata.get("learning_points", [])
    }
    out_fname = f"completion_{stamp}.json"
    out_path = os.path.join(COMPLETED_MODELS_FOLDER, out_fname)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=4)
        logger.info(f"Artifacts saved as {out_fname}")
        return f"Process details saved as {out_fname}."
    except Exception as e:
        logger.error(f"Error saving artifacts: {e}")
        return "Error saving artifacts."

# -------------------------
# DeepSeek Integration
# -------------------------
def deepseek_call(prompt_text: str) -> str:
    """Call the DeepSeek (Ollama) process using subprocess with improved error handling."""
    command = "ollama run deepseek-r1:7b"
    try:
        env_vars = os.environ.copy()
        ollama_dir = r"C:\Users\amirg\AppData\Local\Programs\Ollama"
        env_vars["PATH"] = ollama_dir + ";" + env_vars.get("PATH", "")
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            env=env_vars,
            encoding='utf-8',
            errors='replace'
        )
        stdout, stderr = proc.communicate(input=prompt_text + "\n", timeout=LLM_TIMEOUT_SECS)
        if proc.returncode != 0:
            error_message = stderr.strip() or f"Ollama command failed (code {proc.returncode})"
            logger.error(error_message)
            if "connection refused" in error_message.lower():
                return "Error: Cannot connect to Ollama service."
            return f"Error calling DeepSeek (Ollama): {error_message}"
        # Remove ANSI escape codes if any
        resp = re.sub(r'\x1b\[\d{1,3}m', '', stdout.strip())
        return resp
    except subprocess.TimeoutExpired:
        return f"Error: The request to the language model timed out after {LLM_TIMEOUT_SECS} seconds."
    except FileNotFoundError:
        return ("Error: Ollama command not found. Please ensure Ollama is installed and its location is added to your system's PATH.")
    except Exception as e:
        err = f"Error running LLM: {e}\n{traceback.format_exc()}"
        logger.error(err)
        return err

# -------------------------
# GPT-Neo + LoRA Model Setup and Generation
# -------------------------
g_neo_tokenizer = None
g_neo_model = None

def initialize_gpt_neo_model():
    """Load GPT-Neo base model and attach LoRA modifications."""
    global g_neo_tokenizer, g_neo_model
    logger.info(f"Loading GPT-Neo model {GPTNEO_MODEL_NAME} from cache_dir={CACHE_DIR} ...")
    try:
        g_neo_tokenizer = AutoTokenizer.from_pretrained(GPTNEO_MODEL_NAME, cache_dir=CACHE_DIR)
        base_model = AutoModelForCausalLM.from_pretrained(
            GPTNEO_MODEL_NAME,
            cache_dir=CACHE_DIR,
            device_map="auto",
            torch_dtype=torch.float16
        )
        lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        g_neo_model = get_peft_model(base_model, lora_config)
        logger.info("GPT-Neo + LoRA loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load GPT-Neo model: {e}")
        raise

def generate_gpt_neo_text(prompt: str, temperature=0.7) -> str:
    """Generate text from GPT-Neo using the prompt."""
    global g_neo_model, g_neo_tokenizer
    if g_neo_model is None or g_neo_tokenizer is None:
        return "Error: GPT-Neo model not initialized."
    try:
        enc = g_neo_tokenizer.encode(prompt, add_special_tokens=False)
        if len(enc) > MAX_INPUT_TOKENS:
            enc = enc[:MAX_INPUT_TOKENS]
        inputs_dict = {"input_ids": torch.tensor([enc], dtype=torch.long).to(g_neo_model.device)}
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
    except Exception as e:
        logger.error(f"Error generating GPT-Neo text: {e}")
        return f"Error generating GPT-Neo text: {e}"

# -------------------------
# Fine-Tuning and Reference Handling
# -------------------------
def build_training_dataset(user_id) -> str:
    """Build dataset from reference files and user feedback for fine-tuning."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = f"training_{stamp}.jsonl"
    out_path = os.path.join(TRAINING_DATA_FOLDER, out_fname)
    instruct_samples = []
    # Process reference files
    reference_files = [fname for fname in os.listdir(REFERENCES_FOLDER) if fname.endswith(".json")]
    for fname in reference_files:
        fpath = os.path.join(REFERENCES_FOLDER, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as rf:
                ref_data = json.load(rf)
            title = ref_data.get("title", "")
            paper_text = ref_data.get("paper_text", "")
            code_snip = ref_data.get("code_snippet", "")
            instruction = f"Title: {title}\nPaper:\n{paper_text}\nCode:\n{code_snip}"
            output = "Understood. Incorporate these references in future answers."
            instruct_samples.append({"instruction": instruction, "output": output})
        except Exception as e:
            logger.warning(f"Error reading reference {fname}: {e}")
        # Move processed reference to trained folder
        new_path = os.path.join(TRAINED_REFERENCES_FOLDER, fname)
        try:
            os.rename(fpath, new_path)
        except Exception as e:
            logger.warning(f"Failed to move {fname} to trained references: {e}")

    # Process user feedback stored in memory
    udata = user_data.get(user_id, {})
    user_lessons = udata.get("learning_points", [])
    for lesson in user_lessons:
        instruct_samples.append({
            "instruction": f"User feedback: {lesson}",
            "output": "We incorporate this feedback in future responses."
        })

    data_info = udata.get("data_info", {})
    if "df" in data_info and data_info.get("description", "").strip():
        description = data_info["description"].strip()
        preview = data_info["df"].head(20).to_csv(index=False)
        instruction = f"Data Description: {description}\nData Preview (first 20 rows):\n{preview}"
        output = "User provided data and description integrated."
        instruct_samples.append({"instruction": instruction, "output": output})

    # Write dataset file
    try:
        with open(out_path, "w", encoding="utf-8") as ds_f:
            for s in instruct_samples:
                ds_f.write(json.dumps(s) + "\n")
    except Exception as e:
        logger.error(f"Error building training dataset: {e}")
    return out_path

def fine_tune_model(user_id) -> str:
    """Fine-tune the GPT-Neo model with user references and feedback (stub: runs in-memory)."""
    global g_neo_model, g_neo_tokenizer
    ds_file = build_training_dataset(user_id)
    if not os.path.exists(ds_file) or os.path.getsize(ds_file) < 10:
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
        from transformers import TrainingArguments, Trainer
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
        logger.error(f"Fine-tune error: {e}")
        return f"Fine-tune error: {e}"

# -------------------------
# Multi-Step Generation Pipeline with Progress Logging
# -------------------------
def multi_step_generation(user_msg: str, user_id: str) -> str:
    """
    Streamlined generation pipeline with progress logging:
      1. GPT-Neo generates a domain-specific answer (including a Python code snippet wrapped in triple backticks).
      2. DeepSeek then synthesizes a final answer using its knowledge and the GPT-Neo output.
      3. A progress log with elapsed times is appended to the final answer.
    """
    progress_log = []
    total_start = time.time()
    progress_log.append("== Generation Process Started ==")

    # Stage 1: GPT-Neo Domain-Specific Answer
    stage1_start = time.time()
    progress_log.append("Stage 1: Generating domain-specific answer using GPT-Neo...")
    data_preview = get_data_preview(user_id, num_rows=20)
    gptneo_prompt = (
        "You are GPT-Neo with domain expertise.\n"
        "Below is the uploaded data for context:\n"
        f"{data_preview}\n"
        "User request:\n"
        f"{user_msg}\n\n"
        "Generate a specialized, domain-specific answer. Your answer must include a Python code snippet wrapped in triple backticks if applicable. "
        "Keep the answer concise and clear."
    )
    gptneo_response = generate_gpt_neo_text(gptneo_prompt)
    stage1_elapsed = time.time() - stage1_start
    progress_log.append(f"Stage 1 completed in {stage1_elapsed:.2f} seconds.")

    # Stage 2: DeepSeek Synthesis
    stage2_start = time.time()
    progress_log.append("Stage 2: Synthesizing final answer using DeepSeek...")
    final_prompt = (
        "You are DeepSeek+Ollama. Consider the following domain-specific answer generated by GPT-Neo:\n\n"
        f"{gptneo_response}\n\n"
        "Also consider the uploaded data context below:\n"
        f"{data_preview}\n\n"
        "User request:\n"
        f"{user_msg}\n\n"
        "Using both your own knowledge and the GPT-Neo output, generate a final, coherent answer. "
        "Ensure that the answer includes a Python code snippet wrapped in triple backticks."
    )
    final_response = deepseek_call(final_prompt)
    stage2_elapsed = time.time() - stage2_start
    progress_log.append(f"Stage 2 completed in {stage2_elapsed:.2f} seconds.")

    total_elapsed = time.time() - total_start
    progress_log.append(f"Total generation time: {total_elapsed:.2f} seconds.")
    progress_log.append("== Generation Process Completed ==")
    final_response_with_progress = "\n".join(progress_log) + "\n\nFinal Answer:\n" + final_response
    return final_response_with_progress

# -------------------------
# Chat Flow and User Interaction
# -------------------------
def process_chat(user_msg: str, user_id: str) -> str:
    """
    Process the incoming chat message from the user:
      - Append it to user history.
      - If awaiting feedback, process feedback accordingly.
      - Otherwise, invoke the generation pipeline.
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
    udata["history"].append({"role": "user", "content": user_msg})
    
    # Check if the conversation is in "feedback" mode
    if udata.get("conv_state") == "AWAITING_FEEDBACK":
        decision = interpret_feedback(user_msg)
        if decision == "satisfied":
            artifact_msg = save_artifacts(user_id)
            ft_msg = fine_tune_model(user_id)
            udata["conv_state"] = None
            udata["current_code"] = ""
            udata["original_query"] = ""
            bot_resp = f"Great! {artifact_msg}\n{ft_msg}\nYou can start a new request."
            udata["history"].append({"role": "assistant", "content": bot_resp})
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
            udata["history"].append({"role": "assistant", "content": bot_resp})
            return bot_resp
        else:
            bot_resp = "Not sure if you're done or want changes. Please clarify with 'Looks good' or specify modifications."
            udata["history"].append({"role": "assistant", "content": bot_resp})
            return bot_resp

    # Save the original query and run the multi-step generation (synchronously)
    udata["original_query"] = user_msg
    final_answer = multi_step_generation(user_msg, user_id)
    udata["history"].append({"role": "assistant", "content": final_answer})
    return final_answer

# -------------------------
# Flask Routes and Authentication
# -------------------------
def login_required(f):
    """Decorator to enforce login on routes."""
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
            return render_template("signup3.html", error="Please provide a username and password.")
        all_users = load_users()
        if username in all_users:
            return render_template("signup3.html", error="Username already exists. Please pick another.")
        hashed = hash_password(password)
        new_user_id = str(len(all_users) + 1)
        all_users[username] = {"user_id": new_user_id, "password": hashed}
        save_users(all_users)
        return redirect(url_for("login2"))
    return render_template("signup3.html")

@app.route("/login", methods=["GET", "POST"])
def login2():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()
        if not username or not password:
            return render_template("login3.html", error="Please provide username and password.")
        all_users = load_users()
        user_info = all_users.get(username)
        if not user_info:
            return render_template("login3.html", error="No such username found.")
        hashed = hash_password(password)
        if user_info["password"] != hashed:
            return render_template("login3.html", error="Incorrect password.")
        session["user_id"] = user_info["user_id"]
        session["username"] = username
        return redirect(url_for("index2"))
    return render_template("login3.html")

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
        columns_list = list(data_info["df"].columns)
        current_description = data_info.get("description", "")
        current_filename = data_info.get("filename", "")

    if request.method == "POST":
        action = request.form.get("action")
        if action == "upload_file":
            f = request.files.get("file")
            if f and f.filename and allowed_file(f.filename):
                # Reset conversation for a new file.
                udata["history"] = []
                udata["learning_points"] = []
                udata["current_code"] = ""
                udata["conv_state"] = None
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
                        udata["data_info"] = data_info
                        columns_list = list(df.columns)
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
            udata["data_info"] = data_info
        elif action == "send_message":
            user_msg = request.form.get("message", "").strip()
            if not user_msg:
                feedback = "Please enter a message."
            else:
                feedback = process_chat(user_msg, user_id)
                log_interaction(user_msg, feedback, user_id=user_id)
    chat_history = udata.get("history", [])
    return render_template("index3.html",
                           feedback=feedback,
                           file_feedback=file_feedback,
                           desc_feedback=desc_feedback,
                           columns=columns_list,
                           description=current_description,
                           current_filename=current_filename,
                           username=username,
                           chat_history=chat_history)

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
    return render_template("references3.html", msg=msg)

# -------------------------
# Main Entry Point
# -------------------------
if __name__ == "__main__":
    logger.info("Initializing GPT-Neo + LoRA model ...")
    initialize_gpt_neo_model()
    logger.info("Starting Flask on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)

# app_neo.py (Revised to consider conversation history)
# -*- coding: utf-8 -*-
import os
import json
import traceback
from datetime import datetime

import pandas as pd
import torch
from flask import Flask, request, render_template, session, jsonify
from werkzeug.utils import secure_filename

# Hugging Face + LoRA
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# -----------------------------
# Configuration
# -----------------------------
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

# We'll keep a local cache for the model to avoid re-downloads
CACHE_DIR = "./hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

BASE_MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # ~2048 token context limit

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

uploaded_data = {}

CHOICE_RELEVANT_KEYWORDS = [
    "choice model", "discrete choice", "logit", "mixed logit", "conjoint",
    "preference", "utility", "random utility", "multinomial", "latent class",
    "nested logit", "probit", "mlogit", "market share", "willingness to pay", "wtp"
]

# We'll define that we want 256 new tokens => so we allow 1792 tokens from prompt
MAX_INPUT_TOKENS = 1792
MAX_NEW_TOKENS = 256

# Summation => 1792 + 256 = 2048 total positions

# Global model
g_tokenizer = None
g_model = None

# -----------------------------
# Basic Helpers
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def log_interaction(user_msg, bot_msg):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} - User: {user_msg} | Bot: {bot_msg}\n"
    try:
        with open(os.path.join(LOG_FOLDER, "chat.log"), "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"Logging failed: {e}")

def check_choice_relevance(user_msg: str) -> bool:
    lw = user_msg.lower()
    return any(kw in lw for kw in CHOICE_RELEVANT_KEYWORDS)

def interpret_feedback(user_msg: str) -> str:
    lw = user_msg.lower()
    if "looks good" in lw or "finish" in lw or "satisfied" in lw:
        return "satisfied"
    elif "modify" in lw or "change" in lw or "improve" in lw:
        return "modify"
    return "unclear"

def save_artifacts():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_code = session.get("current_code", "No code.")
    original_query = session.get("original_query", "Unknown query.")
    data_info = session.get("data_info", {})
    user_lessons = session.get("learning_points", [])

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

# -----------------------------
# Init HF + LoRA
# -----------------------------
def initialize_model():
    global g_model, g_tokenizer
    print(f"[INIT] Loading base model {BASE_MODEL_NAME} with cache_dir={CACHE_DIR} ...")
    g_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        cache_dir=CACHE_DIR
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        cache_dir=CACHE_DIR,
        device_map="auto",
        torch_dtype=torch.float16
    )
    lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    g_model = get_peft_model(base_model, lora_config)
    print("[INIT] Base model + LoRA loaded (in memory).")

# -----------------------------
# Generation Helper
# -----------------------------
def generate_text(prompt: str, temperature=0.7) -> str:
    global g_model, g_tokenizer
    if g_model is None or g_tokenizer is None:
        return "Error: Model not initialized."

    enc = g_tokenizer.encode(prompt, add_special_tokens=False)
    if len(enc) > MAX_INPUT_TOKENS:
        enc = enc[:MAX_INPUT_TOKENS]

    inputs_dict = {
        "input_ids": torch.tensor([enc], dtype=torch.long).to(g_model.device)
    }

    with torch.no_grad():
        out_ids = g_model.generate(
            **inputs_dict,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            pad_token_id=g_tokenizer.eos_token_id
        )

    out_text = g_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    generated = out_text[len(g_tokenizer.decode(enc, skip_special_tokens=True)):].strip()
    return generated

# -----------------------------
# Fine-Tuning Logic
# -----------------------------
def build_training_dataset() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = f"training_{stamp}.jsonl"
    out_path = os.path.join(TRAINING_DATA_FOLDER, out_fname)

    instruct_samples = []

    # references
    for fname in os.listdir(REFERENCES_FOLDER):
        if fname.endswith(".json"):
            fpath = os.path.join(REFERENCES_FOLDER, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as rf:
                    ref_data = json.load(rf)
                title = ref_data.get("title", "")
                paper_text = ref_data.get("paper_text", "")
                code_snip = ref_data.get("code_snippet", "")
                instruction = f"Title: {title}\nPaper:\n{paper_text}\nCode:\n{code_snip}"
                output = "Understood. Knowledge stored."
                instruct_samples.append({"instruction": instruction, "output": output})
            except Exception as e:
                print(f"[WARN] reading ref {fname}: {e}")

    # user feedback
    user_lessons = session.get("learning_points", [])
    for lesson in user_lessons:
        instruct_samples.append({
            "instruction": f"User feedback: {lesson}",
            "output": "We incorporate your feedback."
        })

    if not instruct_samples:
        with open(out_path, "w", encoding="utf-8") as bf:
            pass
        return out_path

    with open(out_path, "w", encoding="utf-8") as ds_f:
        for s in instruct_samples:
            ds_f.write(json.dumps(s) + "\n")
    return out_path

def fine_tune_model() -> str:
    global g_model, g_tokenizer
    ds_file = build_training_dataset()
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
            tok = g_tokenizer(full_text, truncation=True, max_length=512)
            tok["labels"] = tok["input_ids"].copy()
            return tok
        ds_mapped = raw_ds.map(tokenize_fn, batched=False)

        def data_collator(features):
            keys = ["input_ids", "attention_mask", "labels"]
            batch = {k: [f[k] for f in features] for k in keys}
            out = g_tokenizer.pad({"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}, return_tensors="pt")
            labs = g_tokenizer.pad({"input_ids": batch["labels"]}, return_tensors="pt")
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
            model=g_model,
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

# -----------------------------
# Code Generation / Refinement
# -----------------------------
def read_all_references_text() -> str:
    lines = []
    for fn in os.listdir(REFERENCES_FOLDER):
        if fn.endswith(".json"):
            path = os.path.join(REFERENCES_FOLDER, fn)
            try:
                with open(path, "r", encoding="utf-8") as rr:
                    ref_data = json.load(rr)
                title = ref_data.get("title", "")
                paper_txt = ref_data.get("paper_text", "")
                code_txt = ref_data.get("code_snippet", "")
                lines.append(f"--- {fn} ---")
                if title.strip():
                    lines.append(f"Title: {title}")
                if paper_txt.strip():
                    lines.append(f"Paper:\n{paper_txt}")
                if code_txt.strip():
                    lines.append(f"Code:\n{code_txt}")
                lines.append("------------------\n")
            except Exception as e:
                print(f"[WARN] reading {fn}: {e}")
    return "\n".join(lines)

# Revised: We incorporate conversation history here.
def build_combined_context(user_msg: str) -> str:
    # Retrieve conversation from session, including prior user queries and bot responses.
    conversation = session.get("history", [])
    # We'll build a conversation string, limiting to the last N messages to avoid token blowup.
    # Let's keep the last 5 user-bot pairs.

    # read references
    ref_text = read_all_references_text()

    # We'll store up to 5 user-bot pairs in a buffer.
    chat_history_lines = []
    # Because we store them as dicts, each item is {"role": "user" or "assistant", "content": ...}
    # We'll skip if we do not have them in place.
    # We'll only store them if the conversation is in that shape, else fallback.

    # Collect the last 10 messages (which is 5 pairs, user+assistant) if available.
    relevant_history = conversation[-10:]

    for msg_item in relevant_history:
        role = msg_item.get("role", "user")
        content = msg_item.get("content", "")
        if role == "user":
            chat_history_lines.append(f"User: {content}")
        else:
            chat_history_lines.append(f"Assistant: {content}")

    chat_history_str = "\n".join(chat_history_lines)

    # Filter references if user specifically says "mixed logit" in the new message.
    msg_lower = user_msg.lower()
    if "mixed logit" in msg_lower:
        # keep only references containing 'mixed logit'
        ref_text = "\n".join(
            [block for block in ref_text.split("\n------------------\n")
             if "mixed logit" in block.lower()]
        )

    lessons = session.get("learning_points", [])
    lessons_block = ""
    if lessons:
        lessons_block += "\nUser's Past Feedback:\n"
        for i, l in enumerate(lessons, start=1):
            lessons_block += f"{i}. {l}\n"

    # Build final prompt with references, conversation history, and the new request.
    prompt = (
        "You are a specialized choice modeling assistant. Use these references:\n\n"
        f"=== References ===\n{ref_text}\n=== End ===\n\n"
        "Conversation so far:\n"
        f"{chat_history_str}\n\n"
        f"User's new request: {user_msg}\n\n"
        f"{lessons_block}"
    )
    return prompt

def generate_choice_code(user_msg: str) -> str:
    info = session.get("data_info", {})
    fname = info.get("filename", "N/A")
    cstr = ", ".join(info.get("columns", []))
    desc = info.get("description", "")

    context = build_combined_context(user_msg)
    final_prompt = (
        f"{context}\n\n"
        f"(Data Info: filename='{fname}', columns=[{cstr}], desc='{desc}')\n"
        "Generate Python code + short explanation for this discrete choice modeling request."
    )
    return generate_text(final_prompt)

def refine_code(original_code: str, user_feedback: str) -> str:
    lessons = session.get("learning_points", [])
    lessons.append(f"Refinement request: {user_feedback}")
    session["learning_points"] = lessons

    context = build_combined_context(user_feedback)
    prompt = (
        f"{context}\n\n"
        f"Current code:\n```python\n{original_code}\n```\n"
        "Refine/improve the code based on user feedback."
    )
    return generate_text(prompt)

# -----------------------------
# Conversation Flow
# -----------------------------
def process_chat(user_msg: str) -> str:
    # We'll store conversation in session as a list of dicts: {"role": "user" or "assistant", "content": ...}
    # 1) Retrieve current conversation from session
    conversation = session.get("history", [])
    if not isinstance(conversation, list):
        conversation = []

    # 2) Append this user message
    conversation.append({"role": "user", "content": user_msg})
    session["history"] = conversation

    conv_state = session.get("conv_state")

    # 3) if we are awaiting feedback
    if conv_state == "AWAITING_FEEDBACK":
        decision = interpret_feedback(user_msg)
        if decision == "satisfied":
            artifact_msg = save_artifacts()
            ft_msg = fine_tune_model()
            session.clear()  # clears entire session
            # We'll keep the conversation though if we want? Let's just clear all for now.
            bot_resp = f"Great! {artifact_msg}\n{ft_msg}\nYou can start a new request."
            # store in conversation
            conversation.append({"role": "assistant", "content": bot_resp})
            session["history"] = conversation
            return bot_resp
        elif decision == "modify":
            orig_code = session.get("current_code", "")
            new_code = refine_code(orig_code, user_msg)
            session["current_code"] = new_code
            ft_msg = fine_tune_model()
            bot_resp = (
                f"Refined code:\n\n{new_code}\n\n"
                f"Fine-tune result: {ft_msg}\n"
                "Anything else? (Say 'Looks good' or describe changes.)"
            )
            conversation.append({"role": "assistant", "content": bot_resp})
            session["history"] = conversation
            return bot_resp
        else:
            bot_resp = "Not sure if you're done or want changes. Please say 'Looks good' or 'modify'."
            conversation.append({"role": "assistant", "content": bot_resp})
            session["history"] = conversation
            return bot_resp

    # 4) If user specifically asks for a known model: e.g. 'mixed logit'
    if "mixed logit" in user_msg.lower():
        mixed_code = (
            "# Mixed Logit in Biogeme\n"
            "from biogeme import biogeme\n"
            "from biogeme.expressions import Beta, RandomVariable, Integrate\n"
            "from biogeme.models import loglogit\n\n"
            "B_TIME = Beta('B_TIME', 0, None, None, 0)\n"
            "B_COST = Beta('B_COST', 0, None, None, 0)\n"
            "omega = RandomVariable('omega')\n"
            "B_TIME_RND = B_TIME + omega\n\n"
            "# Define utility functions based on uploaded data\n"
            "# For example:\n"
            "# U_TRAIN = B_TIME_RND * TRAIN_TT + B_COST * TRAIN_COST\n"
            "# U_CAR = B_TIME_RND * CAR_TT + B_COST * CAR_COST\n"
            "# logprob = loglogit(V, av, CHOICE)\n"
        )
        session["original_query"] = user_msg
        session["current_code"] = mixed_code
        session["conv_state"] = "AWAITING_FEEDBACK"
        bot_resp = (
            f"Here is the adapted mixed logit template based on your request:\n\n```python\n{mixed_code}\n```\n\n"
            "What do you think? (Say 'Looks good' or describe changes.)"
        )
        conversation.append({"role": "assistant", "content": bot_resp})
        session["history"] = conversation
        return bot_resp

    # 5) If not specifically recognized, check if it's relevant to DCM
    if not check_choice_relevance(user_msg):
        bot_resp = "We only handle choice modeling queries. Please focus on discrete choice."
        conversation.append({"role": "assistant", "content": bot_resp})
        session["history"] = conversation
        return bot_resp

    # 6) For general DCM queries
    session["original_query"] = user_msg
    last_up = uploaded_data.get("last_upload", {})
    if "df" in last_up and isinstance(last_up["df"], pd.DataFrame):
        session["data_info"] = {
            "filename": last_up["filename"],
            "columns": last_up["df"].columns.tolist(),
            "description": last_up.get("description", "")
        }
    else:
        session["data_info"] = {}

    code_snip = generate_choice_code(user_msg)
    session["current_code"] = code_snip
    session["conv_state"] = "AWAITING_FEEDBACK"

    bot_resp = (
        f"Here is some code:\n\n{code_snip}\n\n"
        "What do you think? (Say 'Looks good' or describe changes.)"
    )
    conversation.append({"role": "assistant", "content": bot_resp})
    session["history"] = conversation
    return bot_resp

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index1():
    feedback = None
    file_feedback = None
    desc_feedback = None
    columns_list = []
    current_description = ""
    current_filename = None

    if "last_upload" in uploaded_data:
        up = uploaded_data["last_upload"]
        if "df" in up and isinstance(up["df"], pd.DataFrame):
            columns_list = up["df"].columns.tolist()
        current_description = up.get("description", "")
        current_filename = up.get("filename")

    if request.method == "POST":
        action = request.form.get("action")
        if action == "upload_file":
            f = request.files.get("file")
            if f and f.filename and allowed_file(f.filename):
                session.clear()
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
                        file_feedback = f"Error: '{fname}' insufficient rows."
                    else:
                        uploaded_data["last_upload"] = {
                            "filename": fname,
                            "df": df,
                            "description": ""
                        }
                        columns_list = df.columns.tolist()
                        current_filename = fname
                        file_feedback = f"File '{fname}' uploaded. Rows={df.shape[0]}, Cols={df.shape[1]}"
                except Exception as e:
                    file_feedback = f"Error reading '{fname}': {e}"
            else:
                file_feedback = "No valid CSV/XLSX file provided."

        elif action == "save_description":
            dtext = request.form.get("data_description", "").strip()
            if "last_upload" in uploaded_data:
                uploaded_data["last_upload"]["description"] = dtext
                current_description = dtext
                desc_feedback = "Data description saved."
                if "data_info" in session:
                    session["data_info"]["description"] = dtext
                    session.modified = True
            else:
                desc_feedback = "Please upload data first."

        elif action == "send_message":
            user_msg = request.form.get("message", "").strip()
            if not user_msg:
                feedback = "Please enter a message."
            else:
                bot_resp = process_chat(user_msg)
                feedback = bot_resp
                log_interaction(user_msg, bot_resp)

    return render_template(
        "index1.html",
        feedback=feedback,
        file_feedback=file_feedback,
        desc_feedback=desc_feedback,
        columns=columns_list,
        description=current_description,
        current_filename=current_filename
    )

@app.route("/references", methods=["GET", "POST"])
def references():
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
            ref_obj = {
                "title": ref_title,
                "paper_text": paper_txt,
                "code_snippet": code_txt
            }
            try:
                with open(r_path, "w", encoding="utf-8") as r_f:
                    json.dump(ref_obj, r_f, indent=4)
                ft_res = fine_tune_model()
                msg = f"Thank you! Your reference was saved. Fine-tune result: {ft_res}"
            except Exception as e:
                msg = f"Error saving reference: {e}"

    return render_template("references1.html", msg=msg)

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json() or {}
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"response": "Please enter a message."})
    bot_resp = process_chat(user_msg)
    log_interaction(f"[AJAX] {user_msg}", bot_resp)
    return jsonify({"response": bot_resp})

if __name__ == "__main__":
    print("[MAIN] Initializing HF model for LoRA fine-tuning once (cache_dir used).")
    initialize_model()

    print("[MAIN] Starting Flask on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)

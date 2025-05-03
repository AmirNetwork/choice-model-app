# app_custom_bot.py
# -*- coding: utf-8 -*-
import os
import re
import json
import traceback
import subprocess
from datetime import datetime

import pandas as pd
from flask import Flask, request, render_template, session, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
UPLOAD_FOLDER = os.path.join(STORAGE_DIR, "uploads")
LOG_FOLDER = os.path.join(STORAGE_DIR, "logs")
COMPLETED_MODELS_FOLDER = os.path.join(STORAGE_DIR, "completed_models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(COMPLETED_MODELS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx"}

# We only handle "choice modeling" queries
CHOICE_RELEVANT_KEYWORDS = [
    "choice model", "discrete choice", "logit", "mixed logit", "conjoint",
    "preference", "utility", "random utility", "multinomial", "latent class",
    "nested logit", "probit", "mlogit", "market share", "willingness to pay", "wtp"
]

# Adjust if you want a longer/shorter max time for the LLM to respond
LLM_TIMEOUT_SECS = 180  # e.g., 3 minutes

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# We'll store the user's dataset in this global
uploaded_data = {}

# A dynamic knowledge store for references/papers/codes
# Instead of a static dict, we store them in memory with incremental IDs
knowledge_store = {}
next_ref_id = 1  # increment each time a user adds a new reference


# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def log_interaction(user_msg, bot_msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{ts} - User: {user_msg} | Bot: {bot_msg}\n"
    log_path = os.path.join(LOG_FOLDER, "chat.log")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as e:
        print(f"Logging failed: {e}")

def deepseek_call(prompt_text: str) -> str:
    """
    Actually call DeepSeek LLM via Ollama with extended timeout.
    """
    command = "ollama run deepseek-r1:7b"
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            encoding='utf-8',
            errors='replace'
        )
        stdout, stderr = proc.communicate(input=prompt_text + "\n", timeout=LLM_TIMEOUT_SECS)

        if proc.returncode != 0:
            error_message = stderr.strip() or f"Ollama command failed (code {proc.returncode})"
            if "connection refused" in error_message.lower():
                return "Error: Cannot connect to Ollama service."
            return f"Error calling DeepSeek (Ollama): {error_message}"

        # Remove ANSI codes if any
        resp = re.sub(r'\x1b\[\d{1,3}m', '', stdout.strip())
        return resp

    except subprocess.TimeoutExpired:
        return f"Error: The request to the language model timed out after {LLM_TIMEOUT_SECS} seconds."
    except FileNotFoundError:
        return "Error: Ollama command not found."
    except Exception as e:
        return f"Error running LLM: {e}\n{traceback.format_exc()}"

def check_choice_relevance(user_msg: str) -> bool:
    """Quick check if user_msg is about choice modeling."""
    lower_msg = user_msg.lower()
    if any(kw in lower_msg for kw in CHOICE_RELEVANT_KEYWORDS):
        return True

    # Otherwise, do a quick LLM check
    prompt = f'Is the user request about choice modeling? Answer "Yes" or "No".\nUser request: "{user_msg}"'
    ans = deepseek_call(prompt)
    ans_lower = ans.strip().lower()
    return ("yes" in ans_lower) and ("no" not in ans_lower)

def interpret_feedback(user_msg: str) -> str:
    """
    We have code. The user responded. 
    Ask LLM if user is 'satisfied', 'modify', or 'unclear'.
    """
    prompt = (
        f"User was shown some choice model code/explanation. They said:\n"
        f"'{user_msg}'\n"
        f"Do they want 'modify', are they 'satisfied', or is it 'unclear'?\n"
        f"Answer exactly one word: 'modify', 'satisfied', or 'unclear'."
    )
    ans = deepseek_call(prompt).lower()
    if "satisfied" in ans:
        return "satisfied"
    elif "modify" in ans:
        return "modify"
    return "unclear"

def save_artifacts():
    """
    Save the final code snippet once user is satisfied.
    """
    final_code = session.get("current_code", "No code.")
    original_query = session.get("original_query", "Unknown query.")
    data_info = session.get("data_info", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    learning_points = session.get("learning_points", [])

    artifact = {
        "timestamp": datetime.now().isoformat(),
        "original_query": original_query,
        "final_code": final_code,
        "data_info": data_info,
        "learning_points": learning_points
    }
    fname = f"completion_{timestamp}.json"
    fpath = os.path.join(COMPLETED_MODELS_FOLDER, fname)

    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=4)
        return f"Process details saved as {fname}."
    except Exception as e:
        print("Error saving artifacts:", e)
        return "Error saving artifacts."

# -----------------------------
# Build a combined context from:
# 1) knowledge_store references
# 2) user feedback/lessons
# 3) user request
# so the LLM sees all relevant info
# -----------------------------
def build_combined_context(user_msg: str) -> str:
    """
    1) Summarize all references from knowledge_store
    2) Summarize user 'learning_points'
    3) Then the user's actual request
    """
    # 1) Summarize references
    references_text = ""
    for ref_id, ref_data in knowledge_store.items():
        title = ref_data["title"]
        paper = ref_data["paper_text"]
        code_snippet = ref_data["code_snippet"]
        references_text += f"--- Reference {ref_id} ---\n"
        references_text += f"Title: {title}\n"
        references_text += f"Paper/Description:\n{paper}\n\n"
        if code_snippet.strip():
            references_text += f"Associated Code:\n{code_snippet}\n"
        references_text += "------------------------\n\n"

    # 2) Summarize user feedback / lessons
    user_lessons = session.get("learning_points", [])
    if user_lessons:
        references_text += "User's Past Feedback / Lessons:\n"
        for idx, lesson in enumerate(user_lessons, start=1):
            references_text += f"{idx}. {lesson}\n"
        references_text += "\n"

    # 3) Combine with user request
    combined_prompt = (
        f"You are a specialized choice modeling chatbot. Use these references for context.\n\n"
        f"=== Knowledge Store ===\n"
        f"{references_text}\n"
        f"=== End Knowledge ===\n\n"
        f"User Request: {user_msg}"
    )
    return combined_prompt

def generate_choice_code(user_msg: str) -> str:
    """
    Merge references + user lessons + user request => single prompt => DeepSeek => code
    """
    data_info = session.get("data_info", {})
    fname = data_info.get("filename", "N/A")
    columns = data_info.get("columns", [])
    desc = data_info.get("description", "")
    col_str = ", ".join(columns)

    context_text = build_combined_context(user_msg)
    # Then add data context to the final prompt
    final_prompt = (
        f"{context_text}\n\n"
        f"Data info: filename='{fname}', columns=[{col_str}], description='{desc}'.\n"
        f"Please produce Python code (plus short explanation) for this choice modeling request."
    )
    ans = deepseek_call(final_prompt)
    return ans

def refine_code(original_code: str, user_feedback: str) -> str:
    """
    Incorporate user feedback + knowledge store.
    Add the user feedback as a new 'learning point', so the chatbot "improves."
    """
    # store the user feedback as a 'learning point'
    new_points = session.get("learning_points", [])
    new_points.append(f"Refinement request: {user_feedback}")
    session["learning_points"] = new_points

    context_text = build_combined_context(user_feedback)
    final_prompt = (
        f"{context_text}\n\n"
        f"Current code:\n```python\n{original_code}\n```\n"
        f"Please refine the code based on the user feedback above."
    )
    return deepseek_call(final_prompt)


# -----------------------------
# Chat Flow
# -----------------------------
def process_chat(user_msg: str) -> str:
    """
    - If session['conv_state'] = 'AWAITING_FEEDBACK': interpret => finalize or refine
    - Else => new query => check if choice relevant => produce code => store => ask for feedback
    """
    conv_state = session.get("conv_state")

    if conv_state == "AWAITING_FEEDBACK":
        decision = interpret_feedback(user_msg)
        if decision == "satisfied":
            saved_msg = save_artifacts()
            session.clear()
            return f"Great! {saved_msg} You can start a new request now."
        elif decision == "modify":
            original_code = session.get("current_code", "")
            refined = refine_code(original_code, user_msg)
            session["current_code"] = refined
            return (
                f"Here's the refined code:\n\n{refined}\n\n"
                "What do you think now? (Say 'Looks good' or more changes?)"
            )
        else:
            return "Not sure if you're done or want changes. Please clarify: 'Looks good' or how to modify."

    # Otherwise it's a new query
    if not check_choice_relevance(user_msg):
        return "We only handle choice modeling requests. Please focus on discrete choice analysis."

    session["original_query"] = user_msg
    # If user uploaded data, store it
    last_up = uploaded_data.get("last_upload", {})
    if "df" in last_up and isinstance(last_up["df"], pd.DataFrame):
        session["data_info"] = {
            "filename": last_up["filename"],
            "columns": last_up["df"].columns.tolist(),
            "description": last_up.get("description", "")
        }
    else:
        session["data_info"] = {}

    code_snippet = generate_choice_code(user_msg)
    session["current_code"] = code_snippet
    session["conv_state"] = "AWAITING_FEEDBACK"

    return (
        f"Here is some code:\n\n{code_snippet}\n\n"
        "What do you think? (Say 'Looks good' or describe changes, then we refine further.)"
    )


# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    """
    Main page: 
      - upload CSV/XLSX
      - add data description
      - chat with the bot
    """
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
                # new file => clear old conversation
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
                        file_feedback = f"Error: '{fname}' has insufficient rows (<2)."
                    else:
                        uploaded_data["last_upload"] = {
                            "filename": fname,
                            "df": df,
                            "description": ""
                        }
                        columns_list = df.columns.tolist()
                        current_filename = fname
                        file_feedback = f"File '{fname}' uploaded successfully (Rows={df.shape[0]}, Cols={df.shape[1]})"
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
                # store in session if it exists
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
    """
    A new page to add or view references (papers/codes).
    These will be used to 'train' the bot on top of DeepSeek.
    """
    global next_ref_id

    msg = None
    if request.method == "POST":
        # user wants to add a new reference
        title = request.form.get("ref_title", "").strip()
        paper_text = request.form.get("ref_paper", "").strip()
        code_snippet = request.form.get("ref_code", "").strip()

        if not title and not paper_text and not code_snippet:
            msg = "Please provide at least a title or some content."
        else:
            # store it
            knowledge_store[next_ref_id] = {
                "title": title,
                "paper_text": paper_text,
                "code_snippet": code_snippet
            }
            next_ref_id += 1
            msg = "Reference added successfully!"

    # show references
    current_refs = [
        {"ref_id": rid, "title": rdata["title"], "paper_text": rdata["paper_text"], "code_snippet": rdata["code_snippet"]}
        for rid, rdata in knowledge_store.items()
    ]
    return render_template("references.html", msg=msg, references=current_refs)

@app.route("/chat", methods=["POST"])
def chat():
    """
    If you want to handle conversation with AJAX, call /chat with JSON {message: "..."}
    """
    data = request.get_json() or {}
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"response": "Please enter a message."})
    bot_resp = process_chat(user_msg)
    log_interaction(f"[AJAX] {user_msg}", bot_resp)
    return jsonify({"response": bot_resp})


if __name__ == "__main__":
    for folder in [UPLOAD_FOLDER, LOG_FOLDER, COMPLETED_MODELS_FOLDER, "templates", "static"]:
        os.makedirs(folder, exist_ok=True)

    tpl_main = os.path.join("templates", "index1.html")
    tpl_refs = os.path.join("templates", "references.html")
    if not os.path.exists(tpl_main):
        print(f"WARNING: {tpl_main} not found. The main page won't render properly.")
    if not os.path.exists(tpl_refs):
        print(f"WARNING: {tpl_refs} not found. The references page won't render properly.")

    print(f"Starting Custom ChatBot on http://127.0.0.1:5000 (LLM timeout={LLM_TIMEOUT_SECS}s)")
    app.run(debug=True, host="127.0.0.1", port=5000)

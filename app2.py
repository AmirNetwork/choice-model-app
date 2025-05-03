# app_custom_bot.py
# -*- coding: utf-8 -*-
import os
import re
import json
import traceback
import subprocess
from datetime import datetime

import pandas as pd
from flask import Flask, request, render_template, session, jsonify
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
CHOICE_RELEVANT_KEYWORDS = [
    "choice model", "discrete choice", "logit", "mixed logit", "conjoint",
    "preference", "utility", "random utility", "multinomial", "latent class",
    "nested logit", "probit", "mlogit", "market share", "willingness to pay", "wtp"
]

LLM_TIMEOUT_SECS = 180  # Up to 3 minutes

# -----------------------------
# Flask Init
# -----------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)

# We'll keep the uploaded dataset in this global
uploaded_data = {}

# -----------------------------
# A minimal "Knowledge Store"
# for custom references (papers, advanced codes, etc.)
# You can store actual text or partial code snippets:
# e.g. knowledge_store["paper_1"] = "Excerpts from an advanced random-coefficient approach..."
# or knowledge_store["bio_mixed_code"] = "Some specialized code snippet..."
# For demonstration, we store placeholders.
knowledge_store = {
    "paper_1": """
Title: "Advanced Mixed Logit Theory"
Excerpt: Random-coefficient logit models incorporate taste heterogeneity...
(Imagine more text/pseudocode here)
""",
    "paper_2": """
Title: "Latest expansions on Hybrid Choice Models"
Excerpt: Hybrid models combine latent variables with discrete choice...
(Imagine more text/pseudocode here)
"""
}
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

def check_choice_relevance(user_msg: str) -> bool:
    """Check if user_msg is about choice modeling."""
    lower_msg = user_msg.lower()
    if any(kw in lower_msg for kw in CHOICE_RELEVANT_KEYWORDS):
        return True

    # If no direct match, we can ask LLM quickly:
    prompt = f'Is this query about choice modeling? Answer "Yes" or "No".\nUser request: "{user_msg}"'
    ans = deepseek_call(prompt)
    ans_lower = ans.strip().lower()
    return ("yes" in ans_lower) and ("no" not in ans_lower)

def deepseek_call(prompt_text: str) -> str:
    """
    Actually call DeepSeek via Ollama, with extended timeout.
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

        # Remove ANSI codes
        resp = re.sub(r'\x1b\[\d{1,3}m', '', stdout.strip())
        return resp

    except subprocess.TimeoutExpired:
        return f"Error: The request to the language model timed out after {LLM_TIMEOUT_SECS} seconds."
    except FileNotFoundError:
        return "Error: Ollama command not found."
    except Exception as e:
        return f"Error running LLM: {e}\n{traceback.format_exc()}"

def interpret_feedback(user_msg: str) -> str:
    """
    Prompt the LLM to interpret user feedback as "modify", "satisfied", or "unclear".
    """
    prompt = (
        f"User was shown code/explanation for a choice model. They said:\n"
        f"'{user_msg}'\n\n"
        f"Do they want 'modify', are they 'satisfied', or is it 'unclear'?\n"
        f"Answer exactly one word: 'modify', 'satisfied', or 'unclear'."
    )
    ans = deepseek_call(prompt).lower()
    if "satisfied" in ans:
        return "satisfied"
    elif "modify" in ans:
        return "modify"
    else:
        return "unclear"

def save_artifacts():
    """
    Once user is satisfied, we store the final code in a JSON for future reference.
    """
    final_code = session.get("current_code", "No code.")
    original_query = session.get("original_query", "")
    data_info = session.get("data_info", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifact = {
        "timestamp": datetime.now().isoformat(),
        "original_query": original_query,
        "final_code": final_code,
        "data_info": data_info,
        "learning_points": session.get("learning_points", [])
    }
    fname = f"completion_{timestamp}.json"
    fpath = os.path.join(COMPLETED_MODELS_FOLDER, fname)
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=4)
        return f"Process details saved as {fname}."
    except Exception as e:
        print("Error saving artifact:", e)
        return "Error saving final details."

# -----------------------------
# Merging custom references + user feedback
# to create a "context" for the LLM
# -----------------------------
def build_combined_context(user_msg: str) -> str:
    """
    1. Summarize what's in knowledge_store (like advanced papers/codes).
    2. Summarize user 'learning_points' from session (things we've gleaned).
    3. Then append the user's actual request.
    """
    # 1) Summarize knowledge
    # For demonstration, we just join them. In a real system, you might do semantic retrieval.
    knowledge_text = ""
    for kname, ktext in knowledge_store.items():
        knowledge_text += f"[{kname}]: {ktext}\n\n"

    # 2) Summarize user learning points
    learning_points = session.get("learning_points", [])
    if learning_points:
        knowledge_text += "User Feedback / Lessons Learned:\n"
        for i, point in enumerate(learning_points):
            knowledge_text += f"{i+1}. {point}\n"
        knowledge_text += "\n"

    # 3) Combine with user request
    combined_prompt = (
        f"You are a custom choice modeling chatbot. Use the references below to answer.\n\n"
        f"=== Custom Knowledge & Papers ===\n"
        f"{knowledge_text}\n"
        f"=== End Knowledge ===\n\n"
        f"User request: {user_msg}"
    )
    return combined_prompt

def generate_choice_code(user_msg: str) -> str:
    """
    1) Build context with knowledge_store + prior user lessons
    2) Send to DeepSeek
    3) Return the code snippet
    """
    data_info = session.get("data_info", {})
    fname = data_info.get("filename", "N/A")
    col_str = ", ".join(data_info.get("columns", []))
    desc = data_info.get("description", "")

    # Build context from references + user feedback
    context_text = build_combined_context(user_msg)
    # Then we add the data context
    final_prompt = (
        f"{context_text}\n\n"
        f"Data info: filename='{fname}', columns=[{col_str}], description='{desc}'.\n"
        f"Please produce a Python code snippet (and short explanation) for this choice model request."
    )
    ans = deepseek_call(final_prompt)
    return ans

def refine_code(original_code: str, user_feedback: str) -> str:
    """
    Incorporate user feedback + knowledge store for refinement.
    Also store the "lesson learned" in session so the bot 'improves' next time.
    """
    # store the user feedback as a 'learning point'
    new_points = session.get("learning_points", [])
    new_points.append(f"Refinement request: {user_feedback}")
    session["learning_points"] = new_points

    context_text = build_combined_context(user_feedback)
    final_prompt = (
        f"{context_text}\n\n"
        f"Current code:\n```python\n{original_code}\n```\n"
        f"Based on the user feedback above, refine the code."
    )
    return deepseek_call(final_prompt)

# -----------------------------
# Chat Flow
# -----------------------------
def process_chat(user_msg: str) -> str:
    """
    - If session['conv_state'] == 'AWAITING_FEEDBACK': interpret feedback => finalize or refine.
    - Else => new request => check choice relevance => produce new code => store => ask for feedback
    """
    conv_state = session.get("conv_state")

    if conv_state == "AWAITING_FEEDBACK":
        decision = interpret_feedback(user_msg)
        if decision == "satisfied":
            # finalize
            msg = save_artifacts()
            session.clear()
            return f"Great! {msg} You can start a new request now."
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

    # New request (or conversation state is None)
    # 1) Check if relevant
    if not check_choice_relevance(user_msg):
        return "We only handle choice modeling requests. Provide a request about discrete/choice modeling."

    # 2) store original query
    session["original_query"] = user_msg

    # 3) ensure data_info if we have an uploaded dataset
    last_up = uploaded_data.get("last_upload", {})
    if "df" in last_up and isinstance(last_up["df"], pd.DataFrame):
        session["data_info"] = {
            "filename": last_up.get("filename", "N/A"),
            "columns": last_up["df"].columns.tolist(),
            "description": last_up.get("description", "")
        }
    else:
        session["data_info"] = {}

    # 4) generate code
    code_snippet = generate_choice_code(user_msg)
    session["current_code"] = code_snippet
    session["conv_state"] = "AWAITING_FEEDBACK"

    return (
        f"Here is some code:\n\n{code_snippet}\n\n"
        "What do you think? (Say 'Looks good' or describe changes.)"
    )

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
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
                file_feedback = "No valid CSV/XLSX file selected."

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
        "index.html",
        feedback=feedback,
        file_feedback=file_feedback,
        desc_feedback=desc_feedback,
        columns=columns_list,
        description=current_description,
        current_filename=current_filename
    )

@app.route("/chat", methods=["POST"])
def chat():
    """Optional AJAX route."""
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

    tpl_path = os.path.join("templates", "index.html")
    if not os.path.exists(tpl_path):
        print(f"WARNING: No {tpl_path} found. The page won't render fully.")
    print(f"Starting Custom ChatBot on http://127.0.0.1:5000 (LLM timeout={LLM_TIMEOUT_SECS}s)")
    app.run(debug=True, host="127.0.0.1", port=5000)

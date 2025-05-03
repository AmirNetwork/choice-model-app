import os
import logging
from logging.handlers import RotatingFileHandler
from flask import (
    Flask, request, render_template, session, jsonify, redirect, url_for, flash
)
from werkzeug.utils import secure_filename
from sqlalchemy.orm import Session
from celery import Celery
from redis import Redis
from config import Config
from models.llm import LLMManager
from models.finetune import fine_tune_model
from utils.auth import User, SessionLocal, login_required
from utils.data import allowed_file, read_data_file, get_data_preview, analyze_choice_data
from utils.tasks import generate_response, redis_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler(os.path.join(Config.LOG_FOLDER, "app.log"), maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# Initialize Flask
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.from_object(Config)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

# Initialize extensions
llm_manager = LLMManager()
user_data = {}  # In-memory for now; replace with Redis in production

# Ensure directories exist
for folder in [
    Config.UPLOAD_FOLDER, Config.LOG_FOLDER, Config.COMPLETED_MODELS_FOLDER,
    Config.REFERENCES_FOLDER, Config.TRAINING_DATA_FOLDER, Config.TRAINED_REFERENCES_FOLDER, Config.CACHE_DIR
]:
    os.makedirs(folder, exist_ok=True)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()
        if not username or not password:
            flash("Username and password required.", "error")
            return render_template("signup.html")
        with SessionLocal() as db:
            if db.query(User).filter_by(username=username).first():
                flash("Username exists.", "error")
                return render_template("signup.html")
            hashed = hash_password(password)
            user = User(username=username, password=hashed)
            db.add(user)
            db.commit()
            logger.info(f"User {username} signed up")
            return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()
        with SessionLocal() as db:
            user = db.query(User).filter_by(username=username).first()
            if not user:
                flash("Invalid username.", "error")
                return render_template("login.html")
            if not verify_password(password, user.password):
                flash("Incorrect password.", "error")
                return render_template("login.html")
            session["user_id"] = str(user.id)
            session["username"] = username
            logger.info(f"User {username} logged in")
            return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    user_id = session.get("user_id")
    session.clear()
    if user_id:
        logger.info(f"User {user_id} logged out")
    return redirect(url_for("login"))

@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("login"))

@app.route("/app", methods=["GET", "POST"])
@login_required
def index():
    user_id = session["user_id"]
    username = session.get("username", "Unknown")
    user_data.setdefault(user_id, {"history": [], "learning_points": [], "data_info": {}, "conv_state": None})
    udata = user_data[user_id]
    data_info = udata.get("data_info", {})
    columns = data_info.get("df", pd.DataFrame()).columns.tolist()
    description = data_info.get("description", "")
    filename = data_info.get("filename", "")
    choice_analysis = analyze_choice_data(data_info.get("df", pd.DataFrame())) if "df" in data_info else {}

    if request.method == "POST":
        action = request.form.get("action")
        if action == "upload_file":
            file = request.files.get("file")
            if file and allowed_file(file.filename):
                fname = secure_filename(file.filename)
                fpath = os.path.join(Config.UPLOAD_FOLDER, fname)
                try:
                    file.save(fpath)
                    df = read_data_file(fpath, fname)
                    if df.empty or df.shape[0] < 2:
                        flash(f"File {fname} has insufficient data.", "error")
                    else:
                        data_info.update({"filename": fname, "df": df, "description": ""})
                        udata["data_info"] = data_info
                        udata["history"] = []
                        flash(f"Uploaded {fname}: {df.shape[0]} rows, {df.shape[1]} cols.", "success")
                        logger.info(f"User {user_id} uploaded {fname}")
                except Exception as e:
                    flash(f"Error uploading {fname}: {e}", "error")
            else:
                flash("Invalid file. Use CSV or XLSX.", "error")
        elif action == "save_description":
            description = request.form.get("data_description", "").strip()
            if "df" in data_info:
                data_info["description"] = description
                udata["data_info"] = data_info
                flash("Description saved.", "success")
                logger.info(f"User {user_id} saved description")
            else:
                flash("Upload data first.", "error")
        elif action == "send_message":
            message = request.form.get("message", "").strip()
            if not message:
                flash("Enter a message.", "error")
            else:
                udata["history"].append({"role": "user", "content": message})
                task = generate_response.delay(message, user_id, user_data)
                flash(f"Processing request. Task ID: {task.id}", "info")
                logger.info(f"User {user_id} submitted task {task.id}")
        user_data[user_id] = udata
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        username=username,
        chat_history=udata.get("history", []),
        columns=columns,
        description=description,
        current_filename=filename,
        choice_analysis=choice_analysis
    )

@app.route("/task_status", methods=["GET"])
@login_required
def task_status():
    task_id = request.args.get("task_id")
    if not task_id:
        return jsonify({"status": "error", "message": "No task ID."})
    task = generate_response.AsyncResult(task_id)
    if task.state == "PENDING":
        return jsonify({"status": "pending"})
    elif task.state == "SUCCESS":
        result = task.result
        user_id = session["user_id"]
        user_data[user_id]["history"].append({"role": "assistant", "content": result})
        return jsonify({"status": "success", "result": result})
    else:
        return jsonify({"status": task.state, "message": str(task.info)})

@app.route("/references", methods=["GET", "POST"])
@login_required
def references():
    user_id = session["user_id"]
    if request.method == "POST":
        ref_title = request.form.get("ref_title", "").strip()
        paper_txt = request.form.get("ref_paper", "").strip()
        code_txt = request.form.get("ref_code", "").strip()
        if not (ref_title or paper_txt or code_txt):
            flash("Provide reference details.", "error")
        else:
            from datetime import datetime
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"ref_{stamp}.json"
            fpath = os.path.join(Config.REFERENCES_FOLDER, fname)
            ref = {"title": ref_title, "paper_text": paper_txt, "code_snippet": code_txt}
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(ref, f, indent=4)
                ft_result = fine_tune_model(user_id, user_data, llm_manager)
                flash(f"Reference saved. Fine-tuning: {ft_result}", "success")
                logger.info(f"User {user_id} added reference {fname}")
            except Exception as e:
                flash(f"Error saving reference: {e}", "error")
    return render_template("references.html")

if __name__ == "__main__":
    logger.info("Starting Flask app")
    app.run(debug=False, host="127.0.0.1", port=5000)
# app1.py
# -*- coding: utf-8 -*-
import os
import subprocess
import re
from flask import Flask, request, render_template, jsonify, url_for, session
import json
import pandas as pd
from werkzeug.utils import secure_filename
from datetime import datetime
import traceback

# --- Configuration ---
UPLOAD_FOLDER = os.path.join(os.getcwd(), "storage", "uploads")
LOG_FOLDER = os.path.join(os.getcwd(), "storage", "logs")
COMPLETED_MODELS_FOLDER = os.path.join(os.getcwd(), "storage", "completed_models")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(COMPLETED_MODELS_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"xlsx", "csv"}

# General Keywords for relevance check
CHOICE_MODELING_KEYWORDS = [
    "choice model", "logit", "multinomial", "discrete choice",
    "conjoint analysis", "preference", "utility", "willingness to pay", "wtp",
    "market share simulation", "attribute", "level", "respondent", "stated preference",
    "revealed preference", "mixed logit", "latent class", "nested logit",
    "random forest", "gradient boosting", "neural network", "machine learning",
    "feature importance", "predict choice", "classify choice", "model specification"
]

# Specific list of TRADITIONAL model types
TRADITIONAL_MODEL_KEYWORDS = [
    "mixed logit", "multinomial logit", "mnl", "conditional logit",
    "nested logit", "latent class logit", "probit"
]

# Data-Driven Model Sample Code (Keys must be lowercase)
DATA_DRIVEN_SAMPLE_CODE = {
    "random forest": """
# Random Forest Classifier Example
# ... (Full code snippet) ...
import pandas as pd; from sklearn.model_selection import train_test_split; from sklearn.ensemble import RandomForestClassifier; from sklearn.metrics import accuracy_score, classification_report, confusion_matrix; from sklearn.preprocessing import LabelEncoder; import numpy as np
# Assuming df, target_variable, feature_columns defined
X = df[feature_columns].copy(); y = df[target_variable].copy()
if not pd.api.types.is_numeric_dtype(y): le = LabelEncoder(); y = le.fit_transform(y)
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if not categorical_cols.empty: X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
rf_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', max_depth=10, min_samples_split=5); rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test); accuracy = accuracy_score(y_test, y_pred); report = classification_report(y_test, y_pred); conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}"); print("Report:\\n", report); print("Matrix:\\n", conf_matrix)
importances = rf_model.feature_importances_; feature_names = X.columns; feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False).reset_index(drop=True)
print("Importances:\\n", feature_importance_df.head(10))
""",
    "gradient boosting": """
# Gradient Boosting Classifier Example
# ... (Full code snippet) ...
import pandas as pd; from sklearn.model_selection import train_test_split; from sklearn.ensemble import GradientBoostingClassifier; from sklearn.metrics import accuracy_score, classification_report; from sklearn.preprocessing import LabelEncoder
X = df[feature_columns].copy(); y = df[target_variable].copy()
if not pd.api.types.is_numeric_dtype(y): le = LabelEncoder(); y = le.fit_transform(y)
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if not categorical_cols.empty: X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, subsample=0.8); gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test); print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}"); print("Report:\\n", classification_report(y_test, y_pred))
importances = gb_model.feature_importances_; feature_names = X.columns; feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False).reset_index(drop=True)
print("Importances:\\n", feature_importance_df.head(10))
""",
    "neural network": """
# Neural Network Example (using Keras/TensorFlow)
# ... (Full code snippet, ensure TensorFlow is importable) ...
import pandas as pd; import numpy as np; from sklearn.model_selection import train_test_split; from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder; from sklearn.compose import ColumnTransformer; from sklearn.pipeline import Pipeline; import tensorflow as tf; from tensorflow import keras; from tensorflow.keras import layers
X = df[feature_columns].copy(); y = df[target_variable].copy()
if not pd.api.types.is_numeric_dtype(y): le = LabelEncoder(); y = le.fit_transform(y); num_classes = len(le.classes_)
else: num_classes = int(y.max() + 1)
numeric_features = X.select_dtypes(include=np.number).columns.tolist(); categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
numeric_transformer=Pipeline(steps=[('scaler', StandardScaler())]); categorical_transformer=Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor=ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)], remainder='passthrough')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train_processed_check=preprocessor.fit_transform(X_train); input_dim=X_train_processed_check.shape[1]
def build_model(input_shape, num_classes): model=keras.Sequential([layers.Input(shape=(input_shape,)),layers.Dense(128, activation='relu'),layers.Dropout(0.3),layers.Dense(64, activation='relu'),layers.Dropout(0.2),layers.Dense(num_classes if num_classes > 2 else 1, activation='softmax' if num_classes > 2 else 'sigmoid')]); loss_function='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'; model.compile(optimizer='adam',loss=loss_function,metrics=['accuracy']); return model
nn_model=build_model(input_dim, num_classes); nn_model.summary()
X_train_processed=preprocessor.transform(X_train); X_test_processed=preprocessor.transform(X_test)
early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history=nn_model.fit(X_train_processed, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0) # Verbose 0 for less output
loss, accuracy=nn_model.evaluate(X_test_processed, y_test, verbose=0)
y_pred_proba=nn_model.predict(X_test_processed)
if num_classes > 2: y_pred=np.argmax(y_pred_proba, axis=-1)
else: y_pred=(y_pred_proba > 0.5).astype(int).flatten()
print(f"Accuracy: {accuracy:.4f}"); print(f"Loss: {loss:.4f}"); print("Report:\\n", classification_report(y_test, y_pred)); print("(NN Feat Importance Complex)")
""",
}

# Initialize Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Global store of uploaded data
uploaded_data = {}

# ------------------------------------------------------------------
# Helper: check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------------------------------------------------
# Logger: writes user/bot messages
def log_interaction(user_msg, bot_msg):
    log_entry = f"{datetime.now()} - User: {user_msg} | Bot: {bot_msg}\n"
    try:
        with open(os.path.join(LOG_FOLDER, "chat.log"), "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Logging failed: {e}")

# ------------------------------------------------------------------
# LLM interaction (DeepSeek)
def deepseek_response(prompt_text, purpose="general"):
    """Sends prompt to DeepSeek (Ollama) and returns text response."""
    # Minimal sanitizing
    clean_prompt = prompt_text.replace('"', '\\"').replace('\n', ' ').replace("'", "\\'")

    # Determine final prompt based on purpose
    final_prompt = clean_prompt
    if purpose == "classify_relevance":
        final_prompt = (f'Is this query related to choice modeling? Answer "Yes" or "No".\nQuery: "{clean_prompt}"')
    elif purpose == "classify_intent":
        final_prompt = (f'Does the user want to **build/apply a model using data** or ask a **general question**? '
                        f'Answer "Build Intent" or "General Question".\nQuery: "{clean_prompt}"')
    elif purpose in ["interpret_state_response", "adapt_code", "general_knowledge", "traditional_guidance"]:
        pass  # use prompt_text as is
    else:
        # default to general knowledge
        purpose = "general_knowledge"

    command = f'ollama run deepseek-r1:7b'
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            encoding='utf-8',
            errors='replace'
        )
        stdout, stderr = process.communicate(input=final_prompt + "\n", timeout=60)
        if process.returncode != 0:
            error_message = stderr.strip() or f"Ollama command failed (code {process.returncode})"
            if "connection refused" in error_message.lower():
                return "Error: Cannot connect to Ollama service."
            return f"Error calling DeepSeek (Ollama): {error_message}"

        response = stdout.strip()
        # remove ANSI codes
        response = re.sub(r'\x1b\[.*?m', '', response)
        return response

    except subprocess.TimeoutExpired:
        print("ERROR: Ollama command timed out.")
        return "Error: The request to the language model timed out. Please try again."
    except FileNotFoundError:
        return "Error: Ollama command not found."
    except Exception as e:
        print(f"Subprocess Error: {e}\n{traceback.format_exc()}")
        return f"Error running Ollama: {e}"

# ------------------------------------------------------------------
# Save final artifacts
def save_completion_artifacts(session_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = session_data.get("data_driven_model_type") or session_data.get("traditional_model_term", "unknown")
    filename = f"completion_{model_type.replace(' ','_')}_{timestamp}.json"
    filepath = os.path.join(COMPLETED_MODELS_FOLDER, filename)
    artifacts = {
        "timestamp": datetime.now().isoformat(),
        "original_query": session_data.get("original_query"),
        "model_type": model_type,
        "final_explanation_or_code": session_data.get("final_explanation_or_code"),
        "data_info": session_data.get("data_info", {}),
    }
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=4)
        print(f"Saved completion artifacts to {filepath}")
        return f"Process details saved as {filename}."
    except Exception as e:
        print(f"Error saving artifacts: {e}\n{traceback.format_exc()}")
        return "Error saving details."

# ------------------------------------------------------------------
# Generate data-driven code via LLM
def _generate_data_driven_code(chosen_type, user_session):
    sample_code = DATA_DRIVEN_SAMPLE_CODE.get(chosen_type, "# Sample code missing.")
    original_query = user_session.get('original_query', 'Not specified')
    data_info = user_session.get('data_info', {})
    s_filename = data_info.get('filename', 'N/A')
    s_columns = data_info.get('columns', [])
    s_cols_str = ", ".join(s_columns) if s_columns else "N/A"
    s_description = data_info.get('description', 'None')
    data_context = (f"Filename: '{s_filename}'. Cols: {s_cols_str}. Desc: {s_description}")

    prompt = (f"User wants '{chosen_type}' model for task: '{original_query}'.\n"
              f"Data: {data_context}\n\n"
              f"Generic Example:\n```python\n{sample_code}\n```\n\n"
              f"**Task:** Adapt example for user data (`df`). ID X/y from cols ({s_cols_str}). "
              f"Provide runnable Python: 1.Select X/y. 2.Preprocess. 3.Train '{chosen_type}'. "
              f"4.Evaluate. 5.Feat importance. Explain steps.")
    code_explanation = deepseek_response(prompt, purpose="adapt_code")

    user_session["conversation_state"] = "AWAITING_CODE_FEEDBACK"
    user_session["current_code_explanation"] = code_explanation
    user_session["data_driven_model_type"] = chosen_type
    return (f"Okay, here's an example using {chosen_type} for your data:\n\n{code_explanation}\n\n"
            f"What do you think? (Describe changes or say 'Looks good' / 'Finish')")


# ------------------------------------------------------------------
# Core Chat Logic: State machine that calls deepseek_response
def process_chat(message_text):
    message_lower = message_text.lower().strip()
    conversation_state = session.get("conversation_state")

    # Get data context
    last_upload_info = uploaded_data.get('last_upload', {})
    df_columns = []
    df = None
    if "df" in last_upload_info and isinstance(last_upload_info["df"], pd.DataFrame):
        df_columns = last_upload_info["df"].columns.tolist()
        df = last_upload_info["df"]
    has_data = df is not None

    # ----------------------------------
    # 1) If we are waiting for clarification (traditional vs data-driven)
    if conversation_state == "AWAITING_METHOD_CLARIFICATION":
        mentioned_traditional_term = session.get("mentioned_traditional_term", "traditional model")
        prompt = (f"User was asked if they want guidance on the traditional model '{mentioned_traditional_term}' "
                  f"or prefer a data-driven approach. Interpret their response. "
                  f"Did they indicate 'Traditional', 'Data-Driven', or is it 'Unclear'? "
                  f"Answer only 'Traditional', 'Data-Driven', or 'Unclear'.\nUser Response: '{message_text}'")
        interpretation = deepseek_response(prompt, purpose="interpret_state_response").strip().lower()
        print(f"DEBUG: Clarification Interpretation: '{interpretation}' for message '{message_text}'")

        if "traditional" in interpretation:
            session["traditional_model_term"] = mentioned_traditional_term
            data_info = session.get('data_info', {})
            s_filename = data_info.get('filename', 'N/A')
            s_columns = data_info.get('columns', [])
            s_cols_str = ", ".join(s_columns) if s_columns else "N/A"
            s_description = data_info.get('description', 'None')
            data_context = (f"Filename: '{s_filename}'. Cols: {s_cols_str}. Desc: {s_description}")
            original_query = session.get('original_query', '')
            prompt = (f"User wants guidance on specifying '{mentioned_traditional_term}' model for query: '{original_query}'.\n"
                      f"Data: {data_context}\nExplain steps, utility functions, data format, interpretation, considerations. Focus on conceptual guidance.")
            guidance = deepseek_response(prompt, purpose="traditional_guidance")
            session["final_explanation_or_code"] = guidance
            save_feedback = save_completion_artifacts(session)
            session.clear()
            return f"Okay, here's guidance on specifying a {mentioned_traditional_term} model:\n\n{guidance}\n\n{save_feedback}"

        elif "data-driven" in interpretation:
            session["conversation_state"] = "AWAITING_DATADRIVEN_TYPE"
            available_models = ", ".join(DATA_DRIVEN_SAMPLE_CODE.keys())
            # Check if they named a data-driven type in the same message
            chosen_dd_type = None
            for model_key in DATA_DRIVEN_SAMPLE_CODE.keys():
                if re.search(r'\b' + re.escape(model_key).replace('\\ ', '[\\s-]*') + r'\b',
                             message_lower, re.IGNORECASE):
                    chosen_dd_type = model_key
                    break
            if chosen_dd_type:
                return _generate_data_driven_code(chosen_dd_type, session)
            else:
                return f"Okay, let's use a data-driven approach. Which type? Examples: {available_models}?"

        else:  # Unclear
            return (f"Sorry, I'm unclear. Do you want guidance on the traditional '{mentioned_traditional_term}' model, "
                    f"or prefer a data-driven approach (like Random Forest)? Please clarify.")

    # ----------------------------------
    # 2) If we are waiting for a specific data-driven model name
    elif conversation_state == "AWAITING_DATADRIVEN_TYPE":
        chosen_type_kw = None
        for model_key in DATA_DRIVEN_SAMPLE_CODE.keys():
            pattern = r'\b' + re.escape(model_key).replace('\\ ', '[\\s-]*') + r'\b'
            if re.search(pattern, message_lower, re.IGNORECASE):
                chosen_type_kw = model_key
                break

        if chosen_type_kw:
            print(f"DEBUG: Matched data-driven type via keyword: {chosen_type_kw}")
            return _generate_data_driven_code(chosen_type_kw, session)
        else:
            # fallback: let LLM interpret
            prompt = (f"User was asked to name a data-driven model type (like {', '.join(DATA_DRIVEN_SAMPLE_CODE.keys())}). "
                      f"Did their response mention one of these or a similar technique suitable for choice prediction? "
                      f"If yes, state the technique name (lowercase). If no/unclear, answer 'Unclear'.\n"
                      f"User Response: '{message_text}'")
            interpretation = deepseek_response(prompt, purpose="interpret_state_response").strip().lower()
            print(f"DEBUG: AWAIT_DATADRIVEN Interpretation: '{interpretation}'")

            if interpretation != "unclear" and interpretation in DATA_DRIVEN_SAMPLE_CODE:
                print(f"DEBUG: Matched data-driven type via LLM: {interpretation}")
                return _generate_data_driven_code(interpretation, session)
            else:
                available_models = ", ".join(DATA_DRIVEN_SAMPLE_CODE.keys())
                return f"Sorry, please choose a data-driven model type I have sample code for, such as: {available_models}."

    # ----------------------------------
    # 3) If we are waiting for code feedback
    elif conversation_state == "AWAITING_CODE_FEEDBACK":
        # LLM: interpret user response as 'Satisfied', 'Modify', or 'Unclear'
        prompt = (f"User was shown generated code/explanation and asked if it looks right or needs modifications. "
                  f"Does their response indicate satisfaction ('yes', 'finish'), request for modifications, or unclear? "
                  f"Answer 'Satisfied', 'Modify', or 'Unclear'.\nUser Response: '{message_text}'")
        interpretation = deepseek_response(prompt, purpose="interpret_state_response").strip().lower()
        print(f"DEBUG: Code Feedback Interpretation: '{interpretation}'")

        if "satisfied" in interpretation:
            session["final_explanation_or_code"] = session.get("current_code_explanation", "No code generated.")
            session["data_info"] = session.get("data_info", {})
            save_feedback = save_completion_artifacts(session)
            final_response = f"Great! {save_feedback} You can start a new query."
            session.clear()
            return final_response
        elif "modify" in interpretation:
            print("DEBUG: Interpreted as request for modifications.")
            original_code = session.get("current_code_explanation", "No previous code.")
            refinement_request = message_text
            data_info = session.get('data_info', {})
            s_filename = data_info.get('filename', 'N/A')
            s_columns = data_info.get('columns', [])
            s_cols_str = ", ".join(s_columns) if s_columns else "N/A"
            s_description = data_info.get('description', 'None')
            model_type = session.get('data_driven_model_type', 'Unknown')
            data_context = (f"Filename: '{s_filename}'. Cols: {s_cols_str}. Desc: {s_description}")
            prompt = (f"Refining '{model_type}' code.\nData: {data_context}\nPrevious:\n```\n{original_code}\n```\n"
                      f"User request: '{refinement_request}'.\nProvide updated code/explanation.")
            refined_code_explanation = deepseek_response(prompt, purpose="adapt_code")
            session["conversation_state"] = "AWAITING_CODE_FEEDBACK"
            session["current_code_explanation"] = refined_code_explanation
            return (f"Okay, I've tried to incorporate your modifications:\n\n{refined_code_explanation}\n\n"
                    f"How does this look now? (Describe changes or say 'Looks good' / 'Finish')")
        else:
            return ("Sorry, I'm not sure if you're happy with the code or want changes. "
                    "Please say 'Looks good' or describe the modifications you'd like.")

    # ----------------------------------
    # 4) If conversation_state is None => new query
    elif conversation_state is None:
        # Quick keyword check
        is_relevant_keyword = any(keyword in message_lower for keyword in CHOICE_MODELING_KEYWORDS)
        if not is_relevant_keyword:
            # Double check with LLM
            llm_relevance = deepseek_response(message_text, purpose="classify_relevance").strip().lower()
            if "no" in llm_relevance:
                print(f"DEBUG: Query irrelevant: '{message_text}' -> LLM: No")
                return "This query doesn't seem related to choice modeling. Please focus on choice analysis/modeling."
            else:
                print(f"DEBUG: Query relevant via LLM despite no local keyword match.")

        print(f"DEBUG: Query relevant. Checking intent...")
        intent_result = deepseek_response(message_text, purpose="classify_intent")
        print(f"DEBUG: Intent result: '{intent_result}'")

        if "build intent" in intent_result.lower():
            if not has_data:
                return "Looks like you want to build a model, but please upload data first."

            session["original_query"] = message_text
            session["data_info"] = {
                "filename": last_upload_info.get('filename', 'N/A'),
                "columns": df_columns,
                "description": last_upload_info.get('description', '')
            }

            # see if user named a data-driven type
            chosen_dd_type = None
            for model_key in DATA_DRIVEN_SAMPLE_CODE.keys():
                pat = r'\b' + re.escape(model_key).replace('\\ ', '[\\s-]*') + r'\b'
                if re.search(pat, message_lower, re.IGNORECASE):
                    chosen_dd_type = model_key
                    break

            # see if user named a traditional type
            mentioned_traditional_term = None
            for trad_key in TRADITIONAL_MODEL_KEYWORDS:
                pat = r'\b' + re.escape(trad_key).replace('\\ ', '[\\s-]*') + r'\b'
                if re.search(pat, message_lower, re.IGNORECASE):
                    mentioned_traditional_term = trad_key
                    break

            if chosen_dd_type:
                return _generate_data_driven_code(chosen_dd_type, session)
            elif mentioned_traditional_term:
                session["conversation_state"] = "AWAITING_METHOD_CLARIFICATION"
                session["mentioned_traditional_term"] = mentioned_traditional_term
                available_models = ", ".join(DATA_DRIVEN_SAMPLE_CODE.keys())
                return (f"You mentioned '{mentioned_traditional_term}', a traditional model.\n\n"
                        f"Do you want guidance on that **traditional** approach?\n"
                        f"Or prefer a **data-driven** model (e.g., {available_models})?")
            else:
                session["conversation_state"] = "AWAITING_DATADRIVEN_TYPE"
                available_models = ", ".join(DATA_DRIVEN_SAMPLE_CODE.keys())
                return (f"Okay, you want to build a model. Which data-driven type do you want?\n"
                        f"Some options: {available_models}")

        else:
            # general question
            print(f"DEBUG: Query is general question: '{message_text}'")
            data_context_str = ""
            if has_data:
                data_info = {
                    "filename": last_upload_info.get('filename', 'N/A'),
                    "columns": df_columns,
                    "description": last_upload_info.get('description', '')
                }
                data_context_str = (f"User data context: Filename='{data_info['filename']}', "
                                    f"Cols='{', '.join(data_info['columns'])}', "
                                    f"Desc='{data_info['description']}'. Answer with that in mind.\n\n")
            prompt = (f"You are a choice modeling expert assistant.\n{data_context_str}"
                      f"Answer user query:\n{message_text}")
            response = deepseek_response(prompt, purpose="general_knowledge")
            return response

    # ----------------------------------
    # Fallback
    else:
        print(f"WARN: Unknown conversation state '{conversation_state}'. Clearing session.")
        session.clear()
        return "There was an issue tracking the conversation. Let's start over."

# ------------------------------------------------------------------
# Flask Routes
@app.route("/", methods=["GET", "POST"])
def home():
    feedback = None
    file_feedback = None
    description_feedback = None
    current_columns = []
    current_description = ""
    current_filename = None

    if "last_upload" in uploaded_data:
        last_upload_info = uploaded_data["last_upload"]
        if "df" in last_upload_info and isinstance(last_upload_info["df"], pd.DataFrame):
            current_columns = last_upload_info["df"].columns.tolist()
        current_description = last_upload_info.get("description", "")
        current_filename = last_upload_info.get("filename")

    if request.method == "POST":
        form_action = request.form.get("action")
        if form_action == "upload_file" and "file" in request.files and request.files["file"].filename != "":
            # Clear session on new upload
            session.clear()
            feedback, file_feedback, description_feedback = None, None, None

            file = request.files["file"]
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                try:
                    file.save(file_path)
                    if filename.endswith("csv"):
                        try:
                            df = pd.read_csv(file_path, encoding='utf-8')
                        except UnicodeDecodeError:
                            df = pd.read_csv(file_path, encoding='latin-1')
                    else:
                        df = pd.read_excel(file_path)

                    if df.empty or df.shape[0] < 2:
                        error_msg = "empty" if df.empty else f"needs >= 2 rows (has {df.shape[0]})"
                        file_feedback = f"Error: File '{filename}' is {error_msg}."
                        current_columns, current_description, current_filename = [], "", None
                    else:
                        uploaded_data["last_upload"] = {
                            "filename": filename,
                            "df": df,
                            "path": file_path,
                            "description": ""
                        }
                        current_columns = df.columns.tolist()
                        current_description = ""
                        current_filename = filename
                        cols_str = ", ".join(current_columns)
                        file_feedback = f"File '{filename}' uploaded ({df.shape[0]}x{df.shape[1]}). Cols: {cols_str}"

                except Exception as e:
                    file_feedback = f"Error processing '{filename}': {e}"
                    current_columns, current_description, current_filename = [], "", None
            else:
                file_feedback = "Error: Invalid file type (.csv/.xlsx only)."

        elif form_action == "save_description" and "data_description" in request.form:
            if "last_upload" in uploaded_data:
                description_text = request.form["data_description"]
                uploaded_data["last_upload"]["description"] = description_text.strip()
                current_description = description_text.strip()
                description_feedback = "Data description saved."
            else:
                description_feedback = "Please upload data first."

            # If we have a conversation ongoing, store the description
            if "conversation_state" in session and "data_info" in session:
                session["data_info"]["description"] = current_description
                session.modified = True
                log_interaction("User saved data description", f"Len: {len(description_text)}")

        elif form_action == "send_message" and "message" in request.form:
            user_message = request.form["message"]
            if not user_message.strip():
                feedback = "Please enter a message."
            else:
                response_text = process_chat(user_message)
                feedback = response_text
                log_interaction(user_message, response_text)

    return render_template(
        "index.html",
        feedback=feedback,
        file_feedback=file_feedback,
        description_feedback=description_feedback,
        columns=current_columns,
        description=current_description,
        current_filename=current_filename
    )

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message_text = data.get("message", "")
    if not message_text.strip():
        return jsonify({"response": "Please enter a message."})
    response_text = process_chat(message_text)
    log_interaction(f"[AJAX] {message_text}", response_text)
    return jsonify({"response": response_text})

# ------------------------------------------------------------------
# Main block
if __name__ == "__main__":
    # Ensure subfolders exist
    for folder in ["static", "templates", "storage", "storage/uploads", "storage/logs", COMPLETED_MODELS_FOLDER]:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
                print(f"Created folder: {folder}")
            except OSError as e:
                print(f"Warning: Could not create folder {folder}: {e}.")

    logo_path = os.path.join("static", "logo.png")
    template_path = os.path.join("templates", "index.html")

    if not os.path.exists(logo_path):
        print(f"\nWARNING: Logo file not found at {os.path.abspath(logo_path)}\n")
    if not os.path.exists(template_path):
        print(f"\nERROR: 'index.html' not found in {os.path.abspath('templates')}\n")
        exit(1)

    print("Starting Flask application (LLM State Interpretation) on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

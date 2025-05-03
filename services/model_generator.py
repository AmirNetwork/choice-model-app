# services/model_generator.py
import re
import json
from datetime import datetime
from config.constants import DATA_DRIVEN_SAMPLE_CODE, TRADITIONAL_MODEL_KEYWORDS, COMPLETED_MODELS_FOLDER
from services.deepseek import query_model
import os

def generate_data_driven_code(chosen_type: str, user_query: str, data_info: dict) -> str:
    """
    Generates code for the given data-driven model type, referencing user query and data info.
    Returns a string containing code + explanation from LLM.
    """
    # Pull sample snippet from constants
    sample_code = DATA_DRIVEN_SAMPLE_CODE.get(chosen_type.lower(), "# Sample code missing.")
    filename = data_info.get('filename', 'N/A')
    columns = data_info.get('columns', [])
    desc = data_info.get('description', 'No description')
    col_str = ", ".join(columns) if columns else "N/A"

    prompt = (
        f"User wants a '{chosen_type}' model for task: '{user_query}'.\n"
        f"Data context: Filename='{filename}', Columns={col_str}, Desc={desc}.\n\n"
        f"Here's a generic example:\n```python\n{sample_code}\n```\n\n"
        f"Now adapt for the user's data: 1) Identify X/y. 2) Preprocess. 3) Train. 4) Evaluate. "
        f"5) Feature importance. Return a succinct, runnable Python code snippet and explanation."
    )
    response = query_model(prompt, purpose="general")  # or some "adapt_code" purpose
    return response


def save_completion_artifacts(session_data: dict) -> str:
    """
    Saves final conversation artifacts (model code, explanation, etc.) as a JSON file in COMPLETED_MODELS_FOLDER.
    """
    os.makedirs(COMPLETED_MODELS_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = session_data.get("data_driven_model_type") or session_data.get("traditional_model_term", "unknown")
    filename = f"completion_{model_type.replace(' ','_')}_{timestamp}.json"
    filepath = os.path.join(COMPLETED_MODELS_FOLDER, filename)

    artifacts = {
        "timestamp": datetime.now().isoformat(),
        "original_query": session_data.get("original_query"),
        "model_type": model_type,
        "final_explanation_or_code": session_data.get("final_explanation_or_code"),
        "data_info": session_data.get("data_info", {})
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=4)
        return f"Process details saved as {filename}."
    except Exception as e:
        print(f"Error saving artifacts: {e}")
        return "Error saving details."

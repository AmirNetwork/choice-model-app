# utils/data.py
import os
import pandas as pd
from werkzeug.utils import secure_filename
from config.constants import ALLOWED_EXTENSIONS, UPLOAD_FOLDER

def allowed_file(filename: str) -> bool:
    """
    Check if the file extension is allowed.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_upload(file_storage):
    """
    Securely saves an uploaded file (if valid), reads it into a DataFrame, and returns it.
    Raises ValueError if the file is invalid or cannot be processed.
    """
    if not file_storage or file_storage.filename.strip() == "":
        raise ValueError("No file provided.")

    filename = secure_filename(file_storage.filename)
    if not allowed_file(filename):
        raise ValueError(f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(filepath)

    # Attempt to read into DataFrame
    if filename.lower().endswith(".csv"):
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='latin-1')
    else:
        # assume .xlsx
        df = pd.read_excel(filepath)

    if df.empty or df.shape[0] < 2:
        raise ValueError(
            f"Uploaded file must have at least 2 rows. Found {df.shape[0]} rows."
        )

    return df, filename

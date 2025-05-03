# utils/logger.py
import os
from datetime import datetime
from config.constants import LOG_FOLDER

def log_interaction(user_msg, bot_msg):
    """
    Appends a line to 'chat.log' with timestamp, user message, and bot response.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - User: {user_msg} | Bot: {bot_msg}\n"
    try:
        os.makedirs(LOG_FOLDER, exist_ok=True)
        log_path = os.path.join(LOG_FOLDER, "chat.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Logging failed: {e}")

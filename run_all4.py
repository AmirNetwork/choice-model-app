# runall.py
import subprocess
import time
import sys

def start_celery():
    """
    Start the Celery worker using the 'celery' module with the solo pool option.
    This uses the in-memory broker and result backend, so no external broker is needed.
    """
    print("Starting Celery worker (using --pool=solo)...")
    celery_proc = subprocess.Popen(
        [sys.executable, "-m", "celery", "-A", "tasks", "worker", "--pool=solo", "--loglevel=info"]
    )
    return celery_proc

def start_flask():
    """
    Start the Flask application.
    """
    print("Starting Flask app...")
    flask_proc = subprocess.Popen([sys.executable, "app_neo4.py"])
    return flask_proc

if __name__ == "__main__":
    celery_proc = start_celery()
    time.sleep(2)
    flask_proc = start_flask()
    try:
        flask_proc.wait()
    except KeyboardInterrupt:
        print("Stopping processes due to keyboard interrupt...")
        celery_proc.terminate()
        flask_proc.terminate()

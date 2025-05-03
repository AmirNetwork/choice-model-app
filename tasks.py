# tasks.py
from celery import Celery

# Use an in-memory broker and the cache (in-memory) result backend for development.
celery_app = Celery('tasks', broker='memory://', backend='cache+memory://')

# Import your multi-step generation function from the app.
from app_neo3 import multi_step_generation

@celery_app.task
def generate_response(user_msg, user_id):
    """
    Celery task that runs the heavy multi-step generation pipeline.
    """
    response = multi_step_generation(user_msg, user_id)
    return response

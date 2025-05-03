from celery import Celery
from config import Config
from models.llm import LLMManager
from redis import Redis
import logging

logger = logging.getLogger(__name__)

app = Celery("tasks", broker=Config.CELERY_BROKER_URL, backend=Config.CELERY_RESULT_BACKEND)
app.conf.update(task_track_started=True, task_serializer="json", accept_content=["json"], result_serializer="json")

redis_client = Redis.from_url(Config.CELERY_BROKER_URL)

@app.task(bind=True)
def generate_response(self, user_msg: str, user_id: str, user_data: dict, llm_manager: LLMManager):
    try:
        # Placeholder for DeepSeek API (mocked here)
        def deepseek_call(prompt: str) -> str:
            return llm_manager.generate_text(prompt)  # Mock until API available

        def baseline_pass(msg: str, data_info: dict) -> str:
            from utils.data import get_data_preview
            prompt = f"General knowledge answer:\nData:\n{get_data_preview(data_info)}\nQuery: {msg}"
            return deepseek_call(prompt)

        def refine_prompt(msg: str, data_info: dict) -> str:
            from utils.data import get_data_preview
            prompt = (
                f"Refine for GPT-Neo:\nData:\n{get_data_preview(data_info)}\n"
                f"Query: {msg}\nMake concise and clear."
            )
            return deepseek_call(prompt)

        def specialized_pass(refined: str, msg: str, data_info: dict) -> str:
            from utils.data import get_data_preview
            # Simplified reference search (use embeddings in production)
            prompt = (
                f"Specialized answer:\nData:\n{get_data_preview(data_info)}\n"
                f"Refined prompt: {refined}\nUse domain knowledge."
            )
            return llm_manager.generate_text(prompt)

        def merge_answers(baseline: str, specialized: str, msg: str, data_info: dict) -> str:
            from utils.data import get_data_preview
            prompt = (
                f"Combine:\nBaseline: {baseline}\nSpecialized: {specialized}\n"
                f"Data:\n{get_data_preview(data_info)}\nQuery: {msg}"
            )
            return deepseek_call(prompt)

        data_info = user_data.get(user_id, {}).get("data_info", {})
        baseline = baseline_pass(user_msg, data_info)
        refined = refine_prompt(user_msg, data_info)
        specialized = specialized_pass(refined, user_msg, data_info)
        final = merge_answers(baseline, specialized, user_msg, data_info)
        
        # Cache result
        redis_client.setex(f"response:{user_id}:{self.request.id}", 3600, final)
        return final
    except Exception as e:
        logger.error(f"Task error: {e}")
        raise
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from config import Config
import logging

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def initialize_gpt_neo(self):
        if self.model is None:
            logger.info(f"Loading GPT-Neo model {Config.GPTNEO_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(Config.GPTNEO_MODEL_NAME, cache_dir=Config.CACHE_DIR)
            base_model = AutoModelForCausalLM.from_pretrained(
                Config.GPTNEO_MODEL_NAME,
                cache_dir=Config.CACHE_DIR,
                device_map="auto",
                torch_dtype=torch.float16
            )
            lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            self.model = get_peft_model(base_model, lora_config).to(self.device)
            logger.info("GPT-Neo + LoRA loaded")

    def generate_text(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        if self.model is None or self.tokenizer is None:
            self.initialize_gpt_neo()
        
        enc = self.tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=1792)
        inputs = {"input_ids": torch.tensor([enc], dtype=torch.long).to(self.device)}
        try:
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            out_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            return out_text[len(self.tokenizer.decode(enc, skip_special_tokens=True)):].strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating text: {e}"

    def encode_text(self, text: str) -> list:
        return self.embedder.encode(text, convert_to_tensor=False).tolist()
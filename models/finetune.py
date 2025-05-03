import os
import json
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from config import Config
import logging

logger = logging.getLogger(__name__)

def build_training_dataset(user_id: str, user_data: dict, llm_manager) -> str:
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(Config.TRAINING_DATA_FOLDER, f"training_{stamp}.jsonl")
    samples = []

    # Process references
    for folder in [Config.REFERENCES_FOLDER, Config.TRAINED_REFERENCES_FOLDER]:
        for fname in os.listdir(folder):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                        ref = json.load(f)
                    instruction = f"Title: {ref.get('title', '')}\nPaper:\n{ref.get('paper_text', '')}\nCode:\n{ref.get('code_snippet', '')}"
                    samples.append({"instruction": instruction, "output": "References integrated."})
                except Exception as e:
                    logger.warning(f"Error reading reference {fname}: {e}")

    # Add user feedback
    for lesson in user_data.get(user_id, {}).get("learning_points", []):
        samples.append({"instruction": f"Feedback: {lesson}", "output": "Feedback noted."})

    # Add data description
    data_info = user_data.get(user_id, {}).get("data_info", {})
    if "df" in data_info and "description" in data_info.get("description", ""):
        preview = data_info["df"].head(20).to_csv(index=False)
        samples.append({
            "instruction": f"Data Description: {data_info['description']}\nData Preview:\n{preview}",
            "output": "Data integrated."
        })

    if samples:
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    return out_path

def fine_tune_model(user_id: str, user_data: dict, llm_manager) -> str:
    dataset_path = build_training_dataset(user_id, user_data, llm_manager)
    if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) < 10:
        return "No data for fine-tuning."

    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        def preprocess(ex):
            return {"prompt": f"Instruction: {ex['instruction']}\nResponse:", "label": ex["output"]}
        dataset = dataset.map(preprocess)

        def tokenize_fn(ex):
            full_text = ex["prompt"] + " " + ex["label"]
            tok = llm_manager.tokenizer(full_text, truncation=True, max_length=512)
            tok["labels"] = tok["input_ids"].copy()
            return tok

        dataset = dataset.map(tokenize_fn, batched=False)

        def data_collator(features):
            batch = {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
                "labels": [f["labels"] for f in features]
            }
            out = llm_manager.tokenizer.pad(
                {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
                return_tensors="pt"
            )
            out["labels"] = llm_manager.tokenizer.pad({"input_ids": batch["labels"]}, return_tensors="pt")["input_ids"]
            return out

        train_args = TrainingArguments(
            output_dir=os.path.join(Config.TRAINING_DATA_FOLDER, "lora_out"),
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            fp16=True,
            logging_steps=5,
            save_steps=50
        )
        trainer = Trainer(
            model=llm_manager.model,
            args=train_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        trainer.train()
        return f"Fine-tuned on {dataset.num_rows} samples."
    except Exception as e:
        logger.error(f"Fine-tuning error: {e}")
        return f"Fine-tuning failed: {e}"
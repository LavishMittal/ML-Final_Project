import os
import json
import shutil
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# ------------------ CONFIG ------------------
MODEL_NAME = "distilgpt2"
TRAIN_FILE = "data/train.jsonl"
OUTPUT_DIR = "fine_tuned_model"
MAX_LENGTH = 256
BATCH_SIZE = 2
EPOCHS = 1
CHUNK_SIZE = 2000

PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")

# ------------------ HELPERS ------------------
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "instruction" in obj and "response" in obj:
                    text = f"Instruction: {obj['instruction']}\nResponse: {obj['response']}"
                    data.append({"text": text})
            except json.JSONDecodeError:
                print(f"Skipping bad JSON at line {i+1}")
    return data

def get_last_completed_chunk():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f).get("last_chunk", -1) 
    return -1

def save_progress(chunk_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_chunk": chunk_id}, f)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

def clean_checkpoints(chunk_dir):
    """Delete checkpoint-* folders inside a chunk to save space."""
    for item in os.listdir(chunk_dir):
        item_path = os.path.join(chunk_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            shutil.rmtree(item_path)
            print(f"Deleted checkpoint {item_path}")

# LOAD DATA
print("Loading dataset...")
data = load_jsonl(TRAIN_FILE)
total_samples = len(data)
print(f"Loaded {total_samples} samples")

# LOAD MODEL 
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# SETUP LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_proj", "c_attn"],  # GPT-2 layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# CHUNK TRAINING
num_chunks = (total_samples // CHUNK_SIZE) + 1
last_done = get_last_completed_chunk()
print(f"Resuming from chunk {last_done + 1}/{num_chunks}")

for chunk_id in range(last_done + 1, num_chunks):
    start = chunk_id * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, total_samples)
    if start >= end:
        break

    print(f"\n===== CHUNK {chunk_id+1}/{num_chunks} | Samples {start} → {end} =====")

    # Load previous chunk weights
    if chunk_id > 0:
        prev_chunk_dir = os.path.join(OUTPUT_DIR, f"chunk-{chunk_id}")
        if os.path.exists(prev_chunk_dir):
            print(f"Loading model from previous chunk {prev_chunk_dir}")
            model = AutoModelForCausalLM.from_pretrained(prev_chunk_dir)
            model = get_peft_model(model, lora_config)

    # Prepare dataset chunk
    chunk_data = data[start:end]
    dataset = Dataset.from_list(chunk_data)
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    chunk_output_dir = os.path.join(OUTPUT_DIR, f"chunk-{chunk_id+1}")

    training_args = TrainingArguments(
        output_dir=chunk_output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_steps=20,
        save_steps=100,
        save_total_limit=1,
        learning_rate=5e-5,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save chunk model
    model.save_pretrained(chunk_output_dir)
    tokenizer.save_pretrained(chunk_output_dir)
    save_progress(chunk_id)
    clean_checkpoints(chunk_output_dir)
    print(f"Chunk {chunk_id+1} complete and saved at {chunk_output_dir}")

# FINAL MODEL
final_chunk_dir = os.path.join(OUTPUT_DIR, f"chunk-{num_chunks}")
final_dir = os.path.join(OUTPUT_DIR, "final_model")
model = AutoModelForCausalLM.from_pretrained(final_chunk_dir)
tokenizer = AutoTokenizer.from_pretrained(final_chunk_dir)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"\nTraining complete! Final model saved at {final_dir}")

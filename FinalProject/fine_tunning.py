import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # prevent MPS memory crashes

import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# CONFIG
MODEL_NAME = "distilgpt2"
TRAIN_FILE = "train.jsonl"
OUTPUT_DIR = "fine_tuned_model"
MAX_LENGTH = 256
BATCH_SIZE = 2
EPOCHS = 1
CHUNK_SIZE = 2000   # train 2k samples at a time

# LOAD DATA
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

print("Loading dataset...")
data = load_jsonl(TRAIN_FILE)
total_samples = len(data)
print(f"Loaded {total_samples} samples")

# MODEL
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

# CHUNK TRAINING LOOP
num_chunks = (total_samples // CHUNK_SIZE) + 1
print(f"Training in {num_chunks} chunks...\n")

for chunk_id in range(num_chunks):
    start = chunk_id * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, total_samples)

    if start >= end:
        break

    print(f"\n===== CHUNK {chunk_id+1}/{num_chunks} | Samples {start} → {end} =====")

    chunk_data = data[start:end]
    dataset = Dataset.from_list(chunk_data)

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"]
    )

    chunk_output_dir = os.path.join(OUTPUT_DIR, f"chunk-{chunk_id+1}")

    training_args = TrainingArguments(
        output_dir=chunk_output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_steps=20,
        save_steps=100,
        save_total_limit=2,  # keep only last 2 checkpoints per chunk
        learning_rate=5e-5,
        report_to="none",
        fp16=False,  
        resume_from_checkpoint=True  # resume if checkpoint exists
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save after each chunk
    model.save_pretrained(chunk_output_dir)
    tokenizer.save_pretrained(chunk_output_dir)

    print(f"Chunk {chunk_id+1} complete and saved at {chunk_output_dir}")

print("\nTRAINING COMPLETE — entire dataset learned!")

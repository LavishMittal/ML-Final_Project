import random
import json
from datasets import load_dataset, DatasetDict

# Load TinyStories
print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories")

# Define Quest Template
SETTINGS = ["forest", "desert city", "mountain village", "haunted castle", "coastal town"]
TONES = ["epic", "dark", "mysterious", "lighthearted"]
LEVELS = [1, 2, 3, 4, 5]

def build_prompt(level, setting, tone):
    return f"Create a level-{level} fantasy quest set in a {setting} with a {tone} tone."

def convert_to_quest_format(example):
    level = random.choice(LEVELS)
    setting = random.choice(SETTINGS)
    tone = random.choice(TONES)
    
    prompt = build_prompt(level, setting, tone)
    
    return {
        "instruction": prompt,
        "response": example["text"].strip()
    }

# Convert train & validation
print("Converting train dataset...")
train_dataset = dataset["train"].map(convert_to_quest_format)
print("Converting validation dataset...")
val_dataset = dataset["validation"].map(convert_to_quest_format)

# Clean text
def clean(example):
    text = example["response"].replace("\n", " ").strip()
    return {"instruction": example["instruction"], "response": text}

train_dataset = train_dataset.map(clean)
val_dataset = val_dataset.map(clean)

# Create test split from train (10%)
split = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
test_dataset = split["test"]

# Combine into DatasetDict
final_dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Save as JSONL
for split_name, split_data in final_dataset.items():
    filename = f"{split_name}.jsonl"
    print(f"Saving {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        for row in split_data:
            json.dump(row, f)
            f.write("\n")

print(" Dataset preparation complete!")
print("Files generated: train.jsonl, validation.jsonl, test.jsonl")

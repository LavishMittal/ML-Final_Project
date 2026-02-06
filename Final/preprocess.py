import random
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split

MAX_SAMPLES = 350000
OUTPUT_DIR = "data"

print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

print("Shuffling and selecting subset...")
dataset = dataset.shuffle(seed=42).select(range(MAX_SAMPLES))

# ---------- QUEST PROMPT TEMPLATES ----------
SETTINGS = ["forest", "desert city", "mountain village", "ancient ruins", "seaside town"]
TONES = ["epic", "dark", "mysterious", "humorous", "adventurous"]
LEVELS = list(range(1, 11))

def build_prompt():
    return (
        f"Create a level-{random.choice(LEVELS)} quest set in a "
        f"{random.choice(SETTINGS)} with a {random.choice(TONES)} tone."
    )

print("Formatting into instruction-response pairs...")
data = []

for example in dataset:
    prompt = build_prompt()
    story = example["text"].strip()

    data.append({
        "instruction": prompt,
        "response": story
    })

# ---------- SPLIT ----------
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print("Saving files...")

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

save_jsonl(train_data, f"{OUTPUT_DIR}/train.jsonl")
save_jsonl(val_data, f"{OUTPUT_DIR}/val.jsonl")
save_jsonl(test_data, f"{OUTPUT_DIR}/test.jsonl")

print("Done!")
print(f"Train: {len(train_data)}")
print(f"Val: {len(val_data)}")
print(f"Test: {len(test_data)}")

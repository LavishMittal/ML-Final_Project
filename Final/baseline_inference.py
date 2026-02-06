# baseline_inference.py
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import random

# Config
MODEL_NAME = "gpt2"
JSONL_FILE = "data/test.jsonl"
OUTPUT_FILE = "baseline_outputs.jsonl"
MAX_LENGTH = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PROMPTS = 100  # small subset for fast testing

# Load model & tokenizer
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Load test prompts
test_prompts = []
with open(JSONL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        # Take the first key as prompt (robust to different JSONL formats)
        key = list(item.keys())[0]
        test_prompts.append(item[key])

print(f"Loaded {len(test_prompts)} test prompts.")

# Sample small subset for quick baseline
random.seed(42)
test_prompts = random.sample(test_prompts, min(NUM_PROMPTS, len(test_prompts)))
print(f"Using {len(test_prompts)} prompts for fast baseline generation.")

# Generation function
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()  # remove prompt

# Generate outputs
results = []

print("Generating outputs...")
for prompt in tqdm(test_prompts):
    gen_text = generate_text(prompt)
    results.append({
        "prompt": prompt,
        "result": gen_text
    })

# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Generation done! Outputs saved to {OUTPUT_FILE}")

# Show first 5 outputs
print("\n--- Sample Outputs ---")
for item in results[:5]:
    print(f"PROMPT: {item['prompt']}")
    print(f"OUTPUT: {item['result'][:300]}...\n")

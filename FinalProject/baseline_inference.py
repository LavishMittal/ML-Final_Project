# baseline_inference.py
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "distilgpt2"
MAX_NEW_TOKENS = 150  
OUTPUT_FILE = "baseline_quests_output.json"
SUBSET_SIZE = 50  # Use only first 50 prompts for baseline test

# Load pretrained model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

# Load dataset prompts 
def load_prompts(filename, subset_size=None):
    prompts = []
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if subset_size is not None and i >= subset_size:
                break
            row = json.loads(line)
            prompts.append(row["instruction"])
    return prompts

def generate_quest(prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.2
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Truncate output to first 200 characters for readability
    return text[:200] + ("..." if len(text) > 200 else "")

if __name__ == "__main__":
    # Load only a subset of prompts for baseline test
    prompts = load_prompts("train.jsonl", subset_size=SUBSET_SIZE)

    print(f"Generating {len(prompts)} quests...")
    results = []

    for i, prompt in enumerate(prompts):
        output = generate_quest(prompt)
        results.append({
            "prompt": prompt,
            "output": output
        })
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(prompts)} generated...")

    # Save outputs
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Baseline generation complete! Saved to {OUTPUT_FILE}")

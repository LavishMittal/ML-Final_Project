from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm

# Config
MODEL_PATH = "fine_tuned_model/chunk-140"  
BASELINE_PROMPTS_PATH = "baseline_outputs.jsonl"
OUTPUT_PATH = "fine_tuned_outputs.jsonl"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.2
NUM_RETURN_SEQUENCES = 1  

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and tokenizer
print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Function to generate quest
def generate_quest(prompt, num_sequences=NUM_RETURN_SEQUENCES):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            num_return_sequences=num_sequences
        )

    sequences = []
    for seq in outputs:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        # Remove prompt from output
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        sequences.append(text)
    return sequences

# Load baseline prompts (JSONL)
baseline_data = []
with open(BASELINE_PROMPTS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        baseline_data.append(json.loads(line))

results = []

print("Generating fine-tuned outputs...\n")

# Generate quests
for item in tqdm(baseline_data):
    prompt = item["prompt"]
    generated_texts = generate_quest(prompt)
    results.append({
        "prompt": prompt,
        "outputs": generated_texts
    })

# Save results (JSONL)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Done! Saved fine-tuned outputs to {OUTPUT_PATH}")

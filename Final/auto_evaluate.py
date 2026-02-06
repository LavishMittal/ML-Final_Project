import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

# ----- CONFIG -----
TEST_SET_PATH = "evaluation/test_set.jsonl"  # path to your test prompts
OUTPUT_PATH = "evaluation/results_auto.jsonl"
MODEL_NAME = "gpt2"  # replace with your trained model path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 50  # how many tokens to generate per prompt

# ----- DIVERSITY METRIC -----
def distinct_n(generated_texts, n=2):
    ngrams = set()
    total_ngrams = 0
    for text in generated_texts:
        tokens = text.split()
        total_ngrams += max(len(tokens) - n + 1, 1)
        ngrams.update(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return len(ngrams) / max(total_ngrams, 1)

# ----- PERPLEXITY -----
def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return math.exp(loss.item())

# ----- MAIN -----
def main():
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    # Load test set
    test_set = []
    with open(TEST_SET_PATH, "r") as f:
        for line in f:
            test_set.append(json.loads(line))

    results = []
    generated_texts = []
    perplexities = []

    for item in test_set:
        prompt = item["prompt"]

        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,         # random sampling for diversity
            top_k=50,               # optional: top-k sampling
            top_p=0.95,             # optional: nucleus sampling
            temperature=0.8         # optional: creativity factor
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Compute perplexity
        ppl = calculate_perplexity(model, tokenizer, generated)

        # Collect for diversity metrics
        generated_texts.append(generated)
        perplexities.append(ppl)

        results.append({
            "prompt": prompt,
            "generated": generated,
            "perplexity": ppl
        })

    # Compute Distinct-1 and Distinct-2
    distinct_1 = distinct_n(generated_texts, n=1)
    distinct_2 = distinct_n(generated_texts, n=2)

    avg_ppl = sum(perplexities)/len(perplexities)

    print(f"Distinct-1: {distinct_1:.4f}")
    print(f"Distinct-2: {distinct_2:.4f}")
    print(f"Average Perplexity: {avg_ppl:.4f}")

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()

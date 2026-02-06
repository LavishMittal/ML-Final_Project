import json
import csv

# ----- CONFIG -----
TEST_SET_PATH = "evaluation/test_set.jsonl"
OUTPUT_PATH = "evaluation/results_human.jsonl"
SCORES_CSV = "evaluation/human_scores.csv"

# ----- MOCK GENERATOR -----
def mock_generate(prompt_dict):
    """Generate a placeholder response from prompt metadata."""
    prompt_text = prompt_dict.get("prompt", "")
    level = prompt_dict.get("level", "1")
    tone = prompt_dict.get("tone", "neutral")
    length = prompt_dict.get("length", "medium")
    return f"[Quest Level {level}, Tone: {tone}, Length: {length}] {prompt_text}"

# ----- MAIN -----
def main():
    # Load test set
    test_set = []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            test_set.append(json.loads(line))

    # Load human scores
    human_scores = []
    with open(SCORES_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                raise ValueError(f"Expected 3 scores per row, got {len(row)}: {row}")
            human_scores.append([int(x) for x in row])

    # Validate lengths
    if len(human_scores) != len(test_set):
        raise ValueError(
            f"Number of human scores ({len(human_scores)}) "
            f"does not match number of prompts ({len(test_set)})!"
        )

    # Generate outputs and combine with human scores
    results = []
    total_coherence = 0
    total_creativity = 0
    total_faithfulness = 0

    for prompt_dict, scores in zip(test_set, human_scores):
        generated_text = mock_generate(prompt_dict)
        results.append({
            "prompt": prompt_dict.get("prompt", ""),
            "level": prompt_dict.get("level", 1),
            "tone": prompt_dict.get("tone", "neutral"),
            "length": prompt_dict.get("length", "medium"),
            "generated": generated_text,
            "coherence": scores[0],
            "creativity": scores[1],
            "faithfulness": scores[2]
        })
        # Accumulate for averages
        total_coherence += scores[0]
        total_creativity += scores[1]
        total_faithfulness += scores[2]

    # Save to JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute averages
    count = len(results)
    avg_coherence = total_coherence / count
    avg_creativity = total_creativity / count
    avg_faithfulness = total_faithfulness / count

    print(f"Processed {count} prompts. Results saved to {OUTPUT_PATH}.")
    print(f"Average Coherence: {avg_coherence:.2f}")
    print(f"Average Creativity: {avg_creativity:.2f}")
    print(f"Average Faithfulness: {avg_faithfulness:.2f}")

if __name__ == "__main__":
    main()

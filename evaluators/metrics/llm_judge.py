import argparse
import json
from collections import defaultdict

import numpy as np
from openai import OpenAI
import re


import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)
ACCURACY_SCORE_PROMPT = """
    You are an expert in evaluating model answers. Your task is to assign a score between 0 and 1 to a generated answer based on its accuracy with respect to the reference (gold) answer. 
    - 1.0 means fully correct and completely matches the reference answer.
    - 0.0 means completely wrong or missing the core information.
    - Intermediate scores reflect partial correctness or missing key elements.

    Before giving the score, first provide a short one-sentence explanation of your reasoning. Then provide the score.
    Do NOT include any extra text beyond the explanation and the JSON with the score.
    Only provide the score as a float between 0 and 1, with at most two decimal places.

    Example 1:
    Question: What color is the sky on a clear day?
    Reference/Gold Answer: Blue
    Generated Answer: Blue
    Explanation: The generated answer exactly matches the reference answer.
    JSON output: {{"score": 1.0}}

    Example 2:
    Question: What color is the sky on a clear day?
    Reference/Gold Answer: Blue
    Generated Answer: Light blue
    Explanation: The generated answer is partially correct; it captures the main idea but is slightly imprecise.
    JSON output: {{"score": 0.8}}

    Example 3:
    Question: What color is the sky on a clear day?
    Reference/Gold Answer: Blue
    Generated Answer: Green
    Explanation: The generated answer is incorrect and does not match the reference.
    JSON output: {{"score": 0.0}}

    Now evaluate the real question:
    Question: {question}
    Reference/Gold Answer: {gold_answer}
    Generated Answer: {generated_answer}

    First, provide a short one-sentence explanation, then return the result in JSON format as: {{"score": float }}.
    """

def extract_json(text):
    """
    Extracts JSON content from a string, removing enclosing triple backticks and optional 'json' tag if present.
    If no code block is found, returns the text as-is.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text  # assume it's raw JSON
    return json_str

def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    response = client.chat.completions.create(
        model="LLAMA",
        messages=[
            {
                "role": "user",
                "content": ACCURACY_SCORE_PROMPT.format(
                    question=question, gold_answer=gold_answer, generated_answer=generated_answer
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    import pdb

    content = response.choices[0].message.content
    json_str = extract_json(content)
    parsed = json.loads(json_str)
    score = parsed["score"]
    
    # score = json.loads(extract_json(response.choices[0].message.content))["score"]
    return float(score)


def main():
    """Main function to evaluate RAG results using LLM judge."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/default_run_v4_k30_new_graph.json",
        help="Path to the input dataset file",
    )

    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            # Skip category 5
            if int(category) == 5:
                continue

            # Evaluate the answer
            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            LLM_JUDGE[category].append(label)

            # Store the results
            RESULTS[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            # Save intermediate results
            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4)

            # Print current accuracy for all categories
            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:  # Only print if there are results for this category
                    print(f"  Category {cat}: {np.mean(results):.4f} ({sum(results)}/{len(results)})")
            print("------------------------------------------")
        index += 1

    # Save final results
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4)

    # Print final summary
    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for k, v in LLM_JUDGE.items():
        print(k, np.mean(v))


if __name__ == "__main__":
    main()

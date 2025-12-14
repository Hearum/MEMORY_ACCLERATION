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
    - You may assign any float value between 0 and 1 (rounded to at most two decimal places) depending on partial correctness.
    - You can assign any intermediate value (like 0.67, 0.92, etc.) depending on partial correctness
    - If the generated answer is semantically equivalent to the reference answer, even if the wording differs, assign a high score close to 1.0.
    
    Before giving the score, first provide a short one-sentence explanation of your reasoning. Then provide the score.
    Do NOT include any extra text beyond the explanation and the JSON with the score.

    Example 1:
    Question: Who wrote 'Pride and Prejudice'?
    Reference/Gold Answer: Jane Austen
    Generated Answer: Jane Austen wrote 'Pride and Prejudice'
    Explanation: The answer exactly matches the reference and contains no errors.
    JSON output: {{"score": 1.0}}

    Example 2:
    Question: What is the capital of Germany?
    Reference/Gold Answer: Berlin
    Generated Answer: Munich
    Explanation: The generated answer is incorrect; it names a different German city, so it fails to capture the core information.
    JSON output: {{"score": 0.0}}

    Example 3:
    Question: List three primary colors.
    Reference/Gold Answer: Red, Blue, Yellow
    Generated Answer: Red, Yellow
    Explanation: The answer is partially correct; it lists two out of three correct primary colors, missing one key element.
    JSON output: {{"score": 0.67}}

    Example 4:
    Question: What is photosynthesis?
    Reference/Gold Answer: Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water, producing oxygen as a byproduct.
    Generated Answer: Photosynthesis is the process by which plants produce energy from sunlight.
    Explanation: The answer is partially correct; it mentions using sunlight and producing energy, but omits the carbon dioxide, water, and oxygen details, so it is incomplete.
    JSON output: {{"score": 0.6}}

    Example 5:
    Question: What kind of job is Joanna beginning to perform because of her movie scripts?
    Reference/Gold Answer: Filmmaker
    Generated Answer: Screenwriter
    Explanation: The generated answer refers to a specific role (screenwriter) that falls under the broader category of filmmaker, so it is semantically correct.
    JSON output: {{"score": 1.0}}

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

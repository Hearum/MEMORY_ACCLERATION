import os
import json
import re
import threading
import statistics
import argparse
import concurrent.futures
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
import backoff
import openai
from openai import OpenAI

from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics

# 模型配置
# model_zoo = {
#     'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
#     'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
#     'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
# }


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

def get_anscheck_prompt(task, question, answer, response):
    """
    Generate evaluation prompts for different task types.
    For all tasks, the model should first give a short reasoning, then a yes/no answer.
    """
    reasoning_suffix = (
        "First, provide a short (one sentence) explanation of your reasoning, "
        "then answer yes or no. "
        "Do NOT include both yes and no in your response."
    )

    if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Answer yes if the response contains the correct answer; otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, answer yes. "
            "If the response only contains a subset of the required information, answer no.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            f"{reasoning_suffix}"
        )
        prompt = template.format(question, answer, response)

    elif task == 'temporal-reasoning':
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Answer yes if the response contains the correct answer; otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all intermediate steps, answer yes. "
            "If it only contains a subset, answer no. "
            "Do not penalize off-by-one errors for days/weeks/months.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            f"{reasoning_suffix}"
        )
        prompt = template.format(question, answer, response)

    elif task == 'knowledge-update':
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Answer yes if the response contains the correct answer; otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, consider it correct as long as the updated answer is the required answer.\n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            f"{reasoning_suffix}"
        )
        prompt = template.format(question, answer, response)

    elif task == 'single-session-preference':
        template = (
            "I will give you a question, a rubric for the desired personalized response, and a response from a model. "
            "Answer yes if the response satisfies the desired response; otherwise, answer no. "
            "The model does not need to reflect all rubric points; it is correct as long as it recalls and uses the user's personal information correctly.\n\n"
            "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
            f"{reasoning_suffix}"
        )
        prompt = template.format(question, answer, response)

    else:
        template = (
            "I will give you a question, a reference or expected answer, and a response from a model. "
            "Answer yes if the model's response is correct, and no if it is incorrect or missing key information.\n\n"
            "Question: {}\n\nReference/Expected Answer: {}\n\nModel Response: {}\n\n"
            f"{reasoning_suffix}"
        )
        prompt = template.format(question, answer, response)

    return prompt


class Evaluation:
    def __init__(self, input_file, output_file=None, max_workers=8, dataset_type="locomo",
                 metric_model_short="openai"):
        self.input_file = input_file
        self.output_file = output_file or (
            input_file.replace(".jsonl", "").replace(".json", "")
            + f".eval-results-{dataset_type}.json"
        )
        self.max_workers = max_workers
        self.dataset_type = dataset_type
        self.results = defaultdict(list)
        self.lock = threading.Lock()
        self.data = self._load_data()

        if metric_model_short == "openai":
            openai.organization = os.getenv("OPENAI_ORGANIZATION")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_api_base = os.environ.get("OPENAI_API_BASE")

        self.metric_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        self.metric_model = "LLAMA"

    def _make_output_path(self, input_file: str) -> str:
        base, _ = os.path.splitext(input_file)
        return base + "_eval.json"

    def _load_data(self):
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                if self.input_file.endswith(".jsonl"):
                    return [json.loads(line) for line in f if line.strip()]
                else:
                    return json.load(f)
        except json.JSONDecodeError:
            with open(self.input_file, "r", encoding="utf-8") as f:
                content = f.read()
            arrays = re.findall(r'\[.*?\]', content, flags=re.S)
            data = []
            for arr in arrays:
                try:
                    data.extend(json.loads(arr))
                except:
                    continue
            return data

    def _process_item(self, item):

        gt_answer = str(item.get("answer") or item.get("original_answer", ""))
        pred_answer = str(item.get("response") or item.get("system_answer", ""))
        question = str(item.get("question", ""))
        category = str(item.get("category", "0"))

        if category == "5":
            return None

        if self.dataset_type == "locomo":
            metrics = calculate_metrics(pred_answer, gt_answer)
            bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
            llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

            qid = item.get("sample_id", "")
            qtype = str(item.get("question_type", ""))

            prompt = get_anscheck_prompt(qtype, question, gt_answer, pred_answer)
            kwargs = {
                "model": "LLAMA",
                "messages": [{"role": "user", "content": prompt}],
                "n": 1,
                "temperature": 0,
                "max_tokens": 10,
            }
            completion = chat_completions_with_backoff(self.metric_client, **kwargs)
            eval_response = completion.choices[0].message.content.strip()
            label = "yes" in eval_response.lower()

            return {
                "sample_id": qid,
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "question_type": qtype,
                "category": category,
                "f1_score": metrics.get("f1"),
                "coverage": bleu_scores.get("bleu1"),
                "accuracy": 1 if label else 0,
                "llm_score": llm_score, 
            }

    def evaluate(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_item, item) for item in self.data]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    with self.lock:
                        self.results[result["category"]].append(result)

    def aggregate(self):
        summary = {}
        all_accuracy_scores = []
        for category, items in self.results.items():
            f1_scores = [x.get("f1_score") for x in items if x.get("f1_score") is not None]
            bleu_scores = [
                x.get("bleu_score") or x.get("coverage")
                for x in items
                if (x.get("bleu_score") is not None or x.get("coverage") is not None)
            ]
            llm_scores = [x.get("llm_score") for x in items if x.get("llm_score") is not None]
            accuracy_scores = [x.get("accuracy") for x in items if x.get("accuracy") is not None]
            all_accuracy_scores.extend(accuracy_scores)
            summary[category] = {
                "count": len(items),
                "f1_score": statistics.mean(f1_scores) if f1_scores else None,
                "bleu_score": statistics.mean(bleu_scores) if bleu_scores else None,
                "llm_score": statistics.mean(llm_scores) if llm_scores else None,
                "accuracy": statistics.mean(accuracy_scores) if accuracy_scores else None,
            }
        summary["overall_acc"] = {
            "accuracy": statistics.mean(all_accuracy_scores) if all_accuracy_scores else None,
        }

        return summary

    def save(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        print(f"Saved evaluation results to {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON/JSONL file")
    parser.add_argument("--max_workers", type=int, default=8, help="Max parallel workers")
    parser.add_argument("--dataset_type", type=str, default="locomo", choices=["locomo", "longmemeval"])
    parser.add_argument("--metric_model", type=str, default="openai")
    args = parser.parse_args()

    evaluator = Evaluation(
        input_file=args.input_file,
        max_workers=args.max_workers,
        dataset_type=args.dataset_type,
        metric_model_short=args.metric_model
    )
    evaluator.evaluate()
    summary = evaluator.aggregate()
    evaluator.save()

    print("Summary (per category):")
    for cat, metrics in summary.items():
        if cat == "overall_acc":
            continue
        print(f"Category {cat}: {metrics}")

    print("\nSummary (overall):")
    print(f"Overall accuracy: {summary['overall_acc']['accuracy']:.4f}")

# python /home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py --input_file /home/shm/document/MEMORY_ACCLERATION/results/glm-4-9b-chat-1m-GGUF_HippoRAG_locomo10_mem_bf/cleaned_results.jsonl --dataset_type locomo
# python /home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py --input_file /home/shm/document/MEMORY_ACCLERATION/results/MemoryOS_longmemeval_m_results.jsonl --dataset_type longmemeval
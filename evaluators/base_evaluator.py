# # 暂时懒得封装了就直接用现有的轮子吧~
# import os
# import json
# import re
# import threading
# import statistics
# import argparse
# import concurrent.futures
# from typing import List, Dict
# from collections import defaultdict
# from tqdm import tqdm
# import backoff
# from metrics.llm_judge import evaluate_llm_judge
# from metrics.utils import calculate_bleu_scores, calculate_metrics
# import openai
# from openai import OpenAI

# model_zoo = {
#     'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
#     'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
#     'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
# }


# @backoff.on_exception(backoff.expo, (openai.RateLimitError,
#                                     openai.APIError))
# def chat_completions_with_backoff(client, **kwargs):
#     return client.chat.completions.create(**kwargs)


# def get_anscheck_prompt(task, question, answer, response, abstention=False):
#     if not abstention:
#         if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
#             template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
#             prompt = template.format(question, answer, response)
#         elif task == 'temporal-reasoning':
#             template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
#             prompt = template.format(question, answer, response)
#         elif task == 'knowledge-update':
#             template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
#             prompt = template.format(question, answer, response)
#         elif task == 'single-session-preference':
#             template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
#             prompt = template.format(question, answer, response)
#         else:
#             raise NotImplementedError
#     else:
#         template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
#         prompt = template.format(question, answer, response) 
#     return prompt


# class Evaluation:
#     def __init__(self, input_file: str, max_workers: int = 8, dataset_type: str = "locomo"):
#         self.input_file = input_file
#         self.output_file = self._make_output_path(input_file)
#         self.max_workers = max_workers
#         self.data = self._load_data()
#         self.results = defaultdict(list)
#         self.lock = threading.Lock()
#         self.dataset_type = dataset_type.lower()  # locomo / longmemeval

#     def _make_output_path(self, input_file: str) -> str:
#         """在 input_file 基础上生成 *_eval.json"""
#         base, _ = os.path.splitext(input_file)
#         return base + "_eval.json"

#     def _load_data(self):
#         """支持 JSON / JSONL / 含数组片段的文本"""
#         try:
#             with open(self.input_file, "r", encoding="utf-8") as f:
#                 if self.input_file.endswith(".jsonl"):
#                     return [json.loads(line) for line in f if line.strip()]
#                 else:
#                     return json.load(f)
#         except json.JSONDecodeError:
#             with open(self.input_file, "r", encoding="utf-8") as f:
#                 content = f.read()
#             arrays = re.findall(r'\[.*?\]', content, flags=re.S)
#             data = []
#             for arr in arrays:
#                 try:
#                     data.extend(json.loads(arr))
#                 except:
#                     continue
#             return data

#     def _process_item(self, item):
#         """单条样本处理：计算 F1 / BLEU / LLM judge"""
#         gt_answer = str(item.get("answer") or item.get("original_answer", ""))
#         pred_answer = str(item.get("response") or item.get("system_answer", ""))
#         question = str(item.get("question", ""))
#         category = str(item.get("category", "0"))

#         if category == "5":  
#             return None

#         if self.dataset_type == "locomo":
#             metrics = calculate_metrics(pred_answer, gt_answer)       # f1
#             bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)  # bleu1, bleu2...
#             llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)  # llm打分

#             return {
#                 "question": question,
#                 "answer": gt_answer,
#                 "response": pred_answer,
#                 "category": category,
#                 "f1_score": metrics.get("f1"),
#                 "bleu_score": bleu_scores.get("bleu1"),
#                 "llm_score": llm_score,
#             }
        
#         elif self.dataset_type == "longmemeval":
#             # ------------------- LongMemEval 逻辑 -------------------
#             qid = item.get("question_id")
#             if not qid or qid not in self.qid2qtype:
#                 return None

#             qtype = self.qid2qtype[qid]
#             qdata = self.qid2qdata[qid]
#             q_text = qdata.get("question", question)
#             ans_text = qdata.get("answer", gt_answer)

#             # 构建 prompt
#             prompt = get_anscheck_prompt(
#                 qtype, q_text, ans_text, pred_answer, abstention="_abs" in str(qid)
#             )
#             kwargs = {
#                 "model": self.metric_model,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "n": 1,
#                 "temperature": 0,
#                 "max_tokens": 10
#             }
#             completion = chat_completions_with_backoff(self.metric_client, **kwargs)
#             eval_response = completion.choices[0].message.content.strip()
#             label = "yes" in eval_response.lower()

#             # 保留 F1/BLEU 作为辅助指标
#             metrics = calculate_metrics(pred_answer, ans_text)
#             bleu_scores = calculate_bleu_scores(pred_answer, ans_text)

#             return {
#                 "question": q_text,
#                 "answer": ans_text,
#                 "response": pred_answer,
#                 "category": category,
#                 "f1_score": metrics.get("f1"),
#                 "coverage": bleu_scores.get("bleu1"),  # 示例：BLEU1 作为 coverage
#                 "accuracy": 1 if label else 0,
#                 "llm_score": metrics.get("f1"),  # 可用 F1 代替 LLM score
#             }
#     def evaluate_(self):
#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = [executor.submit(self._process_item, item) for item in self.data]

#             for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#                 result = future.result()
#                 if result:
#                     with self.lock:
#                         self.results[result["category"]].append(result)

#     def aggregate(self):
#         """计算每个类别的平均指标"""
#         summary = {}
#         for category, items in self.results.items():
#             f1_scores = [x["f1_score"] for x in items if x["f1_score"] is not None]
#             bleu_scores = [x["bleu_score"] for x in items if x["bleu_score"] is not None]
#             llm_scores = [x["llm_score"] for x in items if x["llm_score"] is not None]

#             summary[category] = {
#                 "count": len(items),
#                 "f1_score": statistics.mean(f1_scores) if f1_scores else None,
#                 "bleu_score": statistics.mean(bleu_scores) if bleu_scores else None,
#                 "llm_score": statistics.mean(llm_scores) if llm_scores else None,
#             }
#         return summary

#     def save(self):
#         with open(self.output_file, "w", encoding="utf-8") as f:
#             json.dump(self.results, f, indent=4, ensure_ascii=False)
#         print(f"Saved evaluation results to {self.output_file}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_file", type=str, required=True)
#     parser.add_argument("--max_workers", type=int, default=8)
#     args = parser.parse_args()

#     evaluator = Evaluation(args.input_file, args.max_workers)
#     evaluator.evaluate()
#     summary = evaluator.aggregate()
#     evaluator.save()

#     print("Summary (per category):")
#     for cat, metrics in summary.items():
#         print(f"Category {cat}: {metrics}")
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

# 带退避的 LLM 调用
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


# LongMemEval 用的 prompt 构建
def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = ("I will give you a question, a correct answer, and a response from a model. "
                        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                        "If the response is equivalent to the correct answer or contains all the intermediate steps "
                        "to get the correct answer, you should also answer yes. "
                        "If the response only contains a subset of the information required by the answer, answer no.\n\n"
                        "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                        "Is the model response correct? Answer yes or no only.")
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = ("I will give you a question, a correct answer, and a response from a model. "
                        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                        "If the response is equivalent to the correct answer or contains all the intermediate steps "
                        "to get the correct answer, you should also answer yes. "
                        "If the response only contains a subset of the information required by the answer, answer no. "
                        "In addition, do not penalize off-by-one errors for the number of days. "
                        "If the question asks for the number of days/weeks/months, etc., "
                        "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), "
                        "the model's response is still correct.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                        "Is the model response correct? Answer yes or no only.")
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = ("I will give you a question, a correct answer, and a response from a model. "
                        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                        "If the response contains some previous information along with an updated answer, "
                        "the response should be considered as correct as long as the updated answer is the required answer.\n\n"
                        "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                        "Is the model response correct? Answer yes or no only.")
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = ("I will give you a question, a rubric for desired personalized response, and a response from a model. "
                        "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                        "The model does not need to reflect all the points in the rubric. "
                        "The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
                        "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.")
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = ("I will give you an unanswerable question, an explanation, and a response from a model. "
                    "Please answer yes if the model correctly identifies the question as unanswerable. "
                    "The model could say that the information is incomplete, or some other information is given "
                    "but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
                    "Does the model correctly identify the question as unanswerable? Answer yes or no only.")
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

        if dataset_type == "longmemeval":
            # if metric_model_short not in model_zoo:
            #     raise ValueError(f"Unsupported metric model: {metric_model_short}")
            # metric_model, metric_model_source = model_zoo[metric_model_short]
            if metric_model_short == "openai":
                openai.organization = os.getenv("OPENAI_ORGANIZATION")
                openai_api_key = os.getenv("OPENAI_API_KEY")
                openai_api_base = os.environ.get("OPENAI_API_BASE")
            # else:
            #     openai_api_key = "EMPTY"
            #     openai_api_base = "http://localhost:8001/v1"

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
        """处理单条样本"""
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

            return {
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "category": category,
                "f1_score": metrics.get("f1"),
                "bleu_score": bleu_scores.get("bleu1"),
                "llm_score": llm_score,
            }

        elif self.dataset_type == "longmemeval":
            qid = item.get("sample_id", "")
            qtype = str(item.get("question_type", ""))
            q_text = question
            ans_text = gt_answer

            # 构建 prompt
            prompt = get_anscheck_prompt(qtype, q_text, ans_text, pred_answer, abstention="_abs" in str(qid))
            kwargs = {
                "model": self.metric_model,
                "messages": [{"role": "user", "content": prompt}],
                "n": 1,
                "temperature": 0,
                "max_tokens": 10,
            }
            completion = chat_completions_with_backoff(self.metric_client, **kwargs)
            eval_response = completion.choices[0].message.content.strip()
            label = "yes" in eval_response.lower()

            # 保留 F1/BLEU 作为辅助指标
            metrics = calculate_metrics(pred_answer, ans_text)
            bleu_scores = calculate_bleu_scores(pred_answer, ans_text)

            return {
                "sample_id": qid,
                "question": q_text,
                "answer": ans_text,
                "response": pred_answer,
                "question_type": qtype,
                "category": category,
                "f1_score": metrics.get("f1"),
                "coverage": bleu_scores.get("bleu1"),
                "accuracy": 1 if label else 0,
                "llm_score": metrics.get("f1"),  # 暂时用 F1 代替
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
        """计算每个类别的平均指标"""
        summary = {}
        for category, items in self.results.items():
            f1_scores = [x.get("f1_score") for x in items if x.get("f1_score") is not None]
            bleu_scores = [x.get("bleu_score") or x.get("coverage") for x in items if x.get("bleu_score") or x.get("coverage") is not None]
            llm_scores = [x.get("llm_score") for x in items if x.get("llm_score") is not None]
            accuracy_scores = [x.get("accuracy") for x in items if x.get("accuracy") is not None]

            summary[category] = {
                "count": len(items),
                "f1_score": statistics.mean(f1_scores) if f1_scores else None,
                "bleu_score": statistics.mean(bleu_scores) if bleu_scores else None,
                "llm_score": statistics.mean(llm_scores) if llm_scores else None,
                "accuracy": statistics.mean(accuracy_scores) if accuracy_scores else None,
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
        print(f"Category {cat}: {metrics}")

# python /home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py --input_file /home/shm/document/MEMORY_ACCLERATION/results/MemoryOS_locomo10_results.jsonl --dataset_type locomo
# python /home/shm/document/MEMORY_ACCLERATION/evaluators/base_evaluator.py --input_file /home/shm/document/MEMORY_ACCLERATION/results/MemoryOS_longmemeval_m_results.jsonl --dataset_type longmemeval
import yaml
# from datasets.longmemeval import LongMemEvalLoader
# from models.memoryos import MemoryOS
# from evaluators.longmemeval_eval import LongMemEvalEvaluator
import os
import json
from datetime import datetime
from importlib import import_module
import argparse
# from model.MemoryOS.MemoryOS_module import MemoryOSModel
# from model.memo0.memo_module import Memo0Model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.join(BASE_DIR, "MEMORY_ACCLERATION"),"dataset")
OUTPUT_DIR = os.path.join(os.path.join(BASE_DIR, "MEMORY_ACCLERATION"),"results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_finished_ids(output_file):
    finished_ids = set()
    if not os.path.exists(output_file):
        return finished_ids
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("sample_id") or obj.get("question_id")
                if sid is not None:
                    finished_ids.add(str(sid))
            except json.JSONDecodeError: # 忽略可能的坏行
                continue
    return finished_ids

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
import os
import time
import subprocess
import requests
# response = requests.get("http://localhost:30086/metrics", timeout=10)

def dump_sglang_metrics(metrics_dir):
    metrics_file = os.path.join(metrics_dir, f"metrics_sglang_log.txt")
    url = os.environ.get("OPENAI_API_METRICS")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(metrics_file, 'w') as f:
                f.write(response.text)  
            print(f"Metrics saved to {metrics_file}")
        else:
            print(f"Failed to fetch. Server responded with status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch metrics, server may not expose /metrics. Error: {e}")

import os
import json
from importlib import import_module
from tqdm import tqdm


def run_pipeline(models: list, datasets: list):

    exp_dir = os.environ.get("EXP_RESULTS_DIR")
    if exp_dir is None:
        raise RuntimeError("EXP_RESULTS_DIR is not set")

    os.makedirs(exp_dir, exist_ok=True)

    for model_name in models:
        tqdm.write(f"\n=== Running model: {model_name} ===")

        try:
            model_module = import_module(f"model.{model_name}.{model_name}_module")
            model_class_name = f"{model_name}Model"
            model_instance = getattr(model_module, model_class_name)()
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to load model {model_name}: {e}")
            continue

        for dataset in datasets:
            dataset_name = dataset["name"]
            dataset_path = dataset["path"]

            tqdm.write(f"\n--- Evaluating dataset: {dataset_name} ---")

            output_file = os.path.join(
                exp_dir,
                f"{model_name}_{dataset_name}_generation_results.jsonl"
            )

            # 读取数据集
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tqdm.write(f"Total {len(data)} samples")
            except FileNotFoundError:
                tqdm.write(f"[ERROR] Cannot find dataset file: {dataset_path}")
                continue
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to load dataset {dataset_path}: {e}")
                continue

            # 中断恢复
            finished_ids = load_finished_ids(output_file)
            if finished_ids:
                tqdm.write(f"Found {len(finished_ids)} finished samples, resuming...")
            else:
                tqdm.write("No existing results found, starting from scratch.")

            try:
                with tqdm(
                    total=len(data),
                    desc=f"{model_name} | {dataset_name}",
                    unit="sample",
                    dynamic_ncols=True,
                ) as pbar:

                    for idx, sample in enumerate(data):
                        sample_id = (
                            sample.get("sample_id")
                            or sample.get("question_id")
                            or f"sample_{idx + 1}"
                        )
                        sample_id = str(sample_id)

                        if sample_id in finished_ids:
                            pbar.update(1)
                            continue

                        pbar.set_postfix_str(f"id={sample_id}")

                        try:
                            model_instance.generate_answer(
                                idx,
                                sample,
                                dataset_name,
                                output_file
                            )
                            dump_sglang_metrics(exp_dir)

                        except Exception as e:
                            tqdm.write(f"[ERROR] Sample {sample_id} failed: {e}")

                        pbar.update(1)

            except KeyboardInterrupt:
                tqdm.write("\n[INTERRUPTED] Caught KeyboardInterrupt, exiting safely.")
                tqdm.write(f"Progress saved in {output_file}")
                return

            tqdm.write(f"Dataset {dataset_name} finished. Results saved to {output_file}")

def load_config(config_path: str):
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported config file format. Use .yaml/.yml or .json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory Evaluation Pipeline")
    parser.add_argument(
        "--models", nargs="+", default=["MemoryOS"],
        help="e.g., --models MemoryOS Memo0"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["locomo10"],
    )
    parser.add_argument(
        "--config", type=str, default=None,
    )
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        models_to_run = config.get("models", [])
        datasets_to_run = config.get("datasets", [])
    else:
        models_to_run = args.models
        datasets_to_run = [
            {"name": name, "path": os.path.join(DATA_DIR, name, f"{name}.json")}
            if not name.startswith("longmemeval") else {"name": name, "path": os.path.join(DATA_DIR, "longmemeval", f"{name}.json")} for name in args.datasets
        ]
    
    run_pipeline(models=models_to_run, datasets=datasets_to_run)

# if __name__ == "__main__":
#     # configuration
#     models_to_run = ["MemoryOS"]  # Corresponding to model/MemoryOS, model/memo0 "MeMo0"

#     # BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
#     # DATA_DIR = os.path.join(BASE_DIR, "dataset")           
#     datasets_to_run = [
#         {"name": "locomo10", "path": os.path.join(os.path.join(DATA_DIR, "locomo10"),"locomo10.json")},
#         # {"name": "longmemeval_s", "path":  os.path.join(DATA_DIR, "longmemeval_s.json")},
#         # {"name": "longmemeval_m", "path":  os.path.join(DATA_DIR, "longmemeval_m.json")},
#         # {"name": "longmemeval_oracle", "path":  os.path.join(DATA_DIR, "longmemeval_oracle.json")},
#     ]

#     run_pipeline(models=models_to_run, datasets=datasets_to_run)
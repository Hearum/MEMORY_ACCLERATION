import yaml
# from datasets.longmemeval import LongMemEvalLoader
# from models.memoryos import MemoryOS
# from evaluators.longmemeval_eval import LongMemEvalEvaluator
import os
import json
from datetime import datetime
from importlib import import_module


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_pipeline(models: list, datasets: list, output_dir: str = "results"):
    """
    Unified pipeline to evaluate multiple models on multiple datasets.

    Args:
        models: list of model names (must correspond to model/<model_name>.py)
        datasets: list of dicts: {"name": "locomo10", "path": "/path/to/data.json"}
        output_dir: directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    for model_name in models:
        print(f"\n=== Running model: {model_name} ===")
        # Dynamically import the model module
        try:
            model_module = import_module(f"model.{model_name}")
            model_client = model_module.get_client()  # Each model exposes a get_client() function
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        for dataset in datasets:
            dataset_name = dataset["name"]
            dataset_path = dataset["path"]
            print(f"\n--- Evaluating dataset: {dataset_name} ---")

            # Create memory/temp directory
            mem_dir = os.path.join(output_dir, f"{model_name}_{dataset_name}_mem")
            os.makedirs(mem_dir, exist_ok=True)

            # Output jsonl file for evaluation script
            output_file = os.path.join(output_dir, f"{model_name}_{dataset_name}_results.jsonl")

            # Load dataset
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"Loaded {len(data)} samples from {dataset_name}")
            except Exception as e:
                print(f"Failed to load dataset {dataset_name}: {e}")
                continue

            results = []

            # Iterate samples
            for idx, sample in enumerate(data):
                sample_id = sample.get("sample_id") or sample.get("question_id") or f"sample_{idx+1}"
                print(f"[{idx+1}/{len(data)}] Processing sample: {sample_id}")

                try:
                    # Each model module exposes a process_sample function
                    hypothesis = model_module.process_sample(
                        sample,
                        client=model_client,
                        mem_dir=mem_dir
                    )
                    results.append({
                        "question_id": sample_id,
                        "hypothesis": hypothesis
                    })

                except Exception as e:
                    print(f"Error processing sample {sample_id}: {e}")
                    continue

                # Save intermediate results
                with open(output_file, "w", encoding="utf-8") as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            print(f"Finished dataset {dataset_name}, results saved to {output_file}")


if __name__ == "__main__":
    # configuration
    models_to_run = ["MemoryOS", "MoMo0"]  # Corresponding to model/MemoryOS.py, model/MoMo0.py

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
    DATA_DIR = os.path.join(BASE_DIR, "dataset")           
    # 读取文件
    datasets_to_run = [
        {"name": "locomo10", "path": os.path.join(DATA_DIR, "locomo10.json")},
        {"name": "longmemeval_s", "path":  os.path.join(DATA_DIR, "longmemeval_s.json")},
        {"name": "longmemeval_m", "path":  os.path.join(DATA_DIR, "longmemeval_m.json")},
        {"name": "longmemeval_oracle", "path":  os.path.join(DATA_DIR, "longmemeval_oracle.json")},
    ]

    run_pipeline(models=models_to_run, datasets=datasets_to_run, output_dir="results")

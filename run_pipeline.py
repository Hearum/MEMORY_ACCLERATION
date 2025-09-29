import yaml
# from datasets.longmemeval import LongMemEvalLoader
# from models.memoryos import MemoryOS
# from evaluators.longmemeval_eval import LongMemEvalEvaluator
import os
import json
from datetime import datetime
from importlib import import_module

# from model.MemoryOS.MemoryOS_module import MemoryOSModel
# from model.memo0.memo_module import Memo0Model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.join(BASE_DIR, "MEMORY_ACCLERATION"),"dataset")
OUTPUT_DIR = os.path.join(os.path.join(BASE_DIR, "MEMORY_ACCLERATION"),"results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_pipeline(models: list, datasets: list):

    for model_name in models:
        print(f"\n=== Running model: {model_name} ===")
        # try:
        model_module = import_module(f"model.{model_name}.{model_name}_module")
        model_class_name = f"{model_name}Model"
        model_instance = getattr(model_module, model_class_name)()
        # except Exception as e:
        #     print(f"Failed to load model {model_name}: {e}")
        #     continue

        for dataset in datasets:
            dataset_name = dataset["name"]
            dataset_path = dataset["path"]
            
            print(f"\n--- Evaluating dataset: {dataset_name} ---")

            # Create memory/temp directory
            # output_file = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset_path.replace('.json','')}_results.json")
            mem_dir = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset_name}_mem")
            os.makedirs(mem_dir, exist_ok=True)
            # Output jsonl file for evaluation script
            output_file = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset_name}_results.jsonl")

            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"total {len(data)} sample")
            except FileNotFoundError:
                print(f"cannot find {dataset_name} file, make sure it in dir")
                return
            except Exception as e:
                print(f"somethin woring happen in loading dataset:{e}")
                continue

            results = []

            for idx, sample in enumerate(data[:1]):
                sample_id = sample.get("sample_id") or sample.get("question_id") or f"sample_{idx+1}"
                import pdb
                pdb.set_trace()

                print(f"[{idx+1}/{len(data)}] Processing sample: {sample_id}")
                # try:
                model_instance.generate_answer(idx, sample,dataset_name,output_file)
                # print(f"[DEBUG] Raw model output for {sample_id}:\n{system_answer}")
                # if system_answer is None:
                #     continue
                # results.append({
                #     "question_id": sample_id,
                #     "hypothesis": system_answer
                # })
                # except Exception as e:
                #     print(f"Error processing sample {sample_id}: {e}")
                #     continue

                # save
                # try:
                #     with open(output_file, "a", encoding="utf-8") as f:
                #         for item in results:
                #             f.write(json.dumps(item, ensure_ascii=False) + "\n")
                # except Exception as e:
                #     print(f"Failed to save intermediate results: {e}")

            print(f"Dataset {dataset_name} processed. Results saved to {output_file}")

if __name__ == "__main__":
    # configuration
    models_to_run = ["MemoryOS"]  # Corresponding to model/MemoryOS, model/memo0 "MeMo0"

    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
    # DATA_DIR = os.path.join(BASE_DIR, "dataset")           
    datasets_to_run = [
        {"name": "locomo10", "path": os.path.join(os.path.join(DATA_DIR, "locomo10"),"locomo10.json")},
        # {"name": "longmemeval_s", "path":  os.path.join(DATA_DIR, "longmemeval_s.json")},
        # {"name": "longmemeval_m", "path":  os.path.join(DATA_DIR, "longmemeval_m.json")},
        # {"name": "longmemeval_oracle", "path":  os.path.join(DATA_DIR, "longmemeval_oracle.json")},
    ]

    run_pipeline(models=models_to_run, datasets=datasets_to_run)
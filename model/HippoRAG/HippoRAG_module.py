import os
import json
import time
from collections import defaultdict
from jinja2 import Template
from openai import OpenAI
import numpy as np
import tiktoken
from tqdm import tqdm
import pdb
from .hipporag.HippoRAG import HippoRAGModule
import shutil

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def clear_save_(save_dir):
    # 遍历目录，删除其中的所有文件
    for root, dirs, files in os.walk(save_dir, topdown=False):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.remove(file_path)  # 删除文件

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  

def clear_folder_contents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子目录
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

class HippoRAGModel:
    def __init__(self, chunk_size=500, save_dir='outputs_locomo10',topk=20):

        self.topk= topk
        self.save_dir= save_dir
        # self.hipporag = HippoRAGModule(save_dir=save_dir, 
        #     llm_model_name='LLAMA',
        #     llm_base_url=os.getenv("OPENAI_API_BASE"),
        #     embedding_model_name="VLLM//mnt/data3/models/Qwen3-Embedding-4B",  
        #     embedding_base_url="http://localhost:8000/v1"
        #     )
        
        # self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") #os.getenv("EMBEDDING_MODEL")
        # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
        # self.client_embedding = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="http://localhost:30099/v1") # embed model写死了，启动脚本在/home/shm/document/MEMORY_ACCLERATION/scripts/begin_embed_model.sh
    
    def generate_answer(self, idx, sample, dataset_name, output_file):

        hippo = HippoRAGModule(
            save_dir=f"{os.environ.get('EXP_RESULTS_DIR')}/memory_temp/{idx}",
            llm_model_name="LLAMA",
            llm_base_url=os.getenv("OPENAI_API_BASE"),
            embedding_model_name="VLLM//mnt/data3/models/Qwen3-Embedding-4B", 
            embedding_base_url="http://localhost:8000/v1"
        )

        sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")

        if dataset_name == "locomo10":
            conversation = sample.get("conversation", {})
            chat_history = []
            for key, chats in conversation.items():
                if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                    continue
                for c in chats:
                    chat_history.append({
                        "speaker": c["speaker"],
                        "text": c["text"],
                        "timestamp": c.get("timestamp", "unknown")
                    })
            qa_pairs = sample.get("qa", [])
            speaker_a = conversation.get("speaker_a", "User")
            speaker_b = conversation.get("speaker_b", "Assistant")

        elif dataset_name.startswith("longmemeval"):
            chat_history = []
            sessions = sample.get("haystack_sessions", [])
            dates = sample.get("haystack_dates", [])

            for i, session in enumerate(sessions):
                timestamp = dates[i] if i < len(dates) else "unknown"
                for turn in session:
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "").strip()
                    if not content:
                        continue
                    chat_history.append({
                        "speaker": "User" if role.lower() == "user" else "Assistant",
                        "text": content,
                        "timestamp": timestamp
                    })

            qa_pairs = [{
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "question_id": sample.get("question_id", ""),
                "question_type": sample.get("question_type", ""),
                "question_date": sample.get("question_date", "")
            }]

            speaker_a = "User"
            speaker_b = "Assistant"

        else:
            raise ValueError(f"Unsupported dataset type: {dataset_name}")

        docs = [f"[{c['speaker']}] {c['text']}" for c in chat_history]
        if not docs:
            print(f"⚠️ Sample {sample_id} has empty docs, skipped.")
            return
        try:
            hippo.index(docs=docs)
        except Exception as e:
            print(f"❌ Indexing failed for sample {sample_id}: {e}")
            return

        # === 5: Retrieval ===
        outputs = []
        for qa in qa_pairs:
            question = qa["question"]
            original_answer = qa.get("answer", "")

            try:
                retrieval_result = hippo.retrieve(queries=[question], num_to_retrieve=self.topk)
            except Exception as e:
                print(f"❌ Retrieval failed for {sample_id}, question: {question}: {e}")
                continue

            try:
                qa_result = hippo.rag_qa(retrieval_result)
            except Exception as e:
                print(f"❌ RAG QA failed for {sample_id}, question: {question}: {e}")
                continue

            if not qa_result or not qa_result[0]:
                continue
            
            system_answer = str(qa_result[0][0].answer) if isinstance(qa_result[0], list) else str(qa_result[0].answer)
            result = {
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "original_answer": original_answer,
                "question": question,
                # "retrieval_result": str(retrieval_result),
                "system_answer": system_answer,
                "timestamp": get_timestamp(),
            }

            if "category" in qa:
                result["category"] = qa.get("category")

            if "question_type" in qa:
                result["question_type"] = qa.get("question_type")

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"✅ Sample {sample_id} processed and saved.")
        clear_folder_contents(f"{os.environ.get('EXP_RESULTS_DIR')}/memory_temp/{idx}")
            # outputs.append({
            #     "sample_id": sample_id,
            #     "speaker_a": speaker_a,
            #     "speaker_b": speaker_b,
            #     "original_answer": original_answer,
            #     "question": question,
            #     # "retrieval_result":str(retrieval_result),
            #     "system_answer": system_answer,
            #     "timestamp": get_timestamp(),
            #     **({"category": qa.get("category")} if "category" in qa else {}),
            #     **({"question_type": qa.get("question_type")} if "question_type" in qa else {})
            # })

        # === 8: append to output file ===
        # os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # with open(output_file, "a", encoding="utf-8") as f:
        #     json.dump(outputs, f, ensure_ascii=False, indent=2)
        #     f.write("\n")

        # print(f"✅ Sample {sample_id} processed and saved.")

    # def generate_answer(self, idx, sample, dataset_name, output_file):

    #     sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")

    #     # ===== Step 1: Parse conversation =====
    #     if dataset_name == "locomo10":
    #         conversation = sample.get("conversation", {})
    #         chat_history = []
    #         for key, chats in conversation.items():
    #             if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
    #                 continue
    #             for c in chats:
    #                 chat_history.append({
    #                     "speaker": c["speaker"],
    #                     "text": c["text"],
    #                     "timestamp": c.get("timestamp", "unknown")
    #                 })
    #         qa_pairs = sample.get("qa", [])
    #         speaker_a = conversation.get("speaker_a", "User")
    #         speaker_b = conversation.get("speaker_b", "Assistant")

    #     elif dataset_name.startswith("longmemeval"):
    #         chat_history = []
    #         sessions = sample.get("haystack_sessions", [])
    #         dates = sample.get("haystack_dates", [])

    #         for i, session in enumerate(sessions):
    #             timestamp = dates[i] if i < len(dates) else "unknown"
    #             for turn in session:
    #                 role = turn.get("role", "unknown")
    #                 content = turn.get("content", "").strip()
    #                 if not content:
    #                     continue  # 跳过空白回合
    #                 chat_history.append({
    #                     "speaker": "User" if role.lower() == "user" else "Assistant",
    #                     "text": content,
    #                     "timestamp": timestamp
    #                 })

    #         qa_pairs = [{
    #             "question": sample.get("question", ""),
    #             "answer": sample.get("answer", ""),
    #             "question_id": sample.get("question_id", ""),
    #             "question_type": sample.get("question_type", ""),
    #             "question_date": sample.get("question_date", "")
    #         }]

    #         speaker_a = "User"
    #         speaker_b = "Assistant"

    #     else:
    #         raise ValueError(f"Unsupported dataset type: {dataset_name}")

    #     # if not chat_history:
    #     #     print(f"⚠️ Sample {sample_id} has no valid conversation data, skipping.")
    #     #     return
        
    #     docs = [f"[{c['speaker']}] {c['text']}" for c in chat_history]
    #     if not docs:
    #         print(f"⚠️ Sample {sample_id} has empty docs, skipped.")
    #         return
        
    #     # pdb.set_trace()
    #     try:
    #         self.hipporag.index(docs=docs)
    #     except Exception as e:
    #         print(e)
    #         return
        
    #     results = []

    #     # ===== Generate answers =====
    #     results = []
    #     questions = [qa.get("question", "") for qa in qa_pairs]
    #     original_answers = [qa.get("answer", "") for qa in qa_pairs]
        
    #     # Retrieve step
    #     retrieval_results = self.hipporag.retrieve(queries=questions, num_to_retrieve=self.topk)

    #     # QA step
    #     qa_results = self.hipporag.rag_qa(retrieval_results)
        
    #     for qa, system_answer_list in zip(qa_pairs, qa_results):
    #         if not system_answer_list:
    #             continue
    #         pdb.set_trace()
    #         system_answer = system_answer_list[0]  # QuerySolution 对象
    #         question = qa.get("question", "")
    #         original_answer = qa.get("answer", "")
            
    #         results.append({
    #             "sample_id": sample_id,
    #             "speaker_a": speaker_a,
    #             "speaker_b": speaker_b,
    #             "question": question,
    #             "system_answer": system_answer.answer,
    #             "original_answer": original_answer,
    #             "timestamp": get_timestamp(),
    #             **({"category": qa.get("category")} if "category" in qa else {}),
    #             **({"question_type": qa.get("question_type")} if "question_type" in qa else {}),
    #         })

    #     # ===== Step 4: Save results =====
    #     clear_save_(self.save_dir)
    #     ensure_dir_exists(self.save_dir)
    #     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    #     try:
    #         with open(output_file, "a", encoding="utf-8") as f:
    #             json.dump(results, f, ensure_ascii=False, indent=2)
    #             f.write("\n")
    #         print(f"✅ Sample {sample_id} processed and saved in {output_file}")
    #     except Exception as e:
    #         print(f"⚠️ Error saving sample {sample_id}: {e}")
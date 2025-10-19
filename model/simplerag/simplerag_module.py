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
from hipporag import HippoRAG


def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

class HippoRAGModel:
    def __init__(self, chunk_size=500, save_dir='outputs',topk=10):

        self.model = "LLAMA"
        self.topk= topk
        self.hipporag = HippoRAG(save_dir=save_dir, 
            llm_model_name='Your LLM Model name',
            llm_base_url=os.getenv("OPENAI_API_BASE"),
            embedding_model_name='Your Embedding model name',  
            embedding_base_url='http://localhost:30099/v1')
        
        # self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") #os.getenv("EMBEDDING_MODEL")
        # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
        # self.client_embedding = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="http://localhost:30099/v1") # embed model写死了，启动脚本在/home/shm/document/MEMORY_ACCLERATION/scripts/begin_embed_model.sh
    

    def generate_answer(self, idx, sample, dataset_name, output_file):

        sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")

        # ===== Step 1: Parse conversation =====
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
                        continue  # 跳过空白回合
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

        # if not chat_history:
        #     print(f"⚠️ Sample {sample_id} has no valid conversation data, skipping.")
        #     return
        
        docs = [f"[{c['speaker']}] {c['text']}" for c in chat_history]
        if not docs:
            print(f"⚠️ Sample {sample_id} has empty docs, skipped.")
            return
        
        save_dir = output_file

        self.hipporag.index(docs=docs)
        results = []

        # ===== Generate answers =====
        results = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            original_answer = qa.get("answer", "")

            # --- Retrieve step ---
            retrieval_results = self.hipporag.retrieve(queries=[question], num_to_retrieve=self.topk)

            # --- QA step ---
            qa_results = self.hipporag.rag_qa(retrieval_results)
            system_answer = qa_results[0] if qa_results else ""

            # 保存检索到的文档，方便评估
            retrieved_docs = retrieval_results[0] if retrieval_results else []

            results.append({
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": question,
                "system_answer": system_answer,
                "original_answer": original_answer,
                # "retrieved_docs": retrieved_docs,  # 新增字段
                "timestamp": get_timestamp(),
                **({"category": qa.get("category")} if "category" in qa else {}),
                **({"question_type": qa.get("question_type")} if "question_type" in qa else {}),
            })

        # ===== Step 4: Save results =====
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"✅ Sample {sample_id} processed and saved in {output_file}")
        except Exception as e:
            print(f"⚠️ Error saving sample {sample_id}: {e}")
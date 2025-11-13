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
PROMPT = """
You are a helpful and concise assistant.

Below is the complete dialogue history between the user and assistant, followed by the user's new question.
Use the information in the dialogue as context to answer accurately and directly.

# Dialogue History:
{{HISTORY}}

# User Question:
{{QUESTION}}

# Your Answer:
"""

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

token_logs = []
log_file = "/home/shm/document/log/QA/log_o.json"
class QAModel:
    def __init__(self, chunk_size=500, top_k=10):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.model = "LLAMA"
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                             base_url=os.getenv("OPENAI_API_BASE"))
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 9000000
        self.seed = 42

    def generate_response_no_rag(self, question, history_text):
        template = Template(PROMPT)
        tokens = self.tokenizer.encode(history_text)
        token_count = len(tokens)
        # 截断
        if token_count > self.max_tokens:
            print(f"Prompt too long ({token_count}), truncating to {self.max_tokens} tokens.")
            tokens = tokens[:self.max_tokens]  # 保留最前部分
            history_text = self.tokenizer.decode(tokens)
  
        prompt = template.render(HISTORY=history_text.strip(), QUESTION=question.strip())
        
        # if True:
        #     num_tokens = len(self.tokenizer.encode(prompt, allowed_special='all'))
        #     # 保存日志
        #     log_entry = {
        #         # "question": question,
        #         # "history_text": history_text,
        #         "prompt_tokens": num_tokens
        #     }
        #     with open(log_file, "a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        #         return None,None
            
        retries = 0
        while retries < 3:
            try:
                t1 = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": (
                            "You are a helpful assistant that answers questions "
                            "based on the full dialogue history. Be concise and factual."
                        )},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                t2 = time.time()
                return response.choices[0].message.content.strip(), t2 - t1
            except Exception as e:
                retries += 1
                time.sleep(1)
                if retries >= 3:
                    raise e

    def clean_chat_history(self, chat_history):
        text = ""
        for c in chat_history:
            text += f"{c['timestamp']} | {c['speaker']}: {c['text']}\n"
        return text.strip()

    def generate_answer(self, idx, sample, dataset_name, output_file):
        sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")

        # ===== Parse conversation =====
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

        if not chat_history:
            print(f"⚠️ Sample {sample_id} has no valid conversation data, skipping.")
            return

        # ===== Convert chat history to plain text =====
        history_text = self.clean_chat_history(chat_history)

        # ===== Generate answers without RAG =====
        results = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            original_answer = qa.get("answer", "")
            system_answer, _ = self.generate_response_no_rag(question, history_text)

            results.append({
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": question,
                "system_answer": system_answer,
                "original_answer": original_answer,
                "timestamp": get_timestamp(),
                **({"category": qa.get("category")} if "category" in qa else {}),
                **({"question_type": qa.get("question_type")} if "question_type" in qa else {}),
            })

        # ===== Save results =====
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ Sample {sample_id} processed, saved in {output_file}")
        except Exception as e:
            print(f"⚠️ Error saving sample {sample_id}: {e}")

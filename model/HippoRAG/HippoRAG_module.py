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
# Question: 
{{QUESTION}}

# Context: 
{{CONTEXT}}

# Short answer:
"""

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

class simpleragModel:
    def __init__(self, chunk_size=500, top_k=1):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.model = "LLAMA"
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") #os.getenv("EMBEDDING_MODEL")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
        self.client_embedding = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="http://localhost:30099/v1") # embed model写死了，启动脚本在/home/shm/document/MEMORY_ACCLERATION/scripts/begin_embed_model.sh
    def generate_response(self, question, context):
        template = Template(PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)
        retries = 0
        while retries < 3:
            try:
                t1 = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": (
                             "You are a helpful assistant that answers questions "
                             "based on the provided context. Provide the shortest possible answer "
                             "and reuse words from the conversation.")},
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
        return text

    def calculate_embedding(self, text):
        response = self.client_embedding.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def calculate_similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def create_chunks(self, chat_history):
        encoding = tiktoken.encoding_for_model(self.embedding_model)
        documents = self.clean_chat_history(chat_history)
        if self.chunk_size == -1:
            return [documents], [self.calculate_embedding(documents)]

        tokens = encoding.encode(documents)
        chunks = []
        embeddings = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i+self.chunk_size]
            chunk = encoding.decode(chunk_tokens)
            chunks.append(chunk)
            embeddings.append(self.calculate_embedding(chunk))
        return chunks, embeddings

    def search_topk(self, query, chunks, embeddings):
        query_emb = self.calculate_embedding(query)
        sims = [self.calculate_similarity(query_emb, emb) for emb in embeddings]
        if self.top_k == 1:
            idxs = [int(np.argmax(sims))]
        else:
            idxs = np.argsort(sims)[-self.top_k:][::-1]
        context = "\n<->\n".join([chunks[i] for i in idxs])
        return context

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

        if not chat_history:
            print(f"⚠️ Sample {sample_id} has no valid conversation data, skipping.")
            return

        # ===== Step 2: Create chunks =====
        chunks, embeddings = self.create_chunks(chat_history)

        # ===== Step 3: Generate answers =====
        results = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            original_answer = qa.get("answer", "")

            context = chunks[0] if self.chunk_size == -1 else self.search_topk(question, chunks, embeddings)

            system_answer, _ = self.generate_response(question, context)

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

        # ===== Step 4: Save results =====
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ Sample {sample_id} processed, saved in {output_file}")
        except Exception as e:
            print(f"⚠️ Error saving sample {sample_id}: {e}")

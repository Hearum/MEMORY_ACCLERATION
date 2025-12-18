import json
import os
import re
import time
from jinja2 import Template
from openai import OpenAI
import numpy as np

PROMPT = """
You are an expert assistant designed for retrieval-augmented question answering.
Your only knowledge source is the retrieved context provided below.

====================  
## User Question  
{{QUESTION}}  
====================  

## Retrieved Context  
The following text consists of one or multiple conversation sessions.  
Each session may contain timestamps, speaker turns, summaries, or keywords.  
This context is the **only** evidence you may rely on.

{{CONTEXT}}

====================  
## Your Task  
Provide the **shortest accurate answer** to the user question **strictly based on the retrieved context**.  
Follow all instructions below:

### 1. Faithfulness and grounding  
- You must only use information explicitly stated in the retrieved context.  
- Do not invent facts or rely on outside knowledge.  

### 2. Answer length  
- The answer must be **as short as possible**, typically **one concise sentence** or a short phrase.  
- No explanations, no reasoning steps, no politeness.

### 3. Use context wording  
- Reuse exact phrases, facts, or wording from the context when possible.  
- Do not paraphrase if the source wording is concise.

### 4. No extra content  
- Do not mention the sessions, timestamps, or speakers unless needed for correctness.  
- Do not output analysis, bullet points, justification, or commentary.

### 5. Ambiguity handling  
- If multiple sessions give conflicting information, choose the most direct and relevant evidence.  
- If the answer depends on multiple pieces of context, combine them briefly.

====================  
## Final Answer  
Write only the final short answer:
"""

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

class simpleragModel:
    def __init__(self, top_k=10):
        self.top_k = top_k
        self.model = \
            "LLAMA"
        self.embedding_model = \
            os.getenv("EMBEDDING_MODEL", "text-embedding-3-small") #os.getenv("EMBEDDING_MODEL")
        self.client = \
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
        self.client_embedding = \
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="http://localhost:8000/v1")
    
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
        if len(text.strip()) >= 8192:
            text = text[:8192]  # Truncate to first 8192 characters
        response = self.client_embedding.embeddings.create(model="/mnt/data3/models/Qwen3-Embedding-4B", input=text)
        return response.data[0].embedding

    def calculate_similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def search_topk(self, query, embeddings):
        query_emb = self.calculate_embedding(query)
        sims = [self.calculate_similarity(query_emb, emb) for emb in embeddings]
        if self.top_k == 1:
            idxs = [int(np.argmax(sims))]
        else:
            idxs = np.argsort(sims)[-self.top_k:][::-1].tolist()
        return idxs

    def generate_answer(self, idx, sample, dataset_name, output_file):

        sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")

        if "29f2956b" in sample_id:
            print(f"Skipping sample {sample_id} as it does not meet the criteria.")
            return

        # ===== Step 1: Parse conversation =====
        chat_history = []
        turn2session = []
        sessions = []
        qa_pairs = []
        speaker_a = None
        speaker_b = None

        # locomo10 dataset
        if dataset_name == "locomo10":
            conversation = sample.get("conversation", {})
            
            for key, chats in conversation.items():
                match = re.match(r"session_(\d+)$", key)
                if match:
                    session_id = int(match.group(1)) - 1
                    timestamp = conversation.get(f"session_{session_id}_date_time")
                    sessions.append(f"timestamp: {timestamp}\n")

                    for c in chats:
                        chat_history.append({
                            "speaker": c["speaker"],
                            "text": c["text"],
                            "timestamp": timestamp,
                        })
                        turn2session.append(session_id)
                        sessions[session_id] += f"{c['speaker']}: {c['text']}\n"

            qa_pairs = sample.get("qa", [])
            speaker_a = conversation.get("speaker_a", "User")
            speaker_b = conversation.get("speaker_b", "Assistant")

        # longmemeval dataset(lme_m, lme_s, lme_oracle)
        elif dataset_name.startswith("longmemeval"):
            haystack_sessions = sample.get("haystack_sessions")
            haystack_dates = sample.get("haystack_dates")
            haystack_session_ids = sample.get("haystack_session_ids")
            assert(len(haystack_dates)   ==len(haystack_session_ids))
            assert(len(haystack_sessions)==len(haystack_session_ids))
            
            for i in range(len(haystack_session_ids)):
                session_id = i
                timestamp = haystack_dates[i]
                sessions.append(f"timestamp: {timestamp}\n")
                
                for turn in haystack_sessions[i]:
                    role = turn.get("role")
                    content = turn.get("content").strip()
                    if not content:
                        continue  # 跳过空白回合
                    chat_history.append({
                        "speaker": "User" if role.lower() == "user" else "Assistant",
                        "text": content,
                        "timestamp": timestamp
                    })
                    turn2session.append(session_id)
                    sessions[session_id] += f"{chat_history[-1]['speaker']}: {content}\n"

            qa_pairs = [{
                "question":      sample.get("question"),
                "answer":        sample.get("answer"),
                "question_id":   sample.get("question_id"),
                "question_type": sample.get("question_type"),
                "question_date": sample.get("question_date"),
            }]

            speaker_a = "User"
            speaker_b = "Assistant"

        else:
            raise ValueError(f"Unsupported dataset type: {dataset_name}")

        if not chat_history:
            print(f"⚠️ Sample {sample_id} has no valid conversation data, skipping.")
            return

        # ===== Step 2: Create chunks =====
        turns = []
        pair2session = []
        for i in range(0, len(chat_history), 2):
            pair2session.append(turn2session[i])
            turn = chat_history[i]
            if (i + 1 < len(chat_history)):
                turn_next = chat_history[i + 1]
                if turn["timestamp"] == turn_next["timestamp"]:
                    turns.append(f"{turn['speaker']}: {turn['text']}\n{turn_next['speaker']}: {turn_next['text']}")
                    continue
            turns.append(f"{turn['speaker']}: {turn['text']}")
        embeddings = [self.calculate_embedding(turn) for turn in turns]

        # ===== Step 3: Generate answers =====
        results = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            original_answer = qa.get("answer", "")

            topk_idx = self.search_topk(question, embeddings)
            context_idxs = set(pair2session[i] for i in topk_idx)
            context = "\n<->\n".join(sessions[i] for i in context_idxs)
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

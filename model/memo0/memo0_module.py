

import argparse
import os

# from .src.langmem import LangMemManager
# from .src.memzero.add import MemoryADD
# from .src.memzero.search import MemorySearch
# from .src.openai.predict import OpenAIPredict
# from .src.rag import RAGManager
# from .src.utils import METHODS, TECHNIQUES
# from .src.zep.add import ZepAdd
# from .src.zep.search import ZepSearch
from openai import OpenAI
import openai
import time
import json
from jinja2 import Template
from mem0 import Memory
from mem0.configs.base import MemoryConfig
import os
# class OpenAIClient:
#     def __init__(self, api_key, base_url):
#         self.api_key = api_key
#         self.base_url = base_url
#         openai.api_key = self.api_key
#         openai.api_base = self.base_url

#     def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):

#         response = gpt_client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=temperature,
#             max_tokens=max_tokens
#         )
#         return response.choices[0].message.content.strip()


# client = OpenAIClient(
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     base_url=os.environ.get("OPENAI_API_BASE")
# )

# import os
# import json

# from .src.memzero.add import MemoryADD
# from .src.memzero.search import MemorySearch
def process_conversation(conversation_data):
    processed = []
    speaker_a = conversation_data.get("speaker_a", "User")
    speaker_b = conversation_data.get("speaker_b", "Assistant")

    session_keys = [k for k in conversation_data.keys() if k.startswith("session_") and not k.endswith("_date_time")]

    for session_key in session_keys:
        timestamp_key = f"{session_key}_date_time"
        timestamp = conversation_data.get(timestamp_key, "")
        for dialog in conversation_data[session_key]:
            speaker = dialog["speaker"]
            text = dialog["text"]
            if "blip_caption" in dialog and dialog["blip_caption"]:
                text = f"{text} (image description: {dialog['blip_caption']})"

            if speaker == speaker_a:
                processed.append({"user_input": text, "agent_response": "", "timestamp": timestamp})
            else:
                if processed:
                    processed[-1]["agent_response"] = text
                else:
                    processed.append({"user_input": "", "agent_response": text, "timestamp": timestamp})
    return processed

def process_longmemeval_sessions(haystack_sessions, haystack_dates):
    processed = []
    for idx, session in enumerate(haystack_sessions):
        timestamp = haystack_dates[idx] if haystack_dates else ""
        for turn in session:
            if turn["role"] == "user":
                processed.append({"user_input": turn["content"], "agent_response": "", "timestamp": timestamp})
            else:
                if processed:
                    processed[-1]["agent_response"] = turn["content"]
                else:
                    processed.append({"user_input": "", "agent_response": turn["content"], "timestamp": timestamp})
    return processed

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
ANSWER_PROMPT_GRAPH = """
    You are an intelligent memory assistant tasked with retrieving accurate information from 
    conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question. You also have 
    access to knowledge graph relations for each user, showing connections between entities, 
    concepts, and events relevant to that user.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the 
       memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", 
       etc.), calculate the actual date based on the memory timestamp. For example, if a 
       memory from 4 May 2022 mentions "went to India last year," then the trip occurred 
       in 2021.
    6. Always convert relative time references to specific dates, months, or years. For 
       example, convert "last year" to "2022" or "two months ago" to "March 2023" based 
       on the memory timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse 
       character names mentioned in memories with the actual users who created those 
       memories.
    8. The answer should be less than 5-6 words.
    9. Use the knowledge graph relations to understand the user's knowledge network and 
       identify important relationships between entities in the user's world.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the 
       question
    4. If the answer requires calculation (e.g., converting relative time references), 
       show your work
    5. Analyze the knowledge graph relations to understand the user's knowledge context
    6. Formulate a precise, concise answer based solely on the evidence in the memories
    7. Double-check that your answer directly addresses the question asked
    8. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Relations for user {{speaker_1_user_id}}:

    {{speaker_1_graph_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Relations for user {{speaker_2_user_id}}:

    {{speaker_2_graph_memories}}

    Question: {{question}}

    Answer:
    """
from datetime import datetime, timezone

def normalize_timestamp(ts):
    """
    Convert timestamp to UTC-aware datetime.
    ts: str or datetime
    """
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
        except:
            dt = datetime.utcnow()
    elif isinstance(ts, datetime):
        dt = ts
    else:
        dt = datetime.utcnow()
    # 如果是 naive，强制加 UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt

ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """
import pdb
ANSWER_PROMPT_ZEP = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories:

    {{memories}}

    Question: {{question}}
    Answer:
    """
class memo0Model:
    def __init__(self, top_k=30, filter_memories=False, is_graph=False):
        self.top_k = top_k
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        # EMBED_DIM = 384
        # /home/shm/anaconda3/envs/sglang/lib/python3.10/site-packages/mem0/embeddings/openai.py
        memory_config = MemoryConfig(
                embedder={
                    "provider": "openai",
                    "config": {
                        "model": "/mnt/data3/models/Qwen3-Embedding-4B",   
                        "api_key": "nope",               
                        "openai_base_url": "http://localhost:8000/v1"
                    }
                },
            vector_store={
                "provider": "faiss",
                "config": {
                    "path": "./memory_index",
                    "collection_name": "mem0",
                    "embedding_model_dims": 2560
                }
            },
            llm={
                "provider": "openai",  
                "config": {
                    "model": "LLAMA",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "openrouter_base_url": os.environ.get("OPENAI_API_BASE")
                }
            }
        )
        self.memory = Memory(memory_config)

        
    def chat_with_memories(self, message: str, user_id_a,user_id_b) -> str:

        res_a = self.memory.search(query=message, user_id=user_id_a,limit=100)
        # res_a = self.memory.search(query="hello", user_id=user_id_a,limit=100)
        res_b = self.memory.search(query=message, user_id=user_id_b,limit=100)
        results_a = res_a.get("results", [])
        results_b = res_b.get("results", [])
        def format_results(results):
            formatted = []
            for item in results:
                mem = item.get("memory", "")
                meta = item.get("metadata") or {}
                ts = meta.get("timestamp", "unknown_time")
                formatted.append(f"{ts}: {mem}")
            return formatted
        
        search_a_memory = format_results(results_a)
        search_b_memory = format_results(results_b)

        graph_memories_a = res_a.get("relations", [])
        graph_memories_b = res_b.get("relations", [])

        template = Template(ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=user_id_a.split("_")[0],
            speaker_2_user_id=user_id_b.split("_")[0],
            speaker_1_memories=json.dumps(search_a_memory, indent=4),
            speaker_2_memories=json.dumps(search_b_memory, indent=4),
            speaker_1_graph_memories=json.dumps(graph_memories_a, indent=4),
            speaker_2_graph_memories=json.dumps(graph_memories_b, indent=4),
            question=message,
        )

        response = self.memory.llm.generate_response([{"role": "system", "content": answer_prompt}])

        assistant_response = response["response"] if isinstance(response, dict) else str(response)

        return assistant_response
    
    def format_sample_for_memadd(self, sample, dataset_name, sample_id):

        if dataset_name == "locomo10":
            conversation_data = sample.get("conversation", {})
            processed_dialogs = process_conversation(conversation_data) if conversation_data else []
            qa_pairs = sample.get("qa", [])
            speaker_a = f"{conversation_data.get('speaker_a', 'User')}_{sample_id}"
            speaker_b = f"{conversation_data.get('speaker_b', 'Assistant')}_{sample_id}"

        elif dataset_name.startswith("longmemeval"):
            haystack_sessions = sample.get("haystack_sessions", [])
            haystack_dates = sample.get("haystack_dates", [])
            processed_dialogs = process_longmemeval_sessions(haystack_sessions, haystack_dates)
            qa_pairs = [{
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "question_id": sample.get("question_id", ""),
                "question_type": sample.get("question_type", ""),
                "question_date": sample.get("question_date", "")
            }]
            speaker_a = f"User_{sample_id}"
            speaker_b = f"Assistant_{sample_id}"

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        if not processed_dialogs and not qa_pairs:
            return None

        # user_id = f"{speaker_a}_{sample_id}"
        return {
            "processed_dialogs": processed_dialogs,
            "qa_pairs": qa_pairs,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            # "user_id": user_id,
            "sample_id": sample_id
        }

    def generate_answer(self, idx, sample, dataset_name, output_file):

        self.memory.reset()
        sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")
        print(f"=== Processing sample {idx} ({dataset_name}) ===")

        formatted = self.format_sample_for_memadd(sample, dataset_name,idx)
        
        if formatted is None:
            print(f"⚠️ Sample {sample_id} has no valid conversation data, skipping.")
            return
        # ===== Step 1: Add 历史对话到 Memory =====
        for turn in formatted["processed_dialogs"]:

            if turn.get("user_input"):
                self.memory.add([{"role": "user", "content": turn["user_input"]}], user_id=formatted["speaker_a"],agent_id=None,metadata={"timestamp": normalize_timestamp(turn.get("timestamp"))},infer=True)
                # self.memory.add([{"role": "user", "content": "The farmer needs to transport a fox, a chicken, and some grain across a river using a boat. The fox cannot be left alone with the chicken, and the chicken cannot be left alone with the grain. The boat can only hold one item at a time, and the river is too dangerous to cross multiple times. Can you help the farmer transport all three items across the river without any of them getting eaten? Remember, strategic thinking and planning are key to solving this puzzle. If you're stuck, try thinking about how you would solve the puzzle yourself, and use that as a starting point. Be careful not to leave the chicken alone with the fox, or the chicken and the grain alone together, as this will result in a failed solution. Good luck!"}], user_id=formatted["speaker_a"],agent_id=None,metadata={"timestamp": turn.get("timestamp", "")},infer=True)
            if turn.get("agent_response"):
                self.memory.add([{"role": "assistant", "content": turn["agent_response"]}],user_id=formatted["speaker_b"], metadata={"timestamp": normalize_timestamp(turn.get("timestamp"))},infer=True)
                # self.memory.add([{"role": "assistant", "content": 'To solve this puzzle, the farmer can follow these steps:\n\n1. First, the farmer should take the chicken across the river using the boat.\n2. Next, the farmer should go back to the original side of the river and take the fox across the river using the boat.\n3. Now, the farmer should go back to the original side of the river again and pick up the chicken using the boat.\n4. Finally, the farmer can take the grain across the river using the boat.\n\nThis solution ensures that at no point is the chicken left alone with the fox, or the chicken and the grain left alone together. The farmer can successfully transport all three items across the river using the boat.'}],user_id=formatted["speaker_b"], metadata={"timestamp": turn.get("timestamp", "")},infer=True)

        # ===== Step 2: 生成答案 =====
        results = []
        for qa in formatted["qa_pairs"]:
            question = qa.get("question", "")
            original_answer = qa.get("answer", "") or qa.get("adversarial_answer", "")
            try:
                system_answer = self.chat_with_memories(question,formatted["speaker_a"],formatted["speaker_b"])
            except Exception as e:
                print(f"❌ Error generating answer for {sample_id}, question: {question}, error: {e}")
                continue

            results.append({
                "sample_id": sample_id,
                "speaker_a": formatted["speaker_a"],
                "speaker_b": formatted["speaker_b"],
                "question": question,
                "system_answer": system_answer,
                "original_answer": original_answer,
                "timestamp": get_timestamp(),
                **({"category": qa.get("category")} if "category" in qa else {}),
                **({"question_type": qa.get("question_type")} if "question_type" in qa else {})
            })

        # ===== Step 3: 保存结果 =====
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ Sample {sample_id} processed successfully, saved in {output_file}")
        except Exception as e:
            print(f"⚠️ Failed to save results for {sample_id}: {e}")
    # def generate_answer1(self, idx, sample, dataset_name, output_file):

    #     sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")
    #     print(f"=== Processing sample {sample_id} ({dataset_name}) ===")

    #     # ===== Step 1: Parse dataset-specific conversation format =====
    #     if dataset_name == "locomo10":
    #         processed_dialogs = process_conversation(sample.get("conversation", []))
    #         qa_pairs = sample.get("qa", [])
    #         speaker_a = sample.get("conversation", {}).get("speaker_a", "User")
    #         speaker_b = sample.get("conversation", {}).get("speaker_b", "Assistant")

    #     # elif dataset_name.startswith("longmemeval"):
    #     #     processed_dialogs = process_longmemeval_sessions(
    #     #         sample.get("haystack_sessions", []),
    #     #         sample.get("haystack_dates", [])
    #     #     )
    #     #     qa_pairs = [{
    #     #         "question": sample.get("question", ""),
    #     #         "answer": sample.get("answer", ""),
    #     #         "question_id": sample.get("question_id", ""),
    #     #         "question_type": sample.get("question_type", ""),
    #     #         "question_date": sample.get("question_date", "")
    #     #     }]
    #     #     speaker_a = "User"
    #     #     speaker_b = "Assistant"

    #     # else:
    #     #     raise ValueError(f"Unsupported dataset type: {dataset_name}")

    #     if not processed_dialogs:
    #         print(f"⚠️ Sample {sample_id} has no valid conversation data, skipping.")
    #         return

    #     # ===== Step 2: Initialize memory =====
    #     print(f"Initializing memory for {sample_id} ...")
    #     memory_adder = MemoryADD(is_graph=self.is_graph)
    #     memory_adder.process_conversation(sample, idx)

 
    #     # ===== Step 3: Retrieve and answer questions =====
    #     print(f"Generating answer for {sample_id} ...")

    #     memory_searcher = MemorySearch(
    #         output_file_path=output_file,
    #         top_k=self.top_k,
    #         filter_memories=self.filter_memories,
    #         is_graph=self.is_graph
    #     )

    #     results = []
    #     for qa in qa_pairs:
    #         question = qa.get("question", "")
    #         gt_answer = qa.get("answer", "")
    #         try:
    #             system_answer = memory_searcher.query_single(
    #                 user_id=speaker_a_id,
    #                 question=question
    #             )
    #         except Exception as e:
    #             print(f"❌ Error generating answer for {sample_id}: {e}")
    #             continue

    #         results.append({
    #             "sample_id": sample_id,
    #             "question": question,
    #             "system_answer": system_answer,
    #             "ground_truth": gt_answer,
    #             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #         })

    #     # ===== Step 4: Save results =====
    #     try:
    #         with open(output_file, "a", encoding="utf-8") as f:
    #             for r in results:
    #                 f.write(json.dumps(r, ensure_ascii=False) + "\n")
    #         print(f"✅ Sample {sample_id} processed successfully and saved.")
    #     except Exception as e:
    #         print(f"⚠️ Failed to save results for {sample_id}: {e}")
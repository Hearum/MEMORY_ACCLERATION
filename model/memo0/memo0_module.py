

import argparse
import os

from .src.langmem import LangMemManager
from .src.memzero.add import MemoryADD
from .src.memzero.search import MemorySearch
from .src.openai.predict import OpenAIPredict
from .src.rag import RAGManager
from .src.utils import METHODS, TECHNIQUES
from .src.zep.add import ZepAdd
from .src.zep.search import ZepSearch
from openai import OpenAI
import openai
import time
import json
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
    """
    Process conversation data from locomo10 format into memory system format.
    Handles both text-only and image-containing messages.
    """
    processed = []
    speaker_a = conversation_data["speaker_a"]
    speaker_b = conversation_data["speaker_b"]
    
    # Find all session keys
    session_keys = [key for key in conversation_data.keys() if key.startswith("session_") and not key.endswith("_date_time")]
    
    for session_key in session_keys:
        timestamp_key = f"{session_key}_date_time"
        timestamp = conversation_data.get(timestamp_key, "")
        
        for dialog in conversation_data[session_key]:
            speaker = dialog["speaker"]
            text = dialog["text"]
            
            # Handle image content if present
            if "blip_caption" in dialog and dialog["blip_caption"]:
                text = f"{text} (image description: {dialog['blip_caption']})"
            
            # Alternate between speakers as user and assistant
            if speaker == speaker_a:
                processed.append({
                    "user_input": text,
                    "agent_response": "",
                    "timestamp": timestamp
                })
            else:
                if processed:
                    processed[-1]["agent_response"] = text
                else:
                    processed.append({
                        "user_input": "",
                        "agent_response": text,
                        "timestamp": timestamp
                    })
    
    return processed

def process_longmemeval_sessions(haystack_sessions, haystack_dates):
    """
    Process LongMemEval sessions into memory system format.
    Each session is a list of turns: {"role": "user"/"assistant", "content": "..."}
    """
    processed = []
    for session_idx, session in enumerate(haystack_sessions):
        timestamp = haystack_dates[session_idx] if haystack_dates else ""
        for turn in session:
            if turn["role"] == "user":
                processed.append({
                    "user_input": turn["content"],
                    "agent_response": "",
                    "timestamp": timestamp
                })
            else:
                if processed:
                    processed[-1]["agent_response"] = turn["content"]
                else:
                    processed.append({
                        "user_input": "",
                        "agent_response": turn["content"],
                        "timestamp": timestamp
                    })
    return processed

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

class memo0Model:
    def __init__(self, top_k=30, filter_memories=False, is_graph=False):
        self.top_k = top_k
        self.filter_memories = filter_memories
        self.is_graph = is_graph

    def format_sample_for_memadd(self,sample, dataset_name):
        """
        Convert a single sample into MemoryADD.process_conversation compatible format.
        """
        if dataset_name == "locomo10":
            conversation_data = sample.get("conversation", {})
            if not conversation_data:
                return None
            return {"conversation": conversation_data}

        elif dataset_name.startswith("longmemeval"):
            haystack_sessions = sample.get("haystack_sessions", [])
            haystack_dates = sample.get("haystack_dates", [])

            speaker_a = "User"
            speaker_b = "Assistant"

            conversation = {"speaker_a": speaker_a, "speaker_b": speaker_b}
            for idx, (session, date) in enumerate(zip(haystack_sessions, haystack_dates)):
                key = f"session_{idx}"
                conversation[key + "_date_time"] = date
                chats = []
                for turn in session:
                    if "user_input" in turn:
                        chats.append({"speaker": speaker_a, "text": turn["user_input"]})
                    if "agent_response" in turn:
                        chats.append({"speaker": speaker_b, "text": turn["agent_response"]})
                conversation[key] = chats

            return {"conversation": conversation}

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    def generate_answer(self, idx, sample, dataset_name, output_file):
        sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")
        print(f"=== Processing sample {sample_id} ({dataset_name}) ===")

        # ===== Step 1: Format sample for MemoryADD =====
        formatted_sample = self.format_sample_for_memadd(sample, dataset_name)
        if formatted_sample is None:
            print(f"⚠️ Sample {sample_id} has no valid conversation data, skipping.")
            return

        # # ===== Step 2: Initialize memory and add conversation =====
        # memory_adder = MemoryADD(is_graph=self.is_graph)
        # memory_adder.process_conversation(formatted_sample, idx)

        # ===== Step 3: Prepare QA pairs =====
        if dataset_name == "locomo10":
            qa_pairs = sample.get("qa", [])
            conversation = sample.get("conversation", {})
            speaker_a = conversation.get("speaker_a", "User")
            speaker_b = conversation.get("speaker_b", "Assistant")
        elif dataset_name.startswith("longmemeval"):
            qa_pairs = [{
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "question_id": sample.get("question_id", ""),
                "question_type": sample.get("question_type", ""),
                "question_date": sample.get("question_date", "")
            }]
            speaker_a = "User"
            speaker_b = "Assistant"

        speaker_a_user_id = f"{speaker_a}_{sample_id}"
        speaker_b_user_id = f"{speaker_b}_{sample_id}"

        # ===== Step 4: Initialize MemorySearch =====
        memory_searcher = MemorySearch(
            output_path=output_file,
            top_k=self.top_k,
            filter_memories=self.filter_memories,
            is_graph=self.is_graph
        )

        results = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            original_answer = qa.get("answer", "") or qa.get("adversarial_answer", "")

            try:
                system_answer, _, _, _, _, _, _, _ = memory_searcher.answer_question(
                    speaker_a_user_id, speaker_b_user_id, question, original_answer, qa.get("category", None)
                )
            except Exception as e:
                print(f"❌ Error generating answer for {sample_id}, question: {question}, error: {e}")
                continue
            # import pdb
            # pdb.set_trace()
            # ===== Step 5: Store in the desired format =====
            results.append({
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": question,
                "system_answer": system_answer,
                "original_answer": original_answer,
                "timestamp": get_timestamp(),
                **({"category": qa.get("category")} if "category" in qa else {}),
                **({"question_type": qa.get("question_type")} if "question_type" in qa else {})
            })

        # ===== Step 6: Save results =====
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ Sample {sample_id} processed successfully, saved in {output_file}")
        except Exception as e:
            print(f"⚠️ Failed to save results for {sample_id}: {e}")


    def generate_answer1(self, idx, sample, dataset_name, output_file):

        sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")
        print(f"=== Processing sample {sample_id} ({dataset_name}) ===")

        # ===== Step 1: Parse dataset-specific conversation format =====
        if dataset_name == "locomo10":
            processed_dialogs = process_conversation(sample.get("conversation", []))
            qa_pairs = sample.get("qa", [])
            speaker_a = sample.get("conversation", {}).get("speaker_a", "User")
            speaker_b = sample.get("conversation", {}).get("speaker_b", "Assistant")

        # elif dataset_name.startswith("longmemeval"):
        #     processed_dialogs = process_longmemeval_sessions(
        #         sample.get("haystack_sessions", []),
        #         sample.get("haystack_dates", [])
        #     )
        #     qa_pairs = [{
        #         "question": sample.get("question", ""),
        #         "answer": sample.get("answer", ""),
        #         "question_id": sample.get("question_id", ""),
        #         "question_type": sample.get("question_type", ""),
        #         "question_date": sample.get("question_date", "")
        #     }]
        #     speaker_a = "User"
        #     speaker_b = "Assistant"

        # else:
        #     raise ValueError(f"Unsupported dataset type: {dataset_name}")

        if not processed_dialogs:
            print(f"⚠️ Sample {sample_id} has no valid conversation data, skipping.")
            return

        # ===== Step 2: Initialize memory =====
        print(f"Initializing memory for {sample_id} ...")
        memory_adder = MemoryADD(is_graph=self.is_graph)
        memory_adder.process_conversation(sample, idx)

        # 将样本的对话历史构造成 memory 格式
        # 每个 sample 独立 user_id，防止记忆污染
        # speaker_a_id = f"{speaker_a}_{sample_id}"
        # speaker_b_id = f"{speaker_b}_{sample_id}"

        # 删除旧记忆
        # memory_adder.mem0_client.delete_all(user_id=speaker_a_id)
        # memory_adder.mem0_client.delete_all(user_id=speaker_b_id)

        # 加入历史对话到记忆中
        # import pdb
        # pdb.set_trace()
        # memory_adder.add_memories_for_speaker(
        #     speaker_a_id,
        #     [{"role": "user", "content": f"{speaker_a}: {d['text']}"} for d in processed_dialogs],
        #     timestamp=time.time(),
        #     desc=f"Adding memory for {speaker_a}"
        # )

        # memory_adder.add_memories_for_speaker(
        #     speaker_b_id,
        #     [{"role": "assistant", "content": f"{speaker_b}: {d['text']}"} for d in processed_dialogs],
        #     timestamp=time.time(),
        #     desc=f"Adding memory for {speaker_b}"
        # )
        # memory_adder.process_conversation
        # speaker_a_msgs = []
        # for d in processed_dialogs:
        #     if "user_input" in d:
        #         speaker_a_msgs.append({"role": "user", "content": f"{speaker_a}: {d['user_input']}"})

        # speaker_b_msgs = []
        # for d in processed_dialogs:
        #     if "agent_response" in d:
        #         speaker_b_msgs.append({"role": "assistant", "content": f"{speaker_b}: {d['agent_response']}"})

        # memory_adder.add_memories_for_speaker(
        #     speaker_a_id,
        #     speaker_a_msgs,
        #     timestamp=time.time(),
        #     desc=f"Adding memory for {speaker_a}"
        # )

        # memory_adder.add_memories_for_speaker(
        #     speaker_b_id,
        #     speaker_b_msgs,
        #     timestamp=time.time(),
        #     desc=f"Adding memory for {speaker_b}"
        # )

        # ===== Step 3: Retrieve and answer questions =====
        print(f"Generating answer for {sample_id} ...")

        memory_searcher = MemorySearch(
            output_file_path=output_file,
            top_k=self.top_k,
            filter_memories=self.filter_memories,
            is_graph=self.is_graph
        )

        results = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            gt_answer = qa.get("answer", "")
            try:
                system_answer = memory_searcher.query_single(
                    user_id=speaker_a_id,
                    question=question
                )
            except Exception as e:
                print(f"❌ Error generating answer for {sample_id}: {e}")
                continue

            results.append({
                "sample_id": sample_id,
                "question": question,
                "system_answer": system_answer,
                "ground_truth": gt_answer,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            })

        # ===== Step 4: Save results =====
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"✅ Sample {sample_id} processed successfully and saved.")
        except Exception as e:
            print(f"⚠️ Failed to save results for {sample_id}: {e}")
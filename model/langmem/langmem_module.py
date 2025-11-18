

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
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict

from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store
from langmem import create_manage_memory_tool, create_search_memory_tool
# from prompts import ANSWER_PROMPT
from tqdm import tqdm

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
ANSWER_PROMPT_TEMPLATE = Template(ANSWER_PROMPT)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),base_url=os.environ.get("OPENAI_API_BASE"))

def prompt(state):
    """Prepare the messages for the LLM."""
    store = get_store()
    memories = store.search(
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful assistant.

    ## Memories
    <memories>
    {memories}
    </memories>
    """
    return [{"role": "system", "content": system_msg}, *state["messages"]]
from sentence_transformers import SentenceTransformer
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# if os.path.exists(LOCAL_MODEL_PATH):
#     model_path = LOCAL_MODEL_PATH
#     embedding_model = SentenceTransformer(model_path, local_files_only=True)
# else:
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# from langchain.embeddings.huggingface_embeddings import HuggingFaceEmbeddings

# embed_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={"device": "cuda"}  # 或 "cpu"
# )
st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")  # 或 "cpu"


class LocalEmbeddings:
    def __init__(self, model):
        self.model = model

    def __call__(self, texts):
        if isinstance(texts, str):
            return self.model.encode([texts], convert_to_numpy=True)[0].tolist()
        else:
            return self.model.encode(list(texts), convert_to_numpy=True).tolist()

    def embed_documents(self, texts):
        return self.__call__(texts)

    def embed_query(self, text):
        return self.__call__(text)

local_embed = LocalEmbeddings(st_model)

from langchain_openai import ChatOpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),base_url=os.environ.get("OPENAI_API_BASE"))
class LangMem:
    def __init__(
        self,
    ):
        self.store = InMemoryStore(index={
            "dims": 384,
            "embed": local_embed,
        })
        self.checkpointer = MemorySaver()  # Checkpoint graph state
        local_chat = ChatOpenAI(
            model_name="LLAMA",
            openai_api_base=os.environ.get("OPENAI_API_BASE"),  
            openai_api_key=os.environ.get("OPENAI_API_KEY"), 
            temperature=0
        )
        self.agent = create_react_agent(
            local_chat,
            prompt=prompt,
            tools=[
                create_manage_memory_tool(namespace=("memories",)),
                create_search_memory_tool(namespace=("memories",)),
            ],
            store=self.store,
            checkpointer=self.checkpointer,
        )

    def add_memory(self, message, config):
        return self.agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)

    def search_memory(self, query, config):
        try:
            t1 = time.time()
            response = self.agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
            t2 = time.time()
            return response["messages"][-1].content, t2 - t1
        except Exception as e:
            print(f"Error in search_memory: {e}")
            return "", t2 - t1
        
class langmemModel:

    def __init__(self, top_k=30, filter_memories=False, is_graph=False):
        self.top_k = top_k
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.agent_a = LangMem()  # speaker A
        self.agent_b = LangMem()  # speaker B

    def chat_with_memories(self, question, speaker_a_id, speaker_b_id,
                           memories_a, memories_b):
        prompt = ANSWER_PROMPT_TEMPLATE.render(
            question=question,
            speaker_1_user_id=speaker_a_id,
            speaker_1_memories=memories_a,
            speaker_2_user_id=speaker_b_id,
            speaker_2_memories=memories_b,
        )

        t1 = time.time()
        response = client.chat.completions.create(
            model="LLAMA",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0
        )
        t2 = time.time()

        return response.choices[0].message.content, t2 - t1


    def format_sample_for_memadd(self, sample, dataset_name, sample_id):
        if dataset_name == "locomo10":
            conversation = sample.get("conversation", {})
            processed = process_conversation(conversation)
            qa_pairs = sample.get("qa", [])

            speaker_a = f"{conversation.get('speaker_a', 'User')}_{sample_id}"
            speaker_b = f"{conversation.get('speaker_b', 'Assistant')}_{sample_id}"

        elif dataset_name.startswith("longmemeval"):
            sessions = sample.get("haystack_sessions", [])
            dates = sample.get("haystack_dates", [])
            processed = process_longmemeval_sessions(sessions, dates)

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

        if not processed and not qa_pairs:
            return None

        return {
            "processed_dialogs": processed,
            "qa_pairs": qa_pairs,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "sample_id": sample_id,
        }


    def generate_answer(self, idx, sample, dataset_name, output_file):

        # 重置agent
        self.agent_a = LangMem()
        self.agent_b = LangMem()

        sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")
        print(f"=== Processing sample {idx} ({dataset_name}) ===")

        formatted = self.format_sample_for_memadd(sample, dataset_name, idx)
        if formatted is None:
            print(f"⚠️ Sample {sample_id} has no valid conversation data, skipping.")
            return

        speaker_a = formatted["speaker_a"]
        speaker_b = formatted["speaker_b"]

        dialogs = formatted["processed_dialogs"]
        qa_pairs = formatted["qa_pairs"]


        print("Adding conversation memories...")
        for turn in dialogs:
            timestamp = turn.get("timestamp", "")

            if turn.get("user_input"):
                # user_input -> speaker A
                self.agent_a.add_memory(
                    message=f"{timestamp} | {speaker_a}: {turn['user_input']}",
                    config={"configurable": {"thread_id": f"{speaker_a}-{idx}"}}
                )

            if turn.get("agent_response"):
                # agent_response -> speaker B
                self.agent_b.add_memory(
                    message=f"{timestamp} | {speaker_b}: {turn['agent_response']}",
                    config={"configurable": {"thread_id": f"{speaker_b}-{idx}"}}
                )
        print("Processing QA pairs...")
        results = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            original_answer = qa.get("answer", "")

            memories_a, ta = self.agent_a.search_memory(
                query=question,
                config={"configurable": {"thread_id": f"{speaker_a}-{idx}"}}
            )
            memories_b, tb = self.agent_b.search_memory(
                query=question,
                config={"configurable": {"thread_id": f"{speaker_b}-{idx}"}}
            )

            try:
                system_answer, t_final = self.chat_with_memories(
                    question,
                    speaker_a, speaker_b,
                    memories_a, memories_b
                )
            except Exception as e:
                print(f"❌ Error generating answer: {e}")
                continue

            results.append({
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": question,
                "system_answer": system_answer,
                "original_answer": original_answer,
                "speaker_a_memory": memories_a,
                "speaker_b_memory": memories_b,
                "search_time_a": ta,
                "search_time_b": tb,
                "final_time": t_final,
                "timestamp": get_timestamp(),
            })

        try:
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            # print(f"✅ Sample {sample_id} processed successfully.")
            print(f"✅ sample {sample_id} success, result save in {output_file}")
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")

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
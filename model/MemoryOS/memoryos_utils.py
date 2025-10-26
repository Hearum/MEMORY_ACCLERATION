import time
import uuid
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

gpt_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

import shutil
def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def generate_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

LOCAL_MODEL_PATH = os.path.expanduser(":~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
# try:
# print(f"[DEBUG] Loading embedding model: {model_name}")
if os.path.exists(LOCAL_MODEL_PATH):
    model_path = LOCAL_MODEL_PATH
    embedding_model = SentenceTransformer(model_path, local_files_only=True)
else:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text, model_name="all-MiniLM-L6-v2"):
    # LOCAL_MODEL_PATH = os.path.expanduser(":~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
    # # try:
    # # print(f"[DEBUG] Loading embedding model: {model_name}")
    # if os.path.exists(LOCAL_MODEL_PATH):
    #     model_path = LOCAL_MODEL_PATH
    #     model = SentenceTransformer(model_path, local_files_only=True)
    # else:
    #     model = SentenceTransformer(model_name)
    # except Exception as e:
    #     print(f"[WARN] Failed to load model {model_name}: {e}")
    #     # 删除本地缓存重试
    #     cache_dir = os.path.join(os.path.expanduser("~/.cache/huggingface/transformers"), model_name)
    #     if os.path.exists(cache_dir):
    #         print(f"[INFO] Removing corrupted cache: {cache_dir}")
    #         shutil.rmtree(cache_dir)
    #     print("[INFO] Retrying download...")
    #     model = SentenceTransformer(model_name)
    if text is None:
        text = ""
    embedding = embedding_model.encode([text], convert_to_numpy=True)[0]
    return embedding

def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

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
import os, json
from datetime import datetime
from threading import Lock
import openai
import tiktoken

class OpenAIClient:
    def __init__(self, api_key, base_url, model="gpt-4o-mini", log_path="/home/shm/document/log/prompt_cal_log_m.json"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.log_path = log_path
        self.lock = Lock()  # 确保多线程写文件安全

        # 初始化文件
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total": 0,  # 所有LLM输入token总数
                    "total_query": 0,  
                    "reuse_continuity": 0,  # 连续对话复用
                    "reuse_meta_summary": 0,  # 摘要生成阶段
                    "reuse_meta_update": 0,  # Meta融合阶段
                    "reuse_profile_merge": 0,  # 用户画像合并
                    "reuse_analysis": 0,  # 各类分析模块
                    "key_word": 0, 
                    "theme":0,
                    "personality_analysis":0,
                    "final_question": 0,
                    "last_update": None
                }, f, indent=2)

    def count_tokens(self, messages):
        total = 0
        for msg in messages:
            tokens = self.tokenizer.encode(msg["content"])
            total += len(tokens)
        return total

    def _update_log(self, tag, total_tokens):
        with self.lock:
            with open(self.log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["total"] += total_tokens
            data["total_query"] += 1

            if tag == "continuity":
                data["reuse_continuity"] += total_tokens
            elif tag == "meta_summary":
                data["reuse_meta_summary"] += total_tokens
            elif tag == "meta_update":
                data["reuse_meta_update"] += total_tokens
            elif tag == "profile_merge":
                data["reuse_profile_merge"] += total_tokens

            elif tag in ("analysis", "user_analysis", "assistant_analysis"):
                data["reuse_analysis"] += total_tokens-258

            elif tag == "key_word":
                    data["key_word"] += total_tokens-53

            elif tag == "theme":
                    data["theme"] += total_tokens

            elif tag == "personality_analysis":
                    data["personality_analysis"] += total_tokens

            elif tag == "final_question":
                    data["final_question"] += total_tokens-344
        elif tag == "final_question":  # 确保这里有处理 final_question
            data["final_question"] += total_tokens
            data["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)


    def _log_request_content(self, messages):
        with self.lock:
            log_path="/home/shm/document/log/log_m_all_query.json"
            with open(log_path, "a", encoding="utf-8") as f:
                entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": []
                }
                for msg in messages:
                    text = msg["content"]
                    length = len(self.tokenizer.encode(text))
                    entry["messages"].append({"text": text, "length": length})
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
    def chat_completion(self, model=None, messages=None, temperature=0.7, max_tokens=2000, tag=None):
        model = model or self.model
        total_tokens = self.count_tokens(messages)
        self._update_log(tag, total_tokens)
        self._log_request_content(messages) 

        response = gpt_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
        # return response.choices[0].message.content.strip()

    def get_reuse_report(self):
        with open(self.log_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = data["total"]
        t1 = data["reuse_continuity"]
        t2 = data["reuse_meta_summary"]
        t3 = data["reuse_meta_update"]
        report = {
            "Total tokens": total,
            "Continuity reuse": f"{t1} ({t1/total:.2%})" if total > 0 else "N/A",
            "Meta summary reuse": f"{t2} ({t2/total:.2%})" if total > 0 else "N/A",
            "Meta update reuse": f"{t3} ({t3/total:.2%})" if total > 0 else "N/A",
            "Overall reusable": f"{(t1+t2+t3)/total:.2%}" if total > 0 else "N/A",
            "Last update": data.get("last_update", "N/A")
        }
        total = data["total"]
        if total == 0:
            return {"message": "No LLM calls recorded yet."}

        def fmt(x):
            return f"{x} ({x/total:.2%})"

        report = {
            "Total tokens": total,
            "Continuity reuse": fmt(data["reuse_continuity"]),
            "Meta summary reuse": fmt(data["reuse_meta_summary"]),
            "Meta update reuse": fmt(data["reuse_meta_update"]),
            "Profile merge reuse": fmt(data["reuse_profile_merge"]),
            "Analysis reuse": fmt(data["reuse_analysis"]),
            # "Overall reusable (potential)": f"{(data['reuse_continuity'] + data['reuse_meta_summary'] + data['reuse_meta_update'] + data['reuse_profile_merge'] + data['reuse_analysis']) / total:.2%}",
            "Last update": data["last_update"]
        }
        return report
    
def gpt_generate_answer(prompt, messages, client,tag=None):
    return client.chat_completion(model="Qwen", messages=messages, temperature=0.7, max_tokens=2000,tag=tag)

def analyze_assistant_knowledge(dialogs, client):
    """
    Analyzes conversations to extract knowledge or identity traits about the assistant.
    Returns: {"assistant_knowledge": str}
    """
    conversation = "\n".join([f"User: {d['user_input']}\nAI: {d['agent_response']}\nTime:{d['timestamp']}\n" for d in dialogs])

    prompt = """
# Assistant Knowledge Extraction Task
Analyze the conversation and extract any fact or identity traits about the assistant. 
If no traits can be extracted, reply with "None". Use the following format for output:
The generated content should be as concise as possible — the more concise, the better.
【Assistant Knowledge】
- [Fact 1]
- [Fact 2]
- (Or "None" if none found)

Few-shot examples:
1. User: Can you recommend some movies.
   AI: Yes, I recommend Interstellar.
   Time: 2023-10-01
   【Assistant Knowledge】
   - I recommend Interstellar on 2023-10-01.

2. User: Can you help me with cooking recipes?
   AI: Yes, I have extensive knowledge of cooking recipes and techniques.
   Time: 2023-10-02
   【Assistant Knowledge】
   - I have cooking recipes and techniques on 2023-10-02.

3. User: That’s interesting. I didn’t know you could do that.
   AI: I’m glad you find it interesting!
   【Assistant Knowledge】
   - None

Conversation:
""" + conversation

    messages = [
        {
            "role": "system",
            "content": """You are an assistant knowledge extraction engine. Rules:
1. Extract ONLY explicit statements about the assistant's identity or knowledge.
2. Use concise and factual statements in the first person.
3. If no relevant information is found, output "None".""" 
        },
        {"role": "user", "content": prompt}
    ]

    print("Analyzing assistant knowledge...")
    result = gpt_generate_answer(prompt, messages, client,tag="analysis")
    
    # Parse output
    assistant_knowledge = result.replace("【Assistant Knowledge】", "").strip()
    return {"assistant_knowledge": assistant_knowledge}

def gpt_summarize(dialogs, client):
    prompt = "Please generate a topic summary based on the following conversation：\n"
    for d in dialogs:
        prompt += f"user: {d.get('user_input','')}\nassiant: {d.get('agent_response','')}\n"
    prompt += "\nSubject Summary："
    messages = [
        {"role": "system", "content": "You are an expert in summarizing dialogue topics, please generate a concise and precise summary."},
        {"role": "user", "content": prompt}
    ]
    print("调用 GPT 生成主题摘要...")
    return gpt_generate_answer(prompt, messages, client,tag="meta_summary")

def gpt_generate_multi_summary(text, client):
    """
    调用 LLM 生成多子主题摘要，返回格式示例如下：
    {
      "input": "对话文本",
      "summaries": [
         {"theme": "出差", "keywords": ["出差", "行程", "工作"], "content": "用户提到出差相关的困扰"},
         {"theme": "健康", "keywords": ["感冒", "难受", "生病"], "content": "用户反馈感冒导致身体不适"}
      ]
    }
    """
    prompt = ("Please analyze the following dialogue and generate multiple subtopic summaries (if applicable), with a maximum of two themes.\n"
              "Each summary should include the subtopic name, keywords (separated by commas), and the summary text, formatted as a JSON array, with an example format as follows:\n"
              "[\n  {\"theme\": \"Business trip\", \"keywords\": [\"Business trip\", \"Itinerary\", \"Work\"], \"content\": \" User mentioned the troubles related to business trips.\"},\n  {\"theme\": \"Health\", \"keywords\": [\"Cold\", \"Uncomfortable\", \"Sick\"], \"content\": \"User reported feeling unwell due to a cold.\"}\n]\n"
              "Please directly output the JSON array, without adding any other content.\n\Conversation content:\n" + text)
    messages = [
        {"role": "system", "content": "You are an expert in analyzing dialogue topics. No more than two topics."},
        {"role": "user", "content": prompt}
    ]
    print("调用 GPT 生成多子主题摘要...")
    response_text = gpt_generate_answer(prompt, messages, client,tag="meta_summary")
    import json
    try:
        summaries = json.loads(response_text)
    except Exception:
        summaries = []
    return {"input": text, "summaries": summaries}

# def gpt_personality_analysis(dialogs, client):
#     prompt = ("Please analyze the following conversation and extract the user profile information and user private data."
#               "Please output in the following format:\n"
#               "【User Profile】\n"
#               "Areas of Interest:\n"
#               "Response Preferences：\n"
#               "Preferred Content Type：\n"
#               "Short vs. Detailed Responses：\n"
#               "Formal vs. Casual Tone：\n"
#               "Other Notes:：\n"
#               "【User Private Data】\n"
#               "Please list all the private information involved (such as account numbers, passwords, user purchase,etc.). If there is none, please write \"None\"\n\n"
#               "The conversation is as follows:\n")
#     for d in dialogs:
#         prompt += f"User: {d.get('user_input','')}\nAssiant: {d.get('agent_response','')}\n"
#     messages = [
#         {"role": "system", "content": "You are a professional user profile analyst who can also identify user private data. Please strictly follow the template for output."},
#         {"role": "user", "content": prompt}
#     ]
#     print("调用 GPT 分析用户画像和私有数据...")
#     result_text = gpt_generate_answer(prompt, messages, client)
#     profile, private = "", ""
#     parts = result_text.split("【User Private Data】")
#     if len(parts) == 2:
#         profile = parts[0].replace("【User Profile】", "").strip()
#         private = parts[1].strip()
#     else:
#         profile = result_text.strip()
#         private = "None"
#     return {"profile": profile, "private": private}
# def gpt_personality_analysis(dialogs, client):
#     """
#     Analyzes conversations to extract structured personality traits, private knowledge, 
#     and assistant-related knowledge.
#     Returns: {"profile": str, "private": str, "assistant_knowledge": str}
#     """
#     conversation = "\n".join([f"User: {d['user_input']}\nAssistant: {d['agent_response']}" for d in dialogs])

#     prompt = """
# # Personality Analysis Task
# Analyze the conversation and output in EXACTLY this format:

# 【User Profile】
# 1. Core Psychological Traits:
#    - [Trait]: [Positive/Negative/Neutral] (Evidence)
#    - (Max 5 most prominent traits)

# 2. Content Preferences:
#    - [Topic]: [Like/Dislike/Neutral] (Evidence)
#    - (Max 5 strongest preferences)

# 3. Interaction Style:
#    - [Style]: [Preference] (Evidence)
#    - (e.g., Direct/Indirect, Detailed/Concise)

# 4. Value Alignment:
#    - [Value]: [Strong/Weak] (Evidence)
#    - (e.g., Honesty, Helpfulness)

# 【User Private Data】
# - [Fact 1]
# - [Fact 2]
# - (Or "None" if none found)

# Conversation:
# """ + conversation

#     messages = [
#         {
#             "role": "system",
#             "content": """You are a personality analysis engine. Rules:
# 1. Extract ONLY observable traits with direct evidence
# 2. Use standardized trait names from psychology
# 3. Mark confidence: Positive=explicit preference, Neutral=implied
# 4. Private data includes possessions, habits, and sensitive preferences"""
#         },
#         {"role": "user", "content": prompt}
#     ]

#     print("Running personality analysis...")
#     result = gpt_generate_answer(prompt, messages, client)
    
#     # Parse output
#     profile, private = result.split("【User Private Data】") if "【User Private Data】" in result else (result, "None")
    
#     # Analyze assistant knowledge
#     assistant_knowledge_result = analyze_assistant_knowledge(dialogs, client)
    
#     return {
#         "profile": profile.replace("【User Profile】", "").strip(),
#         "private": private.strip(),
#         "assistant_knowledge": assistant_knowledge_result["assistant_knowledge"]
#     }
def gpt_personality_analysis(dialogs, client):
    """
    Analyzes conversations to extract structured personality traits, general user data, 
    and assistant-related knowledge.
    Returns: {"profile": str, "user_data": str, "assistant_knowledge": str}
    """
    conversation = "\n".join([f"User: {d['user_input']}\nAssistant: {d['agent_response']}\nTime:{d['timestamp']}" for d in dialogs])

    prompt = """
    # Personality and User Data Analysis Task
    Analyze the conversation and output in EXACTLY this format:

    【User Profile】
    1. Core Psychological Traits:
    - [Trait]: [Positive/Negative/Neutral] (Evidence)
    - (Max 5 most prominent traits)

    2. Content Preferences:
    - [Topic]: [Like/Dislike/Neutral] (Evidence)
    - (Max 5 strongest preferences)

    3. Interaction Style:
    - [Style]: [Preference] (Evidence)
    - (e.g., Direct/Indirect, Detailed/Concise)

    4. Value Alignment:
    - [Value]: [Strong/Weak] (Evidence)
    - (e.g., Honesty, Helpfulness)

    【User Data】
    - [Fact 1]: [Details] (e.g., "User mentioned visiting a park on April 1st, 2025 in New York.")
    - [Fact 2]: [Details] (e.g., "User likes pizza, enjoys sci-fi movies, and dislikes rainy weather.")
    - (Include events, dates, locations, preferences, or other general or private information explicitly mentioned in the conversation. If none, write "None.")

    Conversation:
    """ + conversation

    messages = [
            {
                "role": "system",
                "content": """You are a personality and user data analysis engine. Rules:
    1. Extract ONLY observable traits and data with direct evidence.
    2. Include general user data such as events, dates, locations, and preferences.
    3. Use concise and factual statements.
    4. If no relevant information is found, output "None"."""
            },
            {"role": "user", "content": prompt}
        ]

    print("Running personality and user data analysis...")
    result = gpt_generate_answer(prompt, messages, client,tag="personality_analysis")
    
    # Parse output
    profile, user_data = result.split("【User Data】") if "【User Data】" in result else (result, "None")
    
    # Analyze assistant knowledge
    assistant_knowledge_result = analyze_assistant_knowledge(dialogs, client)
    
    return {
        "profile": profile.replace("【User Profile】", "").strip(),
        "private": user_data.strip(),
        "assistant_knowledge": assistant_knowledge_result["assistant_knowledge"]
    }

def gpt_update_profile(old_profile, new_analysis, client):
    """
    Dynamically merges old and new profile data
    Args:
        old_profile: Previous profile text (structured)
        new_analysis: New analysis text (same format)
    Returns:
        Merged profile text with conflict resolution
    """
    prompt = f"""
# Profile Merge Task
Consolidate these profiles while:
- Preserving all valid observations
- Resolving conflicts
- Adding new dimensions

## Current Profile
{old_profile}

## New Data
{new_analysis}

## Rules
1. Keep ALL verified traits from both
2. Resolve conflicts by:
   a) New explicit evidence > old assumptions
   b) Mark as Neutral if contradictory
3. Add new dimensions from new data
4. Maintain EXACT original format

Output ONLY the merged profile (no commentary):
The generated content should not exceed 1500 words
"""

    messages = [
        {
            "role": "system",
            "content": """You are a profile integration system. Your rules:
1. NEVER discard verified information
2. Conflict resolution hierarchy:
   Explicit statement > Implied trait > Assumption
3. Add timestamps when traits change:
   (Updated: [date]) for modified traits
4. Preserve the 4-category structure"""
        },
        {"role": "user", "content": prompt}
    ]

    print("Updating user profile dynamically...")
    return gpt_generate_answer(prompt, messages, client,tag="profile_merge")

def gpt_extract_theme(answer_text, client):
    prompt = f"请从以下回答中提取主题总结，并以【主题提取】：开头输出：\n{answer_text}\n"
    messages = [
        {"role": "system", "content": "You are an expert in extracting conversation topics."},
        {"role": "user", "content": prompt}
    ]
    print("调用 GPT 提取主题总结...")
    return gpt_generate_answer(prompt, messages, client,tag="theme")

def llm_extract_keywords(text, client):
    prompt = "Please extract the keywords of the conversation topic from the following dialogue, separated by commas, and do not exceed three:\n" + text
    messages = [
        {"role": "system", "content": "You are a keyword extraction expert. Please extract the keywords of the conversation topic."},
        {"role": "user", "content": prompt}
    ]
    print("调用 GPT 提取关键词...")
    keywords_text =gpt_generate_answer(prompt, messages, client,tag="key_word")
    keywords = [w.strip() for w in keywords_text.split(",") if w.strip()]
    return set(keywords)

def compute_time_decay(session_timestamp, current_timestamp, tau=3600):
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    t1 = datetime.strptime(session_timestamp, fmt)
    t2 = datetime.strptime(current_timestamp, fmt)
    delta = (t2 - t1).total_seconds()
    return np.exp(-delta/tau)

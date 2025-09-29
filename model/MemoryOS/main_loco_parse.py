import json
from datetime import datetime, timedelta
from short_term_memory import ShortTermMemory
from mid_term_memory import MidTermMemory
from long_term_memory import LongTermMemory # 
from dynamic_update import DynamicUpdate # 负责从短期记忆中淘汰并更新到中期记忆。
from retrieval_and_answer import RetrievalAndAnswer # 实现记忆检索与回答生成。
from utils import OpenAIClient, gpt_generate_answer, gpt_extract_theme, gpt_update_profile, gpt_generate_multi_summary, get_timestamp, llm_extract_keywords, gpt_personality_analysis
import re
import openai
import time
import tiktoken
import os
total_tokens = 0
num_samples=0
# Initialize OpenAI client

client = OpenAIClient(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

# Heat threshold
H_THRESHOLD = 5.0
'''
如果其“热度”超过阈值，就触发：
- 调用 gpt_personality_analysis 对未分析的对话段落进行分析。
- 得到新的 profile用户画像、private facts私有信息、assistant knowledge。
- 如果已存在旧画像，则调用 gpt_update_profile 融合更新。
- 将私有事实逐条存入长期记忆；助手知识也加入。
- 重置该 segment 的热度指标，避免重复分析。
'''

def update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client):
    """
    Update user profile if heat exceeds threshold and extract assistant knowledge.
    """
    if not mid_mem.heap:
        return
    
    neg_heat, sid = mid_mem.heap[0]
    mid_mem.rebuild_heap()
    current_heat = -neg_heat
    
    if current_heat >= H_THRESHOLD:
        session = mid_mem.sessions.get(sid)
        if not session:
            return
        
        un_analyzed = [p for p in session["details"] if not p.get("analyzed", False)]
        if un_analyzed:
            print(f"Updating user profile: Segment {sid} heat {current_heat:.2f} exceeds threshold, starting profile update...")
            
            old_profile = long_mem.get_raw_user_profile(sample_id)
            
            result = gpt_personality_analysis(un_analyzed, client)
            new_profile = result["profile"]
            new_private = result["private"]
            assistant_knowledge = result["assistant_knowledge"]
            
            if old_profile:
                updated_profile = gpt_update_profile(old_profile, new_profile, client)
            else:
                updated_profile = new_profile
                
            long_mem.update_user_profile(sample_id, updated_profile)
            
            # 修改点：拆分 new_private 并逐个存储
            if new_private and new_private != "- None":
                # 按行拆分，过滤空行和非事实行（如 "【User Data】" 或注释）
                facts = [line.strip() for line in new_private.split("\n")]
                for fact in facts:
                    long_mem.add_knowledge(fact)  # 逐条添加
            
            if assistant_knowledge and assistant_knowledge != "None":
                long_mem.add_assistant_knowledge(assistant_knowledge)
            
            for p in session["details"]:
                p["analyzed"] = True
            session["N_visit"] = 0
            session["L_interaction"] = 0
            session["R_recency"] = 1.0
            session["H_segment"] = 0.0
            session["last_visit_time"] = get_timestamp()
            mid_mem.rebuild_heap()
            mid_mem.save()
            print(f"Update complete: Segment {sid} heat has been reset.")

# 输入：用户问题、短/长记忆、检索结果、知识、说话人信息。
'''
step:
1.从短期记忆构造 近期对话文本。
2.从检索结果构造 历史记忆片段。
3.从长期记忆取 用户画像、助手知识。
4.替换掉 user 和 assistant 的字样，改为具体 speaker_a / speaker_b。
5.组织成 system_prompt角色设定+回答规则和 user_prompt具体上下文+问题。
6.调用 GPT API (原文这里用的是Qwen,我们改成ds) 生成答案。
'''
def generate_system_response_with_meta(query, short_mem, long_mem, retrieval_queue, long_konwledge, client, sample_id, speaker_a, speaker_b, meta_data):
    """
    Generate system response with speaker roles clearly defined.
    """
    history = short_mem.get_all()
    history_text = "\n".join([
        f"{speaker_a}: {qa.get('user_input', '')}\n{speaker_b}: {qa.get('agent_response', '')}\nTime: ({qa.get('timestamp', '')})" 
        for qa in history
    ])
    
    retrieval_text = "\n".join([
        f"【Historical Memory】 {speaker_a}: {page.get('user_input', '')}\n{speaker_b}: {page.get('agent_response', '')}\nTime:({page.get('timestamp', '')})\nConversation chain overview:({page.get('meta_info', '')})\n" 
        for page in retrieval_queue
    ])
    
    profile_obj = long_mem.get_user_profile(sample_id)
    user_profile_text = str(profile_obj.get("data", "None")) if profile_obj else "None"
    
    background = f"【User Profile】\n{user_profile_text}\n\n"
    for kn in long_konwledge:
        background += f"{kn['knowledge']}\n"
    background = re.sub(r'(?i)\buser\b', speaker_a, background)
    background= re.sub(r'(?i)\bassistant\b', speaker_b, background)
    assistant_knowledge = long_mem.get_assistant_knowledge()
    assistant_knowledge_text = "【Assistant Knowledge】\n"
    for ak in assistant_knowledge:
        assistant_knowledge_text += f"- {ak['knowledge']} ({ak['timestamp']})\n"
    #meta_data_text = f"【Conversation Meta Data】\n{json.dumps(meta_data, ensure_ascii=False, indent=2)}\n\n"
    assistant_knowledge_text = re.sub(r'\bI\b', speaker_b, assistant_knowledge_text)
    
    system_prompt = (
        f"You are role-playing as {speaker_b} in a conversation with the user is playing is  {speaker_a}. "
        f"Here are some of your character traits and knowledge:\n{assistant_knowledge_text}\n"
        f"Any content referring to 'User' in the prompt refers to {speaker_a}'s content, and any content referring to 'AI'or 'assiant' refers to {speaker_b}'s content."
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"When the question is: \"What did the charity race raise awareness for?\", you should not answer in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: \"mental health\", as this is more concise."
    )
    
    user_prompt = (
        f"<CONTEXT>\n"
        f"Recent conversation between {speaker_a} and {speaker_b}:\n"
        f"{history_text}\n\n"
        f"<MEMORY>\n"
        f"Relevant past conversations:\n"
        f"{retrieval_text}\n\n"
        f"<CHARACTER TRAITS>\n"
        f"Characteristics of {speaker_a}:\n"
        f"{background}\n\n"
        f"the question is: {query}\n"
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"Please only provide the content of the answer, without including 'answer:'\n"
        f"For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.\n"
        f"If the question is about the duration, answer in the form of several years, months, or days.\n"
        f"Generate answers primarily composed of concrete entities, such as Mentoring program, school speech, etc"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat_completion(model="Qwen", messages=messages, temperature=0.7, max_tokens=2000)
    return response, system_prompt, user_prompt

# 对话预处理
# 输入locomo10 格式的对话数据。
'''
- 遍历每个 session,把对话内容转换为 统一格式[user_input、agent_response、timestamp]。
- 比较有意思的是这里支持图片内容, blip_caption 会作为文字描述拼接。
- 用户/助手说话轮次被组织成成对的 QA。
'''
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


def main_locomomo():

    print("begin for locomo10 dataset...")
    
    # 创建记忆文件存储目录
    os.makedirs("mem_tmp_loco_final", exist_ok=True)
    
    # Load locomo10 dataset
    try:
        with open("locomo10.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
    except FileNotFoundError:
        print("错误：找不到 locomo10.json 文件，请确保文件在当前目录中")
        return
    except Exception as e:
        print(f"加载数据集时出错：{e}")
        return
    
    # 处理整个数据集，不进行切片
    # dataset = dataset  # 处理全部数据
    
    # 设置固定的输出文件名
    output_file = "all_loco_results.json"
    
    results = []
    total_samples = len(dataset)
    
    for idx, sample in enumerate(dataset):
        print(f"正在处理样本 {idx + 1}/{total_samples}: {sample.get('sample_id', 'unknown')}")
        
        sample_id = sample.get("sample_id", "unknown_sample")
        conversation_data = sample["conversation"]
        qa_pairs = sample["qa"]
        
        # Process conversation data
        processed_dialogs = process_conversation(conversation_data)
        
        if not processed_dialogs:
            print(f"样本 {sample_id} 没有有效的对话数据，跳过")
            continue
            
        speaker_a = conversation_data["speaker_a"]
        speaker_b = conversation_data["speaker_b"]
        
        # Initialize memory modules
        short_mem = ShortTermMemory(max_capacity=1, file_path=f"mem_tmp_loco_final/{sample_id}_short_term.json")
        mid_mem = MidTermMemory(max_capacity=2000, file_path=f"mem_tmp_loco_final/{sample_id}_mid_term.json")
        long_mem = LongTermMemory(file_path=f"mem_tmp_loco_final/{sample_id}_long_term.json")
        dynamic_updater = DynamicUpdate(short_mem, mid_mem, long_mem, topic_similarity_threshold=0.6, client=client)
        retrieval_system = RetrievalAndAnswer(short_mem, mid_mem, long_mem, dynamic_updater, queue_capacity=10)
        
        # Store conversation history in memory system
        for dialog in processed_dialogs:
            short_mem.add_qa_pair(dialog)
            if short_mem.is_full():
                dynamic_updater.bulk_evict_and_update_mid_term()
            update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client)
        
        # Process QA pairs
        qa_count = len(qa_pairs)
        for qa_idx, qa in enumerate(qa_pairs):
            print(f"  处理问答 {qa_idx + 1}/{qa_count}")
            question = qa["question"]
            original_answer = qa.get("answer", "")
            category = qa["category"]
            evidence = qa.get("evidence", "")
            if(original_answer == ""):
                original_answer = qa.get("adversarial_answer", "")
            # Retrieve and generate answer
            retrieval_result = retrieval_system.retrieve(
                question, 
                segment_threshold=0.1, 
                page_threshold=0.1, 
                knowledge_threshold=0.1, 
                client=client
            )
            
            # Generate meta data for the conversation
            meta_data = {
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "category": category,
                "evidence": evidence
            }
            
            system_answer, system_prompt, user_prompt = generate_system_response_with_meta(
                question, 
                short_mem, 
                long_mem, 
                retrieval_result["retrieval_queue"], 
                retrieval_result["long_term_knowledge"],
                client, 
                sample_id, 
                speaker_a, 
                speaker_b, 
                meta_data
            )
            
            # Save result for the current QA pair
            results.append({
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": question,
                "system_answer": system_answer,
                "original_answer": original_answer,
                "category": category,
                "evidence": evidence,
                "timestamp": get_timestamp(),
            })
    
        # 每处理完一个样本就保存一次结果（实时保存）
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"样本 {idx + 1} 处理完成，结果已保存到 {output_file}")
        except Exception as e:
            print(f"保存结果时出错：{e}")
    
    # 最终保存
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"最终保存结果时出错：{e}")

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


def main_longmemeval(data_file: str, output_file: str = None):
    """
    Main function to run evaluation on LongMemEval dataset.
    Args:
        data_file: path to longmemeval JSON file
        output_file: path to save results (default: based on input file)
    """
    if output_file is None:
        output_file = f"results_{os.path.basename(data_file).replace('.json','')}.json"

    print(f"开始处理 LongMemEval 数据集: {data_file}")

    # 创建记忆文件存储目录
    dir_name = f"mem_tmp_{os.path.basename(data_file).replace('.json','')}"
    os.makedirs(dir_name, exist_ok=True)
    print(f"临时记忆目录: {dir_name}")

    # Load LongMemEval dataset
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
    except FileNotFoundError:
        print(f"错误：找不到 {data_file} 文件，请确保文件在当前目录中")
        return
    except Exception as e:
        print(f"加载数据集时出错：{e}")
        return

    results = []
    total_samples = len(dataset)

    for idx, sample in enumerate(dataset):
        sample_id = sample.get("question_id", f"sample_{idx+1}")
        print(f"正在处理样本 {idx + 1}/{total_samples}: {sample_id}")

        # Process conversation
        processed_dialogs = process_longmemeval_sessions(
            sample["haystack_sessions"],
            sample.get("haystack_dates", [])
        )

        if not processed_dialogs:
            print(f"样本 {sample_id} 没有有效的对话数据，跳过")
            continue

        # Construct QA pairs
        qa_pairs = [{
            "question": sample["question"],
            "answer": sample["answer"],
            "question_id": sample.get("question_id", ""),
            "question_type": sample.get("question_type", ""),
            "question_date": sample.get("question_date", "")
        }]

        # Speakers
        speaker_a = "User"
        speaker_b = "Assistant"

        # Initialize memory modules
        short_mem = ShortTermMemory(max_capacity=500, file_path=f"{dir_name}/{sample_id}_short_term.json")
        mid_mem = MidTermMemory(max_capacity=2000, file_path=f"{dir_name}/{sample_id}_mid_term.json")
        long_mem = LongTermMemory(file_path=f"{dir_name}/{sample_id}_long_term.json")
        dynamic_updater = DynamicUpdate(short_mem, mid_mem, long_mem, topic_similarity_threshold=0.6, client=client)
        retrieval_system = RetrievalAndAnswer(short_mem, mid_mem, long_mem, dynamic_updater, queue_capacity=10)

        # Store conversation history in memory system
        for dialog in processed_dialogs:
            short_mem.add_qa_pair(dialog)
            if short_mem.is_full():
                dynamic_updater.bulk_evict_and_update_mid_term()
            update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client)

        # Process QA pairs
        for qa_idx, qa in enumerate(qa_pairs):
            question = qa["question"]
            original_answer = qa.get("answer", "")
            if not original_answer:
                original_answer = qa.get("adversarial_answer", "")

            retrieval_result = retrieval_system.retrieve(
                question,
                segment_threshold=0.1,
                page_threshold=0.1,
                knowledge_threshold=0.1,
                client=client
            )

            meta_data = {
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question_type": qa.get("question_type", "")
            }

            system_answer, system_prompt, user_prompt = generate_system_response_with_meta(
                question,
                short_mem,
                long_mem,
                retrieval_result["retrieval_queue"],
                retrieval_result["long_term_knowledge"],
                client,
                sample_id,
                speaker_a,
                speaker_b,
                meta_data
            )

            results.append({
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": question,
                "system_answer": system_answer,
                "original_answer": original_answer,
                "question_type": qa.get("question_type", ""),
                "timestamp": get_timestamp()
            })

        # 每处理完一个样本就保存一次结果
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存结果时出错：{e}")

        print(f"样本 {idx + 1} 处理完成，结果已保存到 {output_file}")

    print(f"全部样本处理完成，最终结果保存在 {output_file}")

if __name__ == "__main__":
    # 测试logomomo
    # main_locomomo()

    # 测试 LongMemEval_S
    main_longmemeval("/home/shm/documents/longmemeval_s.json")
    
    # 测试 LongMemEval_M
    # main_longmemeval("/home/shm/documents/longmemeval_m.json")
    
    # 测试 LongMemEval_Oracle
    # main_longmemeval("longmemeval_oracle.json")

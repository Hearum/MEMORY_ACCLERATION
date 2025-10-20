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
import shutil
from utils import LocalLLM, OpenAILLM
from async_llm import run_async

import torch
import torch.nn.functional as F
import numpy as np
from src.construct.construct_emb import emb_rawdata
from src.construct.construct_asso import construct_asso
from eval_utils import evaluate_retrieval


PROMPT_G = """
You are an intelligent dialog bot. You will be shown History Dialogs. Please read, memorize, and understand the given Dialogs, then generate one concise, coherent and helpful response for the Question.

History Dialogs: {retrieved_texts}

Question Date: {question_date}
Question: {question}
"""

PROMPT_Multigran = """
You are an intelligent dialog bot. You will be shown History Dialogs and corresponding multi-granular information.
Filter the History Dialogs, summaries, and keywords to extract only the parts directly relevant to the Question. Preserve original tokens, do not paraphrase. Remove irrelevant turns, redundant info, and non-essential details.

History Dialogs: {retrieved_texts}

Question Date: {question_date}
Question: {question}
Answer:
"""

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

class MenGASModel:
    def __init__(self, dataset, retriever, method, num_seednodes=15, mem_threshold=30, n_components=2, damping=0.1, temp=0.1):
        self.dataset = dataset
        self.retriever = retriever
        self.method = method
        self.num_seednodes = num_seednodes
        self.mem_threshold = mem_threshold
        self.n_components = n_components
        self.damping = damping
        self.temp = temp

        # Load embeddings
        emb_path = f'../../data/process_embs/{dataset}-{retriever}-emb.pt'
        if os.path.exists(emb_path):
            self.all_emb = torch.load(emb_path)
        else:
            self.all_emb = emb_rawdata(dataset, retriever)

        # For memgas: construct graph
        if method == "memgas":
            graph_path = f"../../graph_cache/graph-{dataset}-{retriever}-{mem_threshold}-{n_components}.pt"
            if os.path.exists(graph_path):
                self.covid2graph = torch.load(graph_path)
            else:
                self.covid2graph = construct_asso(self)

    @staticmethod
    def run_ppr(g, reset_prob, damping):
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = g.personalized_pagerank(
            damping=damping,
            directed=False,
            reset=reset_prob,
            implementation='prpack'
        )
        return pagerank_scores

    @staticmethod
    def multi_granularity_routing(query_emb, granular_embeddings, temp):
        entropies = []
        for emb in granular_embeddings:
            similarity = (query_emb @ emb.T).squeeze()
            prob_dist = F.softmax(similarity / temp, dim=0)
            entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-12))
            entropies.append(entropy)
        entropies = torch.tensor(entropies)
        soft_router_weights = 1 - entropies
        soft_router_weights /= soft_router_weights.sum()
        return soft_router_weights

    def retrieve_for_sample(self, entry, emb):
        """
        对单个样本进行检索
        entry: 一个样本的原始数据 dict
        emb: 对应的 embedding dict
        返回：
            ranked_session_ids: list
        """
        results = []
        for qa_one, q_emb in zip(entry['qa'], emb['questions']):
            # Turn-level mean embeddings
            turn_num_each_session = [len(sess) for sess in entry['sessions']]
            turn_embeddings = []
            start_idx = 0
            for num_turns in turn_num_each_session:
                if num_turns == 0:
                    turn_mean_emb = torch.zeros(emb['turns'].size(1))
                else:
                    session_turn_embs = emb['turns'][start_idx:start_idx + num_turns]
                    turn_mean_emb = session_turn_embs.mean(dim=0)
                turn_embeddings.append(turn_mean_emb)
                start_idx += num_turns
            turn_embeddings = torch.stack(turn_embeddings)

            # Single granularity
            if self.method == 'session_level':
                scores = (q_emb @ emb['sessions'].T).squeeze()
            elif self.method == 'keyword_level':
                scores = (q_emb @ emb['keywords'].T).squeeze()
            elif self.method == 'summary_level':
                scores = (q_emb @ emb['summarys'].T).squeeze()
            elif self.method == 'hybrid_level':
                scores = (q_emb @ emb['hybrid'].T).squeeze()
            elif self.method == 'turn_level':
                scores = (q_emb @ turn_embeddings.T).squeeze()
            
            # Multi-granularity: memgas
            elif self.method == 'memgas':
                emb_list = [emb['sessions'], turn_embeddings, emb['summarys'], emb['keywords']]
                soft_router_weights = self.multi_granularity_routing(q_emb, emb_list, self.temp)
                emb_list = [w * v for w, v in zip(soft_router_weights, emb_list)]
                multi_gran_emb = []
                for i in range(emb['sessions'].size(0)):
                    for e in emb_list:
                        multi_gran_emb.append(e[i])
                multi_gran_emb = torch.stack(multi_gran_emb, dim=0)
                scores = (q_emb @ multi_gran_emb.T).squeeze()

                # topk threshold + PPR
                topk_values, _ = torch.topk(scores, self.num_seednodes)
                scores[scores < topk_values[-1]] = 0
                scores = self.run_ppr(self.covid2graph[entry['conversation_id']], scores, self.damping)
                # sum scores over 4 chunks
                scores = [sum(scores[i:i+4]) for i in range(0, len(scores), 4)]

            rankings = torch.tensor(scores).argsort(descending=True)
            ranked_session_ids = [entry['sessions_ids'][rid] for rid in rankings]
            results.append({
                "question": qa_one['question'],
                "ranked_session_ids": ranked_session_ids
            })
        return results


    def generate_answer(self, idx, sample, dataset_name, output_file, topk=3, llm_model=None, conv2summary=None, conv2keyword=None):
            """
            对单个样本生成答案，并保存结果
            """
            sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")

            # Step 1: 解析 conversation
            chat_history = []
            if dataset_name == "locomo10":
                conversation = sample.get("conversation", {})
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

            # Step 2: Retrieval
            # retrieve_for_sample 返回 [{'question':..., 'ranked_session_ids':[...]}]
            emb = next((e for e in self.all_emb if e['conversation_id'] == sample['conversation_id']), None)
            if emb is None:
                print(f"⚠️ No embedding found for sample {sample_id}")
                return

            retrieval_results = self.retrieve_for_sample(sample, emb)[0]  # 取第一个 QA 对应的结果
            top_sessions = retrieval_results['ranked_session_ids'][:topk]

            # Step 3: 构建 multigran prompt
            retrieved_texts = ""
            for sess_id in top_sessions:
                session_text = {**sample.get("sessions", {})}.get(sess_id, "")  # 单粒度内容
                summary_text = conv2summary[sample['conversation_id']][sess_id] if conv2summary else ""
                keyword_text = conv2keyword[sample['conversation_id']][sess_id] if conv2keyword else ""
                retrieved_texts += f"\n### Session ID: {sess_id}\nSession Content:\n{session_text}\n\nSession Summary:\n{summary_text}\nSession Keyword:\n{keyword_text}\n"

            # Step 4: 用 Multigran prompt 生成中间文本
            prompt_multigran = PROMPT_Multigran.format(
                retrieved_texts=retrieved_texts,
                question=qa_pairs[0]['question'],
                question_date=qa_pairs[0]['question_date']
            )
            async_responses = asyncio.run(run_async([prompt_multigran], llm_model=llm_model))
            filtered_text = async_responses[0]

            # Step 5: 用 G prompt 生成最终答案
            retrieved_texts_final = ""
            for sess_id in top_sessions:
                retrieved_texts_final += f"\n### Session ID: {sess_id}\nSession Content:\n{filtered_text}\n"
            prompt_final = PROMPT_G.format(
                retrieved_texts=retrieved_texts_final,
                question=qa_pairs[0]['question'],
                question_date=qa_pairs[0]['question_date']
            )
            final_response = asyncio.run(run_async([prompt_final], llm_model=llm_model))[0]

            # Step 6: 保存
            results = [{
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": qa_pairs[0]['question'],
                "system_answer": final_response,
                "original_answer": qa_pairs[0]['answer'],
                "timestamp": "unknown",  # 可加真实时间
                **({"category": qa_pairs[0].get("category")} if "category" in qa_pairs[0] else {}),
                **({"question_type": qa_pairs[0].get("question_type")} if "question_type" in qa_pairs[0] else {}),
            }]
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                f.write("\n")

            print(f"✅ Sample {sample_id} processed and saved to {output_file}")
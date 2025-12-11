# standard libraries
import asyncio
import os
import json
import time
from collections import defaultdict
# third-party libraries
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
# local modules
from .async_llm import run_async
from .construct_emb import emb_rawdata
from .construct_asso import construct_asso
from .dataprocess import dataprocess
from .multigran_generation import granularity_generate

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

class MemGASModel:
    def __init__(self):
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.num_seednodes = 15 # for ppr
        self.damping = 0.1 # for ppr
        self.temp = 0.1  # for multi-granularity routing
    
    def construct(self, dataset='longmemeval_s', retriever='contriever', method='memgas'):
        # Do data pre-process for dataset
        if dataset == 'locomo10':
            dataset_path = self.dir + f'/../../dataset/locomo10/locomo10.json'
        elif dataset.startswith('longmemeval'):
            dataset_path = self.dir + f'/../../dataset/longmemeval/{dataset}.json'
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        save_path = self.dir + f'/data/process_data/{dataset}.json'
        os.makedirs(self.dir + f'/data/', exist_ok=True)
        os.makedirs(self.dir + f'/data/process_data/', exist_ok=True)
        if not os.path.exists(save_path):
            print(f"[{get_timestamp()}] Start data processing for {dataset}...")
            dataprocess(dataset, dataset_path, save_path)
        # generate summary
        for level in ['summary_level', 'keyword_level']:
            if not os.path.exists(self.dir + f'/data/multi_granularity_logs/{dataset}-{level}.jsonl'):
                print(f"[{get_timestamp()}] Start multi-granularity generation for {dataset} {level}...")
                i_path = self.dir + f'/data/process_data/{dataset}.json'
                o_path = self.dir + f'/data/multi_granularity_logs/{dataset}-{level}.jsonl'
                os.makedirs(self.dir + f'/data/multi_granularity_logs/', exist_ok=True)
                granularity_generate(dataset, i_path, o_path, level)
        # Load embeddings
        os.makedirs(self.dir + f'/data/process_embs/', exist_ok=True)
        emb_path = self.dir + f'/data/process_embs/{dataset}-{retriever}-emb.pt'
        if os.path.exists(emb_path):
            self.all_emb = torch.load(emb_path)
        else:
            print(f"[{get_timestamp()}] Load embeddings for {dataset}...")
            self.all_emb = emb_rawdata(dataset, self.dir, retriever)
        # make sure method
        self.method = method
        # For memgas: construct graph
        if method == "memgas":
            os.makedirs(self.dir + "/data/graph_cache", exist_ok=True)
            graph_path = self.dir + f"/data/graph_cache/graph-{dataset}-{retriever}.pt"
            if os.path.exists(graph_path):
                self.covid2graph = torch.load(graph_path, weights_only=False)
            else:
                print(f"[{get_timestamp()}] Construct graph for {dataset} with method {method}...")
                self.covid2graph = construct_asso(dataset, self.dir, retriever)
        print(f"[{get_timestamp()}] Construction completed.")

    def run_ppr(self, g, reset_prob, damping):
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = g.personalized_pagerank(
            damping=damping,
            directed=False,
            reset=reset_prob,
            implementation='prpack'
        )
        return pagerank_scores

    def multi_granularity_routing(self, query_emb, granular_embeddings):
        entropies = []
        for emb in granular_embeddings:
            similarity = (query_emb @ emb.T).squeeze()
            prob_dist = F.softmax(similarity / self.temp, dim=0)
            entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-12))
            entropies.append(entropy)
        entropies = torch.tensor(entropies)
        soft_router_weights = 1 - entropies
        soft_router_weights /= soft_router_weights.sum()
        return soft_router_weights

    def retrieve_for_sample(self, entry, emb, topk=10):
        results = []
        for qa_one, q_emb in zip(entry['qa'], emb['questions']):
            # Turn embeddings
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

            # Score computation
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
            elif self.method == 'memgas':
                emb_list = [emb['sessions'], turn_embeddings, emb['summarys'], emb['keywords']]
                soft_router_weights = self.multi_granularity_routing(q_emb, emb_list)
                emb_list = [w * v for w, v in zip(soft_router_weights, emb_list)]
                multi_gran_emb = []
                for i in range(emb['sessions'].size(0)):
                    for e in emb_list:
                        multi_gran_emb.append(e[i])
                multi_gran_emb = torch.stack(multi_gran_emb, dim=0)
                scores = (q_emb @ multi_gran_emb.T).squeeze()

                # 过滤 top-N seeds
                topk_values, _ = torch.topk(scores, min(self.num_seednodes, len(scores)))
                scores[scores < topk_values[-1]] = 0
                scores = self.run_ppr(self.covid2graph[entry['conversation_id']], scores, self.damping)
                # 按 4 turns 聚合
                scores = [sum(scores[i:i+4]) for i in range(0, len(scores), 4)]
                scores = torch.tensor(scores)
            
            rankings = scores.argsort(descending=True)

            # 返回 topk session 的多粒度信息
            ranked_items = []
            for rid in rankings[:topk]:
                ranked_items.append({
                    'corpus_id': entry['sessions_ids'][rid],
                    'timestamp': entry['sessions_dates'][rid],
                    'session_text': entry['sessions'][rid],
                    'summary_text': emb['summarys'][rid] if 'summarys' in emb else "",
                    'keyword_text': emb['keywords'][rid] if 'keywords' in emb else ""
                })

            cur_result = {
                'conversation_id': entry['conversation_id'],
                'question_type': qa_one.get('question_type', ''),
                'question': qa_one['question'],
                'answer': qa_one['answer'],
                'question_date': qa_one.get('question_date', ''),
                'retrieval_results': {
                    'ranked_items': ranked_items,
                    'metrics': {}  # 可以根据需要在这里填 session/turn 的 recall
                }
            }
            results.append(cur_result)
        return results

    def generate_answer(self, idx, sample, dataset_name, output_file):
        """
        对单个样本生成答案，并保存结果
        """
        # ! information extraction
        # * idx
        sample_id = sample.get("sample_id") or sample.get("question_id", f"sample_{idx+1}")
        # * parse conversation, extract info for results
        if dataset_name == "locomo10":
            conversation = sample.get("conversation", {})
            speaker_a = conversation.get("speaker_a", "User")
            speaker_b = conversation.get("speaker_b", "Assistant")
        elif dataset_name.startswith("longmemeval"):
            speaker_a = "User"
            speaker_b = "Assistant"
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_name}")
        
        # ! construct data structure like memgas
        if dataset_name == "locomo10":
            newqa = []
            for qaitem in sample['qa']:
                if 'adversarial_answer' in qaitem:
                    answer = qaitem['adversarial_answer']
                else:
                    answer = qaitem['answer']
                answer_session_ids = []
                for item in qaitem['evidence']:
                    try:
                        turn_id = int(item.split(':')[1])
                    except:
                        continue
                    # answer_session_ids.append(f"{item.replace('D', 'session_').split(':')[0]}-turn_{turn_id  // 2 + 1}")
                    answer_session_ids.append(f"{item.replace('D', 'session_').split(':')[0]}")

                newqa.append(
                    {
                    "question": qaitem['question'],
                    "question_type": qaitem['category'],
                    "question_date": None,
                    "answer": answer,
                    "answer_session_ids":answer_session_ids,
                })
            conversation = sample['conversation']
            sessions_ids = []
            sessions_dates = []
            sessions = []
            for i in range(1000):
                if f'session_{i+1}' in conversation:
                    sessions_ids.append(f'session_{i+1}')
                    sessions_dates.append(conversation[f'session_{i+1}_date_time'])
                    session = []
                    for dialog in conversation[f'session_{i+1}']:
                        # 替换 speaker -> role 和 text -> content
                        if 'blip_caption' in dialog:
                            session.append(f"[{dialog['speaker']}]: {dialog['text']}\n The image Caption: {dialog['blip_caption']}")
                        else:
                            session.append(f"[{dialog['speaker']}]: {dialog['text']}")
                    merged_session = []
                    for i in range(0, len(session), 2):
                        if i + 1 < len(session):
                            merged_session.append(session[i] + "\n" + session[i+1])
                        else:
                            merged_session.append(session[i])

                    sessions.append(merged_session)
            entry = {
                'conversation_id':sample['sample_id'],
                'qa':newqa,
                'sessions_ids':sessions_ids,
                'sessions_dates':sessions_dates,
                'sessions':sessions
                }
        elif dataset_name.startswith("longmemeval"):
            answer_session_ids = []
            for cur_sess_id, sess_entry, ts in zip(sample['haystack_session_ids'], sample['haystack_sessions'], sample['haystack_dates']):
                for turn_id, turn in enumerate(sess_entry):
                    if 'has_answer' in turn and turn['has_answer']==True:
                        # answer_session_ids.append(f"{cur_sess_id.replace('answer_','')}-turn_{turn_id // 2 + 1}")
                        answer_session_ids.append(f"{cur_sess_id.replace('answer_','')}")

            sessions = []
            for sess_entry in sample['haystack_sessions']:
                # new_session = [{k: v for k, v in item.items() if k != "has_answer"} for item in sess_entry]
                session = []
                for item in sess_entry:
                    session.append(f"[{item['role']}]: {item['content']}")
                merged_session = []
                for i in range(0, len(session), 2):
                    if i + 1 < len(session):
                        merged_session.append(session[i] + "\n" + session[i+1])
                    else:
                        merged_session.append(session[i])
                if len(merged_session)==0:
                    print(len(merged_session),len(session))
                sessions.append(merged_session)
                
            entry = {
                'conversation_id':sample['question_id'],
                'qa':[
                    {
                    "question": sample['question'],
                    "question_type": sample['question_type'],
                    "question_date":sample['question_date'],
                    "answer": sample['answer'],
                    "answer_session_ids": answer_session_ids,
                },
                ],
                'sessions_ids':[s.replace('answer_', '') for s in sample['haystack_session_ids']],
                'sessions_dates':sample['haystack_dates'],
                'sessions':sessions
                }
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_name}")

        # ! retrieval for sample questions
        self.construct(dataset=dataset_name)
        emb = None
        for e in self.all_emb:
            if e['conversation_id'] == entry['conversation_id']:
                emb = e
                break
        if emb is None:
            raise ValueError(f"No embedding found for conversation_id: {entry['conversation_id']}")
        retrieval_results = self.retrieve_for_sample(entry=entry, emb=emb)
        assert len(retrieval_results) == len(entry['qa']), "Retrieval results length mismatch with number of questions."

        # ! generate answer for each question
        # * ids2sessions
        ids2session = {k:v for k,v in zip(entry['sessions_ids'],entry['sessions'])}
        # * ids2summary & ids2keyword
        def get_multigran(level):
            summ_dict = {}
            multi_gran_path = self.dir + f'/data/multi_granularity_logs/{dataset_name}-{level}.jsonl'
            with open(multi_gran_path, "r") as f:
                for line in f.readlines():
                    summ_dict.update(json.loads(line.strip()))
            ids2summary = {}
            conv_id = entry['conversation_id']
            for sessid in entry['sessions_ids']:
                summ_text = summ_dict[f'convid-{str(conv_id)}-sessid-{sessid}']
                ids2summary[sessid] = summ_text
            return ids2summary
        ids2summary = get_multigran('summary_level')
        ids2keyword = get_multigran('keyword_level')

        # llm extract useful info
        async_prompts = []
        for idx, question_entry in enumerate(retrieval_results):
            retrieved_texts = ""
            for retrieved_sess in question_entry['retrieval_results']['ranked_items']:
                sess_id = retrieved_sess['corpus_id']
                session_text = ids2session[retrieved_sess['corpus_id']]
                summary_text = ids2summary[retrieved_sess['corpus_id']]
                keyword_text = ids2keyword[retrieved_sess['corpus_id']]
                retrieved_texts += f"\n### Session ID: {sess_id}\nSession Content:\n{session_text}\n\nSession Summary:\n{summary_text}\nSession Keyword:\n{keyword_text}\n"
            prompt_multigran = PROMPT_Multigran.format(
                retrieved_texts=retrieved_texts,
                question=question_entry['question'],
                question_date=question_entry['question_date']
            )
            async_prompts.append(prompt_multigran)
        async_responses = asyncio.run(run_async(async_prompts, model="zhipu-glm-4-9b-chat-1m"))

        # use extracted info to generate final answer
        async_prompts = []
        for idx, question in enumerate(retrieval_results):
            retrieved_texts = ""
            for retrieved_sess in question_entry['retrieval_results']['ranked_items']:
                session = ids2session[retrieved_sess['corpus_id']]
                retrieved_texts += f"\n### Session ID: {sess_id}\nSession Content:\n{async_responses[idx]}\n"
            prompt_g = PROMPT_G.format(
                retrieved_texts=retrieved_texts,
                question=question['question'],
                question_date=question['question_date']
            )
            async_prompts.append(prompt_g)
        async_responses = asyncio.run(run_async(async_prompts, model="zhipu-glm-4-9b-chat-1m"))

        # ! record results
        results = []
        for final_response, question_entry in zip(async_responses, retrieval_results):
            results.append({
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": question_entry['question'],
                "system_answer": final_response,
                "original_answer": question_entry['answer'],
                "timestamp": None, # 可加真实时间
                **({"category": question_entry.get("question_type")} if dataset_name == "locomo10" else {}),
                **({"question_type": question_entry.get("question_type")} if dataset_name.startswith("longmemeval") else {}),
            })
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"✅ Sample {sample_id} processed and saved to {output_file}")
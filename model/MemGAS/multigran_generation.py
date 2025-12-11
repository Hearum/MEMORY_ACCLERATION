import os
import json
from tqdm import tqdm
from openai import OpenAI
import openai
import backoff
import tiktoken

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

@backoff.on_exception(backoff.constant, (openai.RateLimitError), 
                      interval=5)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def summarize_session(entry, model_name, instru_prompt):
    prompt = f"{instru_prompt}\n\n{entry}\n\nYour answer:"
    kwargs = {
        'model': model_name,
        'messages':[
            {"role": "user", "content": prompt}
        ],
        'n': 1,
        'temperature': 0,
        'max_tokens': 500
    }
    completion = chat_completions_with_backoff(client,**kwargs) 
    return completion.choices[0].message.content.strip()


def granularity_generate(dataset, dataset_path, generate_path, level):
    model_name = "zhipu-glm-4-9b-chat-1m"
    in_data = json.load(open(dataset_path))
    
    ids2session_text = {}
    for sample in in_data:
        conv_id = sample["conversation_id"]
        for sessid, sess in zip(sample['sessions_ids'], sample['sessions']):
            ids2session_text[f'convid-{str(conv_id)}-sessid-{sessid}'] = '\n\n'.join(sess)
    
    if dataset == 'locomo10':
        if level == 'summary_level':
            instru_prompt = "Below is an user-user dialogue memory. Please summarize the following dialogue as concisely as possible in a short paragraph, extracting the main themes and key information.\n"
        elif level == 'keyword_level':
            instru_prompt = "Below is an user-user dialogue memory. Please extract the most relevant keywords, separated by semicolon. Make sure no duplicated keywords\n"
    else:
        if level == 'summary_level':
            instru_prompt = "Below is an user-AI assistant dialogue memory. Please summarize the following dialogue as concisely as possible in a short paragraph, extracting the main themes and key information.\n"
        elif level == 'keyword_level':
            instru_prompt = "Below is an user-AI assistant dialogue memory. Please extract the most relevant keywords, separated by semicolon. Make sure no duplicated keywords\n"


    results = []
    generated_ids = set()
    if os.path.exists(generate_path):
        with open(generate_path, 'r', encoding='utf-8') as file:
            for line in file:
                sample = json.loads(line.strip())
                results.append(sample)
                generated_ids.update(sample.keys())
    
    for ids, entry in ids2session_text.items():
        if ids in generated_ids:
            continue
        expansion = summarize_session(entry, model_name, instru_prompt)
        print(ids, expansion)
        results.append({ids:expansion})
    
    with open(generate_path, 'w',encoding='utf-8') as f_write:
        f_write.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in results])

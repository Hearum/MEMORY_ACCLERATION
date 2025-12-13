import os
import sys
import json
from tqdm import tqdm
import backoff
import openai
from openai import OpenAI
import numpy as np


model_zoo = {
    'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
    'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
}


@backoff.on_exception(backoff.expo, (openai.RateLimitError,
                                    openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)



if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python evaluate_qa.py metric_model hyp_file ref_file')
        exit()

    metric_model_short = sys.argv[1]
    hyp_file = sys.argv[2]
    ref_file = sys.argv[3]
    verbose = True
    
    result_file = hyp_file + '.eval-results-{}'.format(metric_model_short)

    if metric_model_short not in model_zoo:
        print('Requested metric model is not supported:', metric_model_short)
        exit()
    metric_model, metric_model_source = model_zoo[metric_model_short]
    if metric_model_source == 'openai':
        openai.organization = os.getenv('OPENAI_ORGANIZATION')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openai_api_base = None
    else:
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8001/v1"
    
    metric_client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    try:
        hypotheses = [json.loads(line) for line in open(hyp_file).readlines()]
    except:
        hypotheses = json.load(open(hyp_file))
    try:
        references = json.load(open(ref_file))
    except:
        references = [json.loads(line) for line in open(ref_file).readlines()]
    qid2qdata = {entry['question_id']: entry for entry in references}
    qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
    qtypes = set(list(qid2qtype.values()))
    qtype2acc = {t: [] for t in qtypes}

    with open(result_file, 'w') as out_f:
        logs = []
        for entry in tqdm(hypotheses):

            if entry['question_id'] not in qid2qtype:
                print('Warning: skipping {} as it is not in reference data.'.format(entry['question_id']))
                continue
            
            qtype = qid2qtype[entry['question_id']]
            q = qid2qdata[entry['question_id']]['question']
            ans = qid2qdata[entry['question_id']]['answer']
            hyp = entry['hypothesis']
            
            prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention='_abs' in entry['question_id'])
            kwargs = {
                'model': metric_model,
                'messages':[
                    {"role": "user", "content": prompt}
                ],
                'n': 1,
                'temperature': 0,
                'max_tokens': 10
            }
            completion = chat_completions_with_backoff(metric_client, **kwargs)
            eval_response = completion.choices[0].message.content.strip()
            label = 'yes' in eval_response.lower()
            entry['autoeval_label'] = {
                'model': metric_model,
                'label': label
            }
            logs.append(entry)
            if verbose:
                print(json.dumps({
                    'question': q,
                    'answer': ans,
                    'hypothesis': hyp,
                    'autoeval_label': label
                }, indent=4), flush=True)
            print(json.dumps(entry), file=out_f)
            qtype2acc[qid2qtype[entry['question_id']]].append(1 if label else 0)

            
    print('Accuracy:', round(np.mean([1 if x['autoeval_label']['label'] else 0 for x in logs]).item(), 4))
    for k,v in qtype2acc.items():
        print('\t{}: {} ({})'.format(k, round(np.mean(v), 4), len(v)))

    print('Saved to', result_file)

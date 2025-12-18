import json

input_path = "/home/shm/document/MEMORY_ACCLERATION/results/Qwen3-32B-1m-GGUF_HippoRAG_longmemeval_s_mem/bf_HippoRAG_longmemeval_s_generation_results.jsonl"
output_path = "/home/shm/document/MEMORY_ACCLERATION/results/Qwen3-32B-1m-GGUF_HippoRAG_longmemeval_s_mem/HippoRAG_longmemeval_s_generation_results.jsonl"

decoder = json.JSONDecoder()

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

pos = 0
length = len(text)

all_dicts = []

while pos < length:

    try:
        obj, end = decoder.raw_decode(text, pos)
        pos = end

        # 如果是单个 dict，加入
        if isinstance(obj, dict):
            all_dicts.append(obj)

        # 如果是 list，则把其中的 dict 全部加入
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    all_dicts.append(item)

    except json.JSONDecodeError:
        pos += 1

# 删除 memoryA / memoryB
for item in all_dicts:
    item.pop("retrieval_result", None)
    # item.pop("memoryB", None)

with open(output_path, "w", encoding="utf-8") as f:
    for item in all_dicts:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("成功提取 dict 数量：", len(all_dicts))
print("结果已写入：", output_path)
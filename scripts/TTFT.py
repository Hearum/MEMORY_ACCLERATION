import json
import subprocess
import statistics
import re
from pathlib import Path

# ===================== 配置 =====================
# 保存的 query 日志文件路径
QUERY_LOG = Path("/home/shm/document/log/log_m_all_query_bf.json")
# 输出 TTFT 结果文件
OUTPUT_LOG = Path("/home/shm/document/log/full_reuse_ttft_results.json")

# evalscope 模板命令，后续会用 query 长度替换 {len}
EVALSCOPE_CMD = """evalscope perf \
  --parallel 1 \
  --model LLAMA \
  --url http://localhost:30045/v1/chat/completions \
  --api openai \
  --api-key "nope" \
  --dataset random \
  --min-tokens 128 \
  --max-tokens 128 \
  --stream \
  --prefix-length {len} \
  --min-prompt-length {len} \
  --max-prompt-length {len} \
  --number 10 \
  --tokenizer-path /mnt/data/models/Llama-3.2-3B-Instruct \
  --extra-args '{{"max_tokens": 128}}'
"""


def load_query_lengths():
    lengths = []
    with open(QUERY_LOG, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ 第 {line_num} 行解析失败: {e}")
                continue

            for msg in record.get("messages", []):
                if "length" in msg:
                    lengths.append(msg["length"])
    print(f"共从 {line_num} 行中解析出 {len(lengths)} 个有效长度。")
    return lengths


import re

def parse_ttft_from_output(text: str):

    if not text:
        return None
    m = re.search(
        r'Average time to first token \(s\)\s*\|\s*([0-9]+\.[0-9]+)',
        text
    )
    if m:
        return float(m.group(1))
    # 从 Percentile 表格中取 50% TTFT
    m = re.search(
        r'\|\s*50%\s*\|\s*([0-9]+\.[0-9]+)',
        text
    )
    if m:
        return float(m.group(1))
    
    return None


def run_evalscope_for_length(length):

    cmd = EVALSCOPE_CMD.format(len=length)
    print(f"\n>>> 正在测试 prompt 长度 {length} ...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        output_text = result.stdout + result.stderr
        # # 只在返回码非0 且 stdout 里没有 TTFT 关键字时才判为失败
        # if "Average time to first token" in result.stdout:
        #     ttft = parse_ttft_from_output(result.stdout)
        #     if ttft is not None:
        #         print(f"✅ TTFT(avg) | len={length}: {ttft:.3f}s")
        #         return ttft
        # else:
        #     # 只有在完全没有结果时，才认为是失败
        #     print(f"⚠️ evalscope 未输出 TTFT，长度 {length}，stderr 可能为警告：{result.stderr.strip()[:200]}")
        #     return None


        ttft = parse_ttft_from_output(output_text)
        if ttft is not None:
            print(f"TTFT(avg) | len={length}: {ttft:.3f}s")
        else:
            print("未解析到 TTFT")
        return ttft
    except subprocess.TimeoutExpired:
        print(f"执行超时，长度 {length}")
        return None

def main():
    lengths = load_query_lengths()
    print(f"已加载 {len(lengths)} 个 query 长度用于模拟 Full Reuse TTFT")

    # 若之前已有结果文件，读取以便断点续跑
    existing_results = {}
    if OUTPUT_LOG.exists():
        try:
            with open(OUTPUT_LOG, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                existing_results = {r["idx"]: r for r in loaded}
            print(f"检测到已有 {len(existing_results)} 条历史结果，将跳过这些样本。")
        except Exception as e:
            print(f"历史结果读取失败：{e}")

    results = []
    cumulative_ttft = []

    for idx, length in enumerate(lengths, 1):
        if idx in existing_results:
            entry = existing_results[idx]
            print(f"⏭ 跳过已完成的样本 idx={idx}, 长度={length}, TTFT={entry['avg_ttft']:.3f}s")
        else:
            ttft = run_evalscope_for_length(length)
            entry = {"idx": idx, "length": length, "avg_ttft": ttft}

            # 实时保存
            if ttft is not None:
                existing_results[idx] = entry
                with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
                    json.dump(list(existing_results.values()), f, indent=2, ensure_ascii=False)

                print(f"✅ idx={idx}, 长度={length}, TTFT={ttft:.3f}s 已保存")
            else:
                print(f"idx={idx}, 长度={length}, TTFT 未获取到")

        results.append(entry)
        if entry["avg_ttft"] is not None:
            cumulative_ttft.append(entry["avg_ttft"])
            print(f"   🔹 当前累计平均 TTFT: {statistics.mean(cumulative_ttft):.3f}s")

    if cumulative_ttft:
        print("\n=== 全量 Full Reuse TTFT 汇总 ===")
        print(f"总 query 样本数: {len(cumulative_ttft)}")
        print(f"平均 TTFT (Full Reuse 模拟): {statistics.mean(cumulative_ttft):.3f}s")
    else:
        print("❌ 未记录到有效 TTFT 数据")


if __name__ == "__main__":
    main()
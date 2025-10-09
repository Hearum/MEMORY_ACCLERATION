import re

log_path = "/home/shm/document/MEMORY_ACCLERATION/log/sglang_simplerag_locomo10_2025-10-08_17-32-17.log"

rates = []
with open(log_path, "r") as f:
    for line in f:
        if "Prefill batch." in line:
            match = re.search(r"#new-token: (\d+), #cached-token: (\d+)", line)
            if match:
                new_token = int(match.group(1))
                cached_token = int(match.group(2))
                total = new_token + cached_token
                if total > 0:
                    rates.append(cached_token / total)

if rates:
    print(f"Average per-request KV reuse: {sum(rates)/len(rates)*100:.2f}%")
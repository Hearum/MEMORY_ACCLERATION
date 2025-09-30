import re

log_path = "/home/shm/document/MEMORY_ACCLERATION/log_file.log"

total_new = 0
total_cached = 0

with open(log_path, "r") as f:
    for line in f:
        if "Prefill batch." in line:
            match = re.search(r"#new-token: (\d+), #cached-token: (\d+)", line)
            if match:
                new_token = int(match.group(1))
                cached_token = int(match.group(2))
                total_new += new_token
                total_cached += cached_token

total_tokens = total_new + total_cached
kv_hit_rate = total_cached / total_tokens if total_tokens > 0 else 0.0

print(f"Total tokens: {total_tokens}")
print(f"Total cached tokens: {total_cached}")
print(f"KV cache hit rate: {kv_hit_rate*100:.2f}%")

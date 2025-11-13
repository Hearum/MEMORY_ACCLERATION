import json
import matplotlib.pyplot as plt

log_file = "/home/shm/document/log/QA/log_o.json"

def visualize_token_distribution(log_file=log_file):
    token_counts = []

    # 读取每行 JSON
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 防止空行
                entry = json.loads(line)
                token_counts.append(entry.get("prompt_tokens", 0))

    if not token_counts:
        print("No token data found in the log file.")
        return

    # 计算均值
    mean_tokens = sum(token_counts) / len(token_counts)
    print(f"Average prompt token length: {mean_tokens:.2f}")

    # 可视化分布
    plt.figure(figsize=(10,6))
    plt.hist(token_counts, bins=50, color='skyblue', edgecolor='black')
    plt.title("Prompt Token Length Distribution")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("/home/shm/document/log/QA/show_o.jpg")

visualize_token_distribution()

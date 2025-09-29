from huggingface_hub import snapshot_download

save_dir = "/mnt/data/models/Meta-Llama-3-8B-Instruct"

snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    local_dir=save_dir,
    local_dir_use_symlinks=False  # 避免软链接，固定目录
)
# modelscope download --model LLM-Research/Llama-3.2-3B-Instruct --local_dir /mnt/data/models
import os
from huggingface_hub import hf_hub_download, snapshot_download


"""
设置临时变量

linux:
    export HF_TOKEN="your token"

powershell:
    $env:HF_TOKEN = "your token"

"""
hf_token = os.getenv("HF_TOKEN", "")

endpoint = "https://hf-mirror.com"
proxies = {"https": "http://localhost:7897"}


snapshot_download(
    repo_id = "maidalun1020/bce-embedding-base_v1",
    local_dir = "./models/bce-embedding-base_v1",
    # proxies = proxies,
    max_workers = 8,
    # endpoint = endpoint,
    token = hf_token
)

snapshot_download(
    repo_id = "maidalun1020/bce-reranker-base_v1",
    local_dir = "./models/bce-reranker-base_v1",
    # proxies = proxies,
    max_workers = 8,
    # endpoint = endpoint,
    token = hf_token
)

snapshot_download(
    repo_id = "internlm/internlm2-chat-1_8b",
    local_dir = "./models/internlm2-chat-1_8b",
    # proxies = proxies,
    max_workers = 8,
    # endpoint = endpoint,
)

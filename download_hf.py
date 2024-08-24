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


EMBEDDING_MODEL_PATH: str = "./models/bce-embedding-base_v1"
RERANKER_MODEL_PATH: str = "./models/bce-reranker-base_v1"
MODEL_PATH = "./models/internlm2_5-1_8b-chat"


snapshot_download(
    repo_id = "maidalun1020/bce-embedding-base_v1",
    local_dir = EMBEDDING_MODEL_PATH,
    # proxies = proxies,
    max_workers = 8,
    # endpoint = endpoint,
    token = hf_token
)

snapshot_download(
    repo_id = "maidalun1020/bce-reranker-base_v1",
    local_dir = RERANKER_MODEL_PATH,
    # proxies = proxies,
    max_workers = 8,
    # endpoint = endpoint,
    token = hf_token
)

snapshot_download(
    repo_id = "internlm/internlm2_5-1_8b-chat",
    local_dir = MODEL_PATH,
    # proxies = proxies,
    max_workers = 8,
    # endpoint = endpoint,
)

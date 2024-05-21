import os
from huggingface_hub import hf_hub_download, snapshot_download


endpoint = "https://hf-mirror.com"
proxies = {"https": "http://localhost:7897"}
# 下载 bce 模型需要在 huggingface 网站同意协议,然后创建 token,将 token 替换为自己的就可以下载
hf_token = ""


# 设置环境变量
# os.environ['HF_ENDPOINT'] = endpoint
# 下载模型
# os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir ./models/paraphrase-multilingual-MiniLM-L12-v2')


snapshot_download(
    repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_dir = "./models/paraphrase-multilingual-MiniLM-L12-v2",
    # proxies = proxies,
    max_workers = 8,
    # endpoint = endpoint,
)

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

# snapshot_download(
#     repo_id = "internlm/internlm2-chat-1_8b",
#     local_dir = "./models/internlm2-chat-1_8b",
#     # proxies = proxies,
#     max_workers = 8,
#     # endpoint = endpoint,
# )

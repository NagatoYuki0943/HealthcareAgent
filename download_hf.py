import os
from huggingface_hub import hf_hub_download, snapshot_download


endpoint = "https://hf-mirror.com"
# proxies = {"https": "http://localhost:7897"}


# 设置环境变量
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 下载模型
# os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir ./models/sentence-transformer')


snapshot_download(
    repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_dir="./models/sentence-transformer",
    resume_download=True,
    # proxies=proxies,
    max_workers=8,
    endpoint=endpoint,
)

snapshot_download(
    repo_id="internlm/internlm2-chat-1_8b",
    local_dir="./models/internlm2-chat-1_8b",
    resume_download=True,
    # proxies=proxies,
    max_workers=8,
    endpoint=endpoint,
)

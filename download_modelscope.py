from modelscope import snapshot_download


EMBEDDING_MODEL_PATH: str = "./models/bce-embedding-base_v1"
RERANKER_MODEL_PATH: str = "./models/bce-reranker-base_v1"


# 下载 embedding 和 reranker 模型,不会重复下载
snapshot_download(
    "maidalun/bce-embedding-base_v1",
    local_dir = EMBEDDING_MODEL_PATH,
)
snapshot_download(
    "maidalun/bce-reranker-base_v1",
    local_dir = RERANKER_MODEL_PATH,
)

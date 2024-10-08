from vector_database import VectorDatabase


DATA_PATH: str = "./data"
EMBEDDING_MODEL_PATH: str = "./models/bce-embedding-base_v1"
RERANKER_MODEL_PATH: str = "./models/bce-reranker-base_v1"
PERSIST_DIRECTORY: str = "./vector_db/faiss"
SIMILARITY_TOP_K: int = 4
SIMILARITY_FETCH_K: int = 10
SCORE_THRESHOLD: float = 0.15
ALLOW_SUFFIX: tuple[str] = (".txt", ".md", ".docx", ".doc", ".pdf")
VECTOR_DEVICE = "cuda"
TEXT_SPLITTER_TYPE = "RecursiveCharacterTextSplitter"

# 向量数据库
vector_database = VectorDatabase(
    data_path=DATA_PATH,
    embedding_model_path=EMBEDDING_MODEL_PATH,
    reranker_model_path=RERANKER_MODEL_PATH,
    persist_directory=PERSIST_DIRECTORY,
    similarity_top_k=SIMILARITY_TOP_K,
    similarity_fetch_k=SIMILARITY_FETCH_K,
    score_threshold=SCORE_THRESHOLD,
    allow_suffix=ALLOW_SUFFIX,
    device=VECTOR_DEVICE,
    text_splitter_type=TEXT_SPLITTER_TYPE,
)

# 创建数据库
vector_database.create_faiss_vectordb(force=True)

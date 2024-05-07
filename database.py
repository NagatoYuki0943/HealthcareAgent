from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os
from loguru import logger
from utils import get_filename, format_documents, format_references


class VectorDatabase:
    def __init__(
        self,
        data_path: str = "./data",
        embedding_model_path: str = "./models/bce-embedding-base_v1",
        reranker_model_path : str | None = "./models/bce-reranker-base_v1",
        persist_directory: str = "./vector_db/faiss",
        similarity_top_k: int = 7,
        score_threshold: float = 0.15,
        allow_suffix: tuple[str] | str = (".txt", ".md", ".docx", ".doc", ".pdf"),
    ):
        """
        Args:
            data_path (str, optional): 数据集路径. Defaults to "./data".
            embedding_model_path (str, optional): embedding 模型路径. Defaults to "./models/bce-embedding-base_v1".
            reranker_model_path (str | None, optional): embedding 模型路径,可以为空,使用 rerank 必须提供. Defaults to "./models/bce-reranker-base_v1".
            persist_directory (str, optional): 数据持久化路径. Defaults to "./vector_db/faiss".
            similarity_top_k (int, optional): 相似数据 top_k. Defaults to 7.
            score_threshold (float, optional): 最低分数. Defaults to 0.15.
            allow_suffix (tuple[str] | str, optional): 读取文件的后缀. Defaults to (".txt", ".md", ".docx", ".doc", ".pdf").
        """
        self.data_path = data_path
        self.persist_directory = persist_directory
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path
        self.similarity_top_k = similarity_top_k
        self.score_threshold = score_threshold
        self.allow_suffix = allow_suffix

        # 加载开源词向量模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name = embedding_model_path,
            model_kwargs = {'device': 'cuda'},
            encode_kwargs = {
                'normalize_embeddings': True
            }
        )
        self.embeddings.client = self.embeddings.client.half()
        self.retriever = None

    def get_files(self, dir_path: str,) -> list:
        """遍历文件夹获取所有目标文件路径

        Args:
            dir_path (str): 目标文件夹

        Returns:
            list: 目标文件路径列表
        """
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(dir_path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(self.allow_suffix):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_text(self, dir_path: str) -> list:
        """遍历文件夹获取所有目标文件

        Args:
            dir_path (str): 目标文件夹

        Returns:
            list: 目标文件列表
        """
        # args：dir_path，目标文件夹路径
        # 首先调用上文定义的函数得到目标文件路径列表
        file_lst = self.get_files(dir_path)
        # docs 存放加载之后的纯文本对象
        docs = []
        # 遍历所有目标文件
        for one_file in tqdm(file_lst):
            if one_file.endswith(".txt"):
                # txt, md, docx, doc: pip install unstructured
                loader = UnstructuredFileLoader(one_file)
            elif one_file.endswith(".md"):
                loader = UnstructuredMarkdownLoader(one_file)
            elif one_file.endswith((".docx", ".doc")):
                # pip install python-docx
                loader = UnstructuredWordDocumentLoader(one_file)
            elif one_file.endswith(".pdf"):
                # pip install pypdf
                loader = PyPDFLoader(one_file)
            docs.extend(loader.load())
        return docs

    def create_faiss_vectordb(self, force: bool = False):
        """创建数据库

        Args:
            force (bool, optional): 强制重新创建数据库. Defaults to False.
        """
        if os.path.exists(self.persist_directory):
            if not force:
                logger.warning(f"`{self.persist_directory}` 路径已存在, 无需创建数据库, 直接读取数据库即可, 如果想强制重新传建, 请设置参数 `force = True`")
                self.load_faiss_vectordb()
                return
            else:
                from shutil import rmtree
                rmtree(self.persist_directory)
                logger.warning(f"\033[0;31;40m`{self.persist_directory}` 路径已删除,即将重新创建数据库\033[0m")

        from langchain_community.vectorstores import FAISS

        dirs = os.listdir(self.data_path)
        dirs = [os.path.join(self.data_path, dir) for dir in dirs]
        dirs = [dir for dir in dirs if os.path.isdir(dir)]

        # 加载目标文件
        docs = []
        for dir_path in dirs:
            docs.extend(self.get_text(dir_path))

        # 对文本进行分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 150
        )
        split_docs = text_splitter.split_documents(docs)

        # 构建向量数据库
        self.vectordb = FAISS.from_documents(
            documents = split_docs,
            embedding = self.embeddings,
        )

        # 将加载的向量数据库持久化到磁盘上
        self.vectordb.save_local(folder_path = self.persist_directory)

    def load_faiss_vectordb(self):
        """载入数据库"""
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy

        # 加载数据库
        self.vectordb = FAISS.load_local(
            folder_path = self.persist_directory,
            embeddings = self.embeddings,
            allow_dangerous_deserialization = True, # 允许读取pickle
            # faiss 仅支持 EUCLIDEAN_DISTANCE MAX_INNER_PRODUCT COSINE
            distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT,  # refer: https://github.com/InternLM/HuixiangDou/blob/main/huixiangdou/service/retriever.py
            normalize_L2 = False
        )

    def create_faiss_retriever(self):
        """创建 Retriever"""
        # search_type: 'similarity', 'similarity_score_threshold', 'mmr'
        self.retriever: VectorStoreRetriever = self.vectordb.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {
                "k": self.similarity_top_k,
                "score_threshold": self.score_threshold,
                "fetch_k": self.similarity_top_k * 5
            }
        )

    def create_faiss_reranker_retriever(self):
        """创建重排序 Retriever"""
        assert self.reranker_model_path is not None, "使用 reranker 必须指定 `reranker_model_path`"

        from BCEmbedding.tools.langchain import BCERerank
        from langchain.retrievers import ContextualCompressionRetriever

        # search_type: 'similarity', 'similarity_score_threshold', 'mmr'
        retriever: VectorStoreRetriever = self.vectordb.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {
                "k": self.similarity_top_k * 5,
                "score_threshold": self.score_threshold,
            }
        )

        # 载入reranker模型
        reranker = BCERerank(
            top_n = self.similarity_top_k,
            model = self.reranker_model_path,
            device = 'cuda',
            use_fp16 = True
        )

        # 创建检索器
        self.retriever = ContextualCompressionRetriever(
            base_compressor = reranker,
            base_retriever = retriever
        )

    def similarity_search(
        self,
        query: str,
    ) -> tuple[str, str]:
        assert self.retriever is not None, "请先调用 `create_faiss_retriever` 或者 `create_faiss_reranker_retriever` 创建检索器"

        # similarity search
        documents = self.retriever.get_relevant_documents(query)
        documents_str = format_documents(documents)
        # 获取参考文档并去重
        references = list(set([get_filename(doc.metadata['source']) for doc in documents]))
        references_str = format_references(references)
        return documents_str, references_str


if __name__ == "__main__":
    vector_database = VectorDatabase()
    vector_database.create_faiss_vectordb()
    vector_database.load_faiss_vectordb()
    vector_database.create_faiss_retriever()
    documents_str, references_str = vector_database.similarity_search("Eye Pressure Lowering Effect of Vitamin C")
    logger.info(f"references_str: {references_str}")
    documents_str, references_str = vector_database.similarity_search("吃了吗")
    logger.info(f"references_str: {references_str}")

    vector_database.create_faiss_reranker_retriever()
    documents_str, references_str = vector_database.similarity_search("Eye Pressure Lowering Effect of Vitamin C")
    logger.info(f"references_str: {references_str}")
    documents_str, references_str = vector_database.similarity_search("吃了吗")
    logger.info(f"references_str: {references_str}")

# 首先导入所需第三方库
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os
from utils import get_filename, format_references


# 获取文件路径函数
# 获取文件路径函数
def get_files(dir_path: str) -> list:
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 只读取pdf
            if filename.endswith(".pdf"):
                file_list.append(os.path.join(filepath, filename))
            # 通过后缀名判断文件类型是否满足要求
            # if filename.endswith((".txt", ".md", ".docx", ".doc", ".pdf")):
            #     file_list.append(os.path.join(filepath, filename))
    return file_list


# 加载文件函数
def get_text(dir_path: str) -> list:
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
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


def create_vectordb(
    tar_dirs: str = "./data",
    embedding_dir: str = "./models/sentence-transformer",
    persist_directory: str = "./vector_db/chroma"
):
    dirs = os.listdir(tar_dirs)
    dirs = [os.path.join(tar_dirs, dir) for dir in dirs]
    dirs = [dir for dir in dirs if os.path.isdir(dir)]

    # 加载目标文件
    docs = []
    for dir_path in dirs:
        docs.extend(get_text(dir_path))

    # 对文本进行分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    # 加载开源词向量模型
    embeddings = HuggingFaceEmbeddings(model_name=embedding_dir)

    # 构建向量数据库

    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()


def load_vectordb(
    embedding_dir: str,
    persist_directory: str,
) -> Chroma:
    # 加载开源词向量模型
    embeddings = HuggingFaceEmbeddings(model_name=embedding_dir)
    # 加载数据库
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vectordb


def similarity_search(
    vectordb: Chroma,
    query: str,
    similarity_top_k: int = 4,
) -> tuple[str, list, str]:
    # similarity search
    documents_with_score = vectordb.similarity_search_with_score(query=query, k=similarity_top_k)
    documents, scores = zip(*documents_with_score)
    documents_str = "\n".join([doc.page_content for doc in documents])
    # 获取参考文档并去重
    references = list(set([get_filename(doc.metadata['source']) for doc in documents]))
    references_str = format_references(references)
    return documents_str, references_str

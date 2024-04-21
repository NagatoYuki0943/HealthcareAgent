from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def load_chain(
    llm_path: str = "./models/internlm2-chat-1_8b",
    embedding_model_name: str = "./models/sentence-transformer",
    persist_directory: str = "./vector_db/chroma",
    adapter_dir: str = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    system_prompt: str = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """
):
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # 向量数据库持久化路径
    persist_directory = persist_directory

    # 加载数据库
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    )

    llm = InternLM_LLM(
        pretrained_model_name_or_path=llm_path,
        adapter_dir=adapter_dir,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        system_prompt=system_prompt
    )

    # 你可以修改这里的 prompt template 来试试不同的问答效果
    template = """请使用以下提供的上下文来回答用户的问题。如果无法从上下文中得到答案，请回答你不知道，并总是使用中文回答。
    提供的上下文：
    ···
    {context}
    ···
    用户的问题: {question}
    你给的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    return qa_chain

if __name__ == "__main__":
    load_chain()

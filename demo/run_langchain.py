# 导入必要的库
from load_chain import load_chain


class ModelCenter():
    """
    存储问答 Chain 的对象
    """
    def __init__(self):
        self.chain = load_chain(
        llm_path = "../models/internlm2-chat-1_8b",
        embedding_model_name = "./sentence-transformer",
        persist_directory = "./vector_db/chroma"
    )

    def qa_chain_self_answer(self, query: str, history: list = []):
        """
        调用不带历史记录的问答链进行回答
        """
        if query == None or len(query) < 1:
            return "", history
        try:
            # invoke(input: Dict[str, Any], config: Optional[langchain_core.runnables.config.RunnableConfig] = None, **kwargs: Any) -> Dict[str, Any]
            # method of langchain.chains.retrieval_qa.base.RetrievalQA instance
            response = self.chain.invoke({"query": query})["result"]
            history.append(
                (query, response))
            return response, history
        except Exception as e:
            return e, history


model_center = ModelCenter()


while True:
    query = input("请输入提示:")
    if query == None or len(query) < 1:
        continue
    if query.lower() == "exit":
        break

    # 不使用历史记录
    history = []
    response, history = model_center.qa_chain_self_answer(query, history)
    # print("history:", history)
    print("回答:", response)

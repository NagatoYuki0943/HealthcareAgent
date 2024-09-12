# 导入必要的库
from old_file.load_chain import load_chain


class ModelCenter:
    """
    存储问答 Chain 的对象
    """

    def __init__(self):
        self.chain = load_chain(
            llm_path="./models/internlm2_5-1_8b-chat",
            embedding_model_name="./models/sentence-transformer",
            persist_directory="./vector_db/chroma",
            adapter_dir="./models/internlm2_chat_1_8b_qlora_huatuo_e3/epoch_3_hf",
            load_in_8bit=False,
            load_in_4bit=False,
            system_prompt="你现在是一名医生，具备丰富的医学知识和临床经验。你擅长诊断和治疗各种疾病，能为病人提供专业的医疗建议。你有良好的沟通技巧，能与病人和他们的家人建立信任关系。请在这个角色下为我解答以下问题。",
        )

    def qa_chain_self_answer(
        self,
        query: str,
        history: list = [],
    ) -> list:
        """
        调用不带历史记录的问答链进行回答
        """

        if query == None or len(query) < 1:
            return history
        try:
            # invoke(input: Dict[str, Any], config: Optional[langchain_core.runnables.config.RunnableConfig] = None, **kwargs: Any) -> Dict[str, Any]
            # method of langchain.chains.retrieval_qa.base.RetrievalQA instance
            # ['query', 'result', 'source_documents']
            response: dict = self.chain.invoke(input={"query": query})
            history.append([query, response["result"]])
            return history
        except Exception as e:
            return history

    def qa_chain_self_answer_stream(
        self,
        query: str,
        history: list = [],
    ):
        """
        调用不带历史记录的问答链进行回答
        """
        query = query.strip()
        if query == None or len(query) < 1:
            ...
            # yield history
        try:
            # stream(input: 'Input', config: 'Optional[RunnableConfig]' = None, **kwargs: 'Optional[Any]') -> 'Iterator[Output]' method of langchain.chains.retrieval_qa.base.RetrievalQA instance    Default implementation of stream, which calls invoke.
            # Subclasses should override this method if they support streaming output.
            for response in self.chain.stream(input={"query": query}):
                print(response)
                # yield history + [query, response]
        except Exception as e:
            ...
            # yield history


model_center = ModelCenter()


while True:
    query = input("请输入提示:")
    if query == None or len(query) < 1:
        continue
    if query.lower() == "exit":
        break

    # 不使用历史记录
    history = []
    history = model_center.qa_chain_self_answer(query, history)
    # print("history:", history)
    print("回答:", history)

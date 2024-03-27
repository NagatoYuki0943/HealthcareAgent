# 导入必要的库
import gradio as gr
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
            return "", history
        except Exception as e:
            return e, history


model_center = ModelCenter()


block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>InternLM</center></h1>
                <center>书生浦语</center>
                """)
        # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")

            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")

        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)

# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()

# 导入必要的库
import gradio as gr
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
            adapter_dir=None,
            load_in_8bit=False,
            load_in_4bit=False,
            system_prompt="你现在是一名医生，具备丰富的医学知识和临床经验。你擅长诊断和治疗各种疾病，能为病人提供专业的医疗建议。你有良好的沟通技巧，能与病人和他们的家人建立信任关系。请在这个角色下为我解答以下问题。",
        )

    def qa_chain_self_answer(self, query: str, history: list = []):
        """
        调用不带历史记录的问答链进行回答

        history: [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        """
        if query is None or len(query) < 1:
            return history
        try:
            # invoke(input: Dict[str, Any], config: Optional[langchain_core.runnables.config.RunnableConfig] = None, **kwargs: Any) -> Dict[str, Any]
            # method of langchain.chains.retrieval_qa.base.RetrievalQA instance
            response = self.chain.invoke(input={"query": query})["result"]
            history.append([query, response])
            return history
        except Exception as e:
            return history


model_center = ModelCenter()


def revocery(history: list = []) -> tuple[str, list]:
    """恢复到上一轮对话"""
    query = ""
    if len(history) > 0:
        query, _ = history.pop(-1)
    return query, history


def main():
    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown("""<h1><center>InternLM</center></h1>
                    <center>InternLM2</center>
                    """)
            # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

        with gr.Row():
            with gr.Column(scale=4):
                # 创建聊天框
                chatbot = gr.Chatbot(height=500, show_copy_button=True)

                with gr.Row():
                    max_new_tokens = gr.Slider(
                        minimum=1,
                        maximum=2048,
                        value=1024,
                        step=1,
                        label="Maximum new tokens",
                    )
                    top_p = gr.Slider(
                        minimum=0.01, maximum=1, value=0.8, step=0.01, label="Top_p"
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=100, value=40, step=1, label="Top_k"
                    )
                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=2,
                        value=0.8,
                        step=0.01,
                        label="Temperature",
                    )

                with gr.Row():
                    # 创建一个文本框组件，用于输入 prompt。
                    query = gr.Textbox(label="Prompt/问题")
                    # 创建提交按钮。
                    # variant https://www.gradio.app/docs/button
                    # scale https://www.gradio.app/guides/controlling-layout
                    submit = gr.Button("💬 Chat", variant="primary", scale=0)

                with gr.Row():
                    # 创建一个重新生成按钮，用于重新生成当前对话内容。
                    # regen = gr.Button("🔄 Retry", variant="secondary")
                    # undo = gr.Button("↩️ Undo", variant="secondary")
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(
                        components=[chatbot], value="🗑️ Clear", variant="stop"
                    )

            # 回车提交
            query.submit(
                model_center.qa_chain_self_answer,
                inputs=[query, chatbot],
                outputs=[chatbot],
            )

            # 清空query
            query.submit(
                lambda: gr.Textbox(value=""),
                [],
                [query],
            )

            # 按钮提交
            submit.click(
                model_center.qa_chain_self_answer,
                inputs=[query, chatbot],
                outputs=[chatbot],
            )

            # 清空query
            submit.click(
                lambda: gr.Textbox(value=""),
                [],
                [query],
            )

            # # 重新生成
            # regen.click(
            #     model_center.qa_chain_self_answer,
            #     inputs=[query, chatbot],
            #     outputs=[chatbot]
            # )

            # # 撤销
            # undo.click(
            #     model_center.qa_chain_self_answer,
            #     inputs=[chatbot],
            #     outputs=[query, chatbot]
            # )

        gr.Markdown("""提醒：<br>
        1. 初始化数据库时间可能较长，请耐心等待。
        2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
        """)

    # threads to consume the request
    gr.close_all()

    # 设置队列启动，队列最大长度为 100
    demo.queue(max_size=100)

    # 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
    # demo.launch(share=True, server_port=int(os.environ['PORT1']))
    # 直接启动
    # demo.launch(server_name="127.0.0.1", server_port=7860)
    demo.launch()


if __name__ == "__main__":
    main()

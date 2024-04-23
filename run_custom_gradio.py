import os
from infer_engine import InferEngine, TransformersConfig
from create_db import load_faiss_vectordb, similarity_search
import gradio as gr
from typing import Generator, Any


print("gradio version: ", gr.__version__)


EMBEDDING_MODEL_PATH = "./models/paraphrase-multilingual-MiniLM-L12-v2"
PERSIST_DIRECTORY = "./vector_db/faiss"

# 载入向量数据库
vectordb = load_faiss_vectordb(
    embedding_model_path = EMBEDDING_MODEL_PATH,
    persist_directory = PERSIST_DIRECTORY
)

# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = './models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_DIR = None
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False

SYSTEM_PROMPT = "你现在是一名医生，具备丰富的医学知识和临床经验。你擅长诊断和治疗各种疾病，能为病人提供专业的医疗建议。你有良好的沟通技巧，能与病人和他们的家人建立信任关系。请在这个角色下为我解答以下问题。"

TEMPLATE = """请使用以下提供的上下文来回答用户的问题。如果无法从上下文中得到答案，请回答你不知道，并总是使用中文回答。
提供的上下文：
···
{context}
···
用户的问题: {question}
你给的回答:"""

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_dir = ADAPTER_DIR,
    load_in_8bit = LOAD_IN_8BIT,
    load_in_4bit = LOAD_IN_4BIT,
    system_prompt = SYSTEM_PROMPT
)

# 载入模型
infer_engine = InferEngine(
    backend = 'transformers', # transformers, lmdeploy
    transformers_config = TRANSFORMERS_CONFIG,
)


def chat(
    query: str,
    history: list = [],  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    top_p: float = 0.8,
    top_k: int = 40,
    temperature: float = 0.8,
    similarity_top_k: int = 4,
    regenerate: bool = False
) -> Generator[Any, Any, Any]:
    # 重新生成时要把最后的query和response弹出,重用query
    if regenerate:
        # 有历史就重新生成,没有历史就返回空
        if len(history) > 0:
            query, _ = history.pop(-1)
        else:
            yield history
            return
    else:
        query = query.strip()
        if query == None or len(query) < 1:
            yield history
            return

    print({
        "similarity_top_k": similarity_top_k
    })

    # 只有第一轮才使用rag
    if len(history) == 0:
        # similarity search
        documents_str, references_str = similarity_search(
            vectordb = vectordb,
            query = query,
            similarity_top_k = similarity_top_k,
        )
        prompt = TEMPLATE.format(context=documents_str, question=query)
        print(prompt)
    else:
        prompt = query
        references_str = ""

    print(f"query: {query}; response: ", end="", flush=True)
    length = 0
    for response, history in infer_engine.chat_stream(
        query = query,
        history = history,
        max_new_tokens = max_new_tokens,
        top_p = top_p,
        top_k = top_k,
        temperature = temperature,
    ):
        print(response[length:], flush=True, end="")
        length = len(response)
        yield history
    # 加上参考文档
    yield history[:-1] + [[query, response + references_str]]
    print("\n")


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
                    <center>Medical-RAG</center>
                    """)
            # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

        with gr.Row():
            with gr.Column(scale=4):
                # 创建聊天框
                chatbot = gr.Chatbot(height=500, show_copy_button=True)

                with gr.Row():
                    # 创建一个文本框组件，用于输入 prompt。
                    query = gr.Textbox(label="Prompt/问题", placeholder="请输入你的问题，按 Enter 或者右边的按钮提交，按 Shift + Enter 可以换行")
                    # 创建提交按钮。
                    # variant https://www.gradio.app/docs/button
                    # scale https://www.gradio.app/guides/controlling-layout
                    submit = gr.Button("💬 Chat", variant="primary", scale=0)

                with gr.Row():
                    # 创建一个重新生成按钮，用于重新生成当前对话内容。
                    regen = gr.Button("🔄 Retry", variant="secondary")
                    undo = gr.Button("↩️ Undo", variant="secondary")
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(components=[chatbot], value="🗑️ Clear", variant="stop")

                # 折叠
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            minimum=1,
                            maximum=2048,
                            value=1024,
                            step=1,
                            label='Max new tokens'
                        )
                        top_p = gr.Slider(
                            minimum=0.01,
                            maximum=1,
                            value=0.8,
                            step=0.01,
                            label='Top_p'
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=40,
                            step=1,
                            label='Top_k'
                        )
                        temperature = gr.Slider(
                            minimum=0.01,
                            maximum=1.5,
                            value=0.8,
                            step=0.01,
                            label='Temperature'
                        )
                        similarity_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=4,
                            step=1,
                            label='Similar_Top_k'
                        )

            # 回车提交
            query.submit(
                chat,
                inputs=[query, chatbot, max_new_tokens, top_p, top_k, temperature, similarity_top_k],
                outputs=[chatbot]
            )

            # 清空query
            query.submit(
                lambda: gr.Textbox(value=""),
                [],
                [query],
            )

            # 按钮提交
            submit.click(
                chat,
                inputs=[query, chatbot, max_new_tokens, top_p, top_k, temperature, similarity_top_k],
                outputs=[chatbot]
            )

            # 清空query
            submit.click(
                lambda: gr.Textbox(value=""),
                [],
                [query],
            )

            # 重新生成
            regen.click(
                chat,
                inputs=[query, chatbot, max_new_tokens, top_p, top_k, temperature, similarity_top_k, regen],
                outputs=[chatbot]
            )

            # 撤销
            undo.click(
                revocery,
                inputs=[chatbot],
                outputs=[query, chatbot]
            )

        gr.Markdown("""提醒：<br>
        1. 第一次输入会使用 RAG 进行检索,后续对话是在 RAG 的检索结果基础上进行的。<br>
        2. 源码地址：https://github.com/NagatoYuki0943/medical-rag
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

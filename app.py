from load_model import load_model, load_vectordb
import gradio as gr
from typing import Generator, Any
from utils import get_filename, format_references


print("gradio version: ", gr.__version__)


# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = './models/internlm2-chat-1_8b'
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_DIR = None
# 量化
LOAD_IN_8BIT= False
LOAD_IN_4BIT = False
tokenizer, model = load_model(PRETRAINED_MODEL_NAME_OR_PATH, ADAPTER_DIR, LOAD_IN_8BIT, LOAD_IN_4BIT)

# vectordb
PERSIST_DIRECTORY = "./vector_db/chroma"
EMBEDDING_DIR = "./models/sentence-transformer"
vectordb = load_vectordb(PERSIST_DIRECTORY, EMBEDDING_DIR)

SYSTEM_PROMPT = "你现在是一名医生，具备丰富的医学知识和临床经验。你擅长诊断和治疗各种疾病，能为病人提供专业的医疗建议。你有良好的沟通技巧，能与病人和他们的家人建立信任关系。请在这个角色下为我解答以下问题。"
print("system_prompt: ", SYSTEM_PROMPT)

TEMPLATE = """请使用以下提供的上下文来回答用户的问题。如果无法从上下文中得到答案，请回答你不知道，并总是使用中文回答。
提供的上下文：
···
{context}
···
用户的问题: {question}
你给的回答:"""


def chat(
    query: str,
    history: list = [],  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    top_p: float = 0.8,
    top_k: int = 40,
    temperature: float = 0.8,
    similarity_top_k: int = 5,
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
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "similarity_top_k": similarity_top_k
    })

    # 只有第一轮才使用rag
    if len(history) == 0:
        # similarity search
        docuemnts = vectordb.similarity_search(query=query, k=similarity_top_k)
        similarity_documents = "\n".join([doc.page_content for doc in docuemnts])
        # 获取参考文档并去重
        similarity_documents_references = list(set([get_filename(doc.metadata['source']) for doc in docuemnts]))
        print(similarity_documents_references)
        similarity_documents_references = format_references(similarity_documents_references)
        prompt = TEMPLATE.format(context=similarity_documents, question=query)
        print(prompt)
    else:
        prompt = query
        similarity_documents_references = ""

    # generate response
    # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1185
    # stream_chat 返回的句子长度是逐渐边长的,length的作用是记录之前的输出长度,用来截断之前的输出
    print(f"query: {query}; response: ", end="", flush=True)
    length = 0
    for response, _ in model.stream_chat(
            tokenizer = tokenizer,
            query = prompt,
            history = history,
            max_new_tokens = max_new_tokens,
            do_sample = True,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            meta_instruction = SYSTEM_PROMPT,
        ):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)
            # yield _
            yield history + [[query, response]]
    # 最后添加reference
    yield history + [[query, response + similarity_documents_references]]
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
                        value=5,
                        step=1,
                        label='Similar_Top_k'
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
                    regen = gr.Button("🔄 Retry", variant="secondary")
                    undo = gr.Button("↩️ Undo", variant="secondary")
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(components=[chatbot], value="🗑️ Clear", variant="stop")

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
        1. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。<br>
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

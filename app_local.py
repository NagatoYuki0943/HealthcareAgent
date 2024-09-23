import gradio as gr
from typing import Generator, Sequence, Any
import threading
from loguru import logger
from infer_engine import InferEngine, TransformersConfig, LmdeployConfig
from vector_database import VectorDatabase
from utils import remove_history_references


log_file = logger.add("log/runtime_{time}.log", rotation="00:00")
logger.info(f"gradio version: {gr.__version__}")


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
vector_database.create_faiss_vectordb(force=False)
# 载入数据库(创建数据库后不需要载入也可以)
vector_database.load_faiss_vectordb()
# 创建相似度 retriever
# vector_database.create_faiss_retriever()
# 创建重排序 retriever
vector_database.create_faiss_reranker_retriever()

# clone 模型
PRETRAINED_MODEL_NAME_OR_PATH = "./models/internlm2_5-1_8b-chat"
# os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2_5-1_8b-chat.git {PRETRAINED_MODEL_NAME_OR_PATH}')
# os.system(f'cd {PRETRAINED_MODEL_NAME_OR_PATH} && git lfs pull')
ADAPTER_PATH = None
# 量化
LOAD_IN_8BIT = False
LOAD_IN_4BIT = False

SYSTEM_PROMPT = """
你是医疗保健智能体，名字叫做 "HeathcareAgent"。
    - "HeathcareAgent" 可以根据自己丰富的医疗知识来回答问题。
    - "HeathcareAgent" 的回答应该是有益的、诚实的和无害的。
    - "HeathcareAgent" 可以使用用户选择的语言（如英语和中文）进行理解和交流。
"""

TEMPLATE = """上下文:
<context>
{context}
</context>
问题:
<question>{question}</question>
请使用提供的上下文来回答问题，如果上下文不足请根据自己的知识给出合适的回答，回答应该有条理(除非用户指定了回答的语言，否则用户使用什么语言就用什么语言回答):"""
# 请使用提供的上下文来回答问题，如果上下文不足请根据自己的知识给出合适的回答，回答应该有条理:"""

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_path=ADAPTER_PATH,
    load_in_8bit=LOAD_IN_8BIT,
    load_in_4bit=LOAD_IN_4BIT,
    model_name="internlm2",
    system_prompt=SYSTEM_PROMPT,
)

LMDEPLOY_CONFIG = LmdeployConfig(
    model_path=PRETRAINED_MODEL_NAME_OR_PATH,
    backend="turbomind",
    model_name="internlm2",
    model_format="hf",
    cache_max_entry_count=0.5,  # 调整 KV Cache 的占用比例为0.5
    quant_policy=0,  # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
    system_prompt=SYSTEM_PROMPT,
    deploy_method="local",
    log_level="ERROR",
)

# 载入模型
infer_engine = InferEngine(
    backend="transformers",  # transformers, lmdeploy, api
    transformers_config=TRANSFORMERS_CONFIG,
    lmdeploy_config=LMDEPLOY_CONFIG,
)


class InterFace:
    global_session_id: int = 0
    lock = threading.Lock()


enable_btn = gr.update(interactive=True)
disable_btn = gr.update(interactive=False)
btn = dict[str, Any]


def chat(
    query: str,
    history: Sequence
    | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    session_id: int | None = None,
) -> Generator[tuple[Sequence, btn, btn, btn, btn], None, None]:
    history = [] if history is None else list(history)

    # 是否是有效的问题
    query = query.strip()
    if query is None or len(query) < 1:
        logger.warning("query is None, return history")
        yield history, enable_btn, enable_btn, enable_btn, enable_btn
        return
    logger.info(f"query: {query}")

    # 数据库检索
    documents_str, references_str = vector_database.similarity_search(
        query=query,
    )

    # 格式化rag文件
    prompt = (
        TEMPLATE.format(context=documents_str, question=query)
        if documents_str
        else query
    )
    logger.info(f"prompt: {prompt}")

    # 给模型的历史记录去除参考文档
    history_without_reference = remove_history_references(history=history)
    logger.info(f"history_without_reference: {history_without_reference}")

    yield history + [[query, None]], disable_btn, disable_btn, disable_btn, disable_btn

    for response in infer_engine.chat_stream(
        query=prompt,
        history=history_without_reference,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        session_id=session_id,
    ):
        yield history + [[query, response]], disable_btn, disable_btn, disable_btn, disable_btn

    # 加上参考文档
    yield history + [[query, response + references_str]], enable_btn, enable_btn, enable_btn, enable_btn
    logger.info(f"references_str: {references_str}")
    logger.info(
        f"history_without_rag: {history + [[query, response + references_str]]}"
    )


def regenerate(
    history: Sequence
    | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    session_id: int | None = None,
) -> Generator[tuple[Sequence, btn, btn, btn, btn], None, None]:
    history = [] if history is None else list(history)

    # 重新生成时要把最后的query和response弹出,重用query
    if len(history) > 0:
        query, _ = history.pop(-1)
        yield from chat(
            query=query,
            history=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            session_id=session_id,
        )
    else:
        logger.warning("no history, can't regenerate")
        yield history, enable_btn, enable_btn, enable_btn, enable_btn


def revocery(history: Sequence | None = None) -> tuple[str, Sequence]:
    """恢复到上一轮对话"""
    history = [] if history is None else list(history)
    query = ""
    if len(history) > 0:
        query, _ = history.pop(-1)
    return query, history


def main() -> None:
    block = gr.Blocks()
    with block as demo:
        state_session_id = gr.State(0)

        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown("""<h1><center>Healthcare Agent</center></h1>""")
            # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

        # 智能问答页面
        with gr.Tab("医疗智能问答"):
            with gr.Row():
                with gr.Column(scale=4):
                    # 创建聊天框
                    chatbot = gr.Chatbot(
                        height=500,
                        show_copy_button=True,
                        placeholder="内容由 AI 大模型生成，不构成专业医疗意见或诊断。",
                    )

                    # 组内的组件没有间距
                    with gr.Group():
                        with gr.Row():
                            # 创建一个文本框组件，用于输入 prompt。
                            query = gr.Textbox(
                                lines=1,
                                label="Prompt / 问题",
                                placeholder="Enter 发送; Shift + Enter 换行 / Enter to send; Shift + Enter to wrap",
                            )
                            # 创建提交按钮。
                            # variant https://www.gradio.app/docs/button
                            # scale https://www.gradio.app/guides/controlling-layout
                            submit = gr.Button("💬 Chat", variant="primary", scale=0)

                    gr.Examples(
                        examples=[
                            ["维生素E有什么作用，请详细说明"],
                            ["维生素C对治疗眼睛疾病有什么作用，请详细说明"],
                            [
                                "Please explain the effect of vitamin C on the treatment of eye diseases"
                            ],
                        ],
                        inputs=[query],
                        label="示例问题 / Example questions",
                    )

                    with gr.Row():
                        # 创建一个重新生成按钮，用于重新生成当前对话内容。
                        regen = gr.Button("🔄 Retry", variant="secondary")
                        undo = gr.Button("↩️ Undo", variant="secondary")
                        # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                        clear = gr.ClearButton(
                            components=[chatbot], value="🗑️ Clear", variant="stop"
                        )

                    # 折叠
                    with gr.Accordion("Advanced Options", open=False):
                        with gr.Row():
                            max_new_tokens = gr.Slider(
                                minimum=1,
                                maximum=2048,
                                value=1024,
                                step=1,
                                label="Max new tokens",
                            )
                            temperature = gr.Slider(
                                minimum=0.01,
                                maximum=2,
                                value=0.8,
                                step=0.01,
                                label="Temperature",
                            )
                            top_p = gr.Slider(
                                minimum=0.01,
                                maximum=1,
                                value=0.8,
                                step=0.01,
                                label="Top_p",
                            )
                            top_k = gr.Slider(
                                minimum=1, maximum=100, value=40, step=1, label="Top_k"
                            )

                # 回车提交(无法禁止按钮)
                query.submit(
                    chat,
                    inputs=[
                        query,
                        chatbot,
                        max_new_tokens,
                        temperature,
                        top_p,
                        top_k,
                        state_session_id,
                    ],
                    outputs=[chatbot, submit, regen, undo, clear],
                )

                # 清空query
                query.submit(
                    lambda: gr.Textbox(value=""),
                    inputs=[],
                    outputs=[query],
                )

                # 按钮提交
                submit.click(
                    chat,
                    inputs=[
                        query,
                        chatbot,
                        max_new_tokens,
                        temperature,
                        top_p,
                        top_k,
                        state_session_id,
                    ],
                    outputs=[chatbot, submit, regen, undo, clear],
                )

                # 清空query
                submit.click(
                    lambda: gr.Textbox(value=""),
                    inputs=[],
                    outputs=[query],
                )

                # 重新生成
                regen.click(
                    regenerate,
                    inputs=[
                        chatbot,
                        max_new_tokens,
                        temperature,
                        top_p,
                        top_k,
                        state_session_id,
                    ],
                    outputs=[chatbot, submit, regen, undo, clear],
                )

                # 撤销
                undo.click(revocery, inputs=[chatbot], outputs=[query, chatbot])

        gr.Markdown("""
        ### 内容由 AI 大模型生成，不构成专业医疗意见或诊断。
        """)

        # 初始化session_id
        def init():
            with InterFace.lock:
                InterFace.global_session_id += 1
            new_session_id = InterFace.global_session_id
            return new_session_id

        demo.load(init, inputs=None, outputs=[state_session_id])

    # threads to consume the request
    gr.close_all()

    # 设置队列启动
    demo.queue(
        max_size=None,  # If None, the queue size will be unlimited.
        default_concurrency_limit=40,  # 最大并发限制
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        max_threads=40,
    )


if __name__ == "__main__":
    main()

# https://github.com/NagatoYuki0943/xlab-huanhuan/blob/master/load/infer_engine_fastapi_server.py

from typing import Sequence
from loguru import logger
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from infer_engine import InferEngine, TransformersConfig, LmdeployConfig
from infer_utils import random_uuid_int
from vector_database import VectorDatabase


log_file = logger.add("log/runtime_{time}.log", rotation="00:00")


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


app = FastAPI()


# 与声明查询参数一样，包含默认值的模型属性是可选的，否则就是必选的。默认值为 None 的模型属性也是可选的。
class ChatRequest(BaseModel):
    messages: list[dict[str, str]] = Field(
        None,
        description="List of dictionaries containing the input text and the corresponding user id",
        examples=[
            [{"role": "user", "content": "维生素E对于眼睛有什么作用?"}]
        ]
    )
    max_tokens: int = Field(
        1024, ge=1, le=2048, description="Maximum number of new tokens to generate"
    )
    temperature: float = Field(
        0.8,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (lower temperature results in less random completions",
    )
    top_p: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling top-p (top-p sampling chooses from the smallest possible set of tokens whose cumulative probability mass exceeds the probability top_p)",
    )
    top_k: int = Field(
        50,
        ge=0,
        le=100,
        description="Top-k sampling chooses from the top k tokens with highest probability",
    )
    stream: bool = Field(
        False,
        description="Whether to stream the output or wait for the whole response before returning it",
    )


class Response(BaseModel):
    response: str = Field(
        None,
        description="Generated text response",
        examples=["InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室)."]
    )
    references: list[str] = Field(
        [],
        description="List of references retrieved from the database",
    )

    def __str__(self) -> str:
        return self.model_dump_json()


def generate(
    messages: Sequence[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    stream: bool = False,
) -> StreamingResponse | Response:
    content: str = messages[-1].get("content", "")

    # 数据库检索
    documents_str, references = vector_database.similarity_search(
        content,
    )

    # 格式化rag文件
    prompt = (
        TEMPLATE.format(context=documents_str, question=content)
        if documents_str
        else content
    )
    logger.info(f"prompt: {prompt}")

    # 更新最后一条消息
    messages[-1]['content'] = prompt

    if stream:
        async def generate():
            for response in infer_engine.chat_stream(
                messages,
                None,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                random_uuid_int(),
            ):
                # openai api returns \n\n as a delimiter for messages
                yield Response(response=response, references=[]).model_dump_json() + "\n\n"
            yield Response(response="", references=references).model_dump_json() + "\n\n"

        return StreamingResponse(generate())

    # 生成回复
    response = infer_engine.chat(
        messages,
        None,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        random_uuid_int(),
    )

    return Response(response=response, references=references)


# 将请求体作为 JSON 读取
# 在函数内部，你可以直接访问模型对象的所有属性
# http://127.0.0.1:8000/docs
@app.post("/chat", response_model=Response)
async def chat(request: ChatRequest) -> StreamingResponse | Response:
    print(request)

    if not request.messages or len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="No messages provided")

    role = request.messages[-1].get("role", "")
    if role not in ["user", "assistant"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    content = request.messages[-1].get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="content is empty")

    return generate(
        request.messages,
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.top_k,
        request.stream,
    )

# uvicorn app_local_fastapi_server:app --reload --port=8000
# uvicorn main:app --reload --port=8000
#   main: main.py 文件(一个 Python「模块」)。
#   app: 在 main.py 文件中通过 app = FastAPI() 创建的对象。
#   --reload: 让服务器在更新代码后重新启动。仅在开发时使用该选项。

# https://github.com/NagatoYuki0943/xlab-huanhuan/blob/master/load/infer_engine_fastapi_server.py
import os
import time
from typing import Sequence
from loguru import logger
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from infer_engine import InferEngine, ApiConfig
from infer_utils import random_uuid_int
from vector_database import VectorDatabase


log_file = logger.add("log/runtime_{time}.log", rotation="00:00")

# -------------------------silicon API---------------------------#
api_key = os.getenv("API_KEY", "")
logger.info(f"{api_key = }")


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

API_CONFIG = ApiConfig(
    base_url="https://api.siliconflow.cn/v1",
    api_key=api_key,
    model="internlm/internlm2_5-7b-chat",
    system_prompt=SYSTEM_PROMPT,
)

# 载入模型
infer_engine = InferEngine(
    backend="api",  # transformers, lmdeploy, api
    api_config=API_CONFIG,
)

app = FastAPI()


# 与声明查询参数一样，包含默认值的模型属性是可选的，否则就是必选的。默认值为 None 的模型属性也是可选的。
class ChatRequest(BaseModel):
    model: str | None = Field(
        None,
        description="The model used for generating the response",
        examples=["gpt4o", "gpt4"],
    )
    messages: list[dict[str, str | list]] = Field(
        None,
        description="List of dictionaries containing the input text and the corresponding user id",
        examples=[
            [{"role": "user", "content": "你是谁?"}],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "图片中有什么内容?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.jpg"},
                        },
                    ],
                }
            ],
        ],
    )
    max_tokens: int = Field(
        1024, ge=1, le=2048, description="Maximum number of new tokens to generate"
    )
    n: int = Field(
        1,
        ge=1,
        le=10,
        description="Number of completions to generate for each prompt",
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


# -------------------- 非流式响应模型 --------------------#
class ChatCompletionMessage(BaseModel):
    content: str | None = Field(
        None,
        description="The input text of the user or assistant",
        examples=["你是谁?"],
    )
    # 允许添加额外字段
    references: list[str] | None = Field(
        None,
        description="The reference text(s) used for generating the response",
        examples=[["book1", "book2"]],
    )
    role: str = Field(
        None,
        description="The role of the user or assistant",
        examples=["system", "user", "assistant"],
    )
    refusal: bool = Field(
        False,
        description="Whether the user or assistant refused to provide a response",
        examples=[False, True],
    )
    function_call: str | None = Field(
        None,
        description="The function call that the user or assistant made",
        examples=["ask_name", "ask_age", "ask_location"],
    )
    tool_calls: str | None = Field(
        None,
        description="The tool calls that the user or assistant made",
        examples=["weather", "calendar", "news"],
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


class ChatCompletionChoice(BaseModel):
    index: int = Field(
        None,
        description="The index of the choice",
        examples=[0, 1, 2],
    )
    finish_reason: str | None = Field(
        None,
        description="The reason for finishing the conversation",
        examples=[None, "stop"],
    )
    logprobs: list[float] | None = Field(
        None,
        description="The log probabilities of the choices",
        examples=[-1.3862943611198906, -1.3862943611198906, -1.3862943611198906],
    )
    message: ChatCompletionMessage | None = Field(
        None,
        description="The message generated by the model",
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


class CompletionUsage(BaseModel):
    prompt_tokens: int = Field(
        0,
        description="The number of tokens in the prompt",
        examples=[10],
    )
    completion_tokens: int = Field(
        0,
        description="The number of tokens in the completion",
        examples=[10],
    )
    total_tokens: int = Field(
        0,
        description="The total number of tokens generated",
        examples=[10],
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


class ChatCompletion(BaseModel):
    id: str | int | None = Field(
        None,
        description="The id of the conversation",
        examples=[123456, "abc123"],
    )
    choices: list[ChatCompletionChoice] = Field(
        [],
        description="The choices generated by the model",
    )
    created: int | float | None = Field(
        None,
        description="The timestamp when the conversation was created",
    )
    model: str | None = Field(
        None,
        description="The model used for generating the response",
        examples=["gpt4o", "gpt4"],
    )
    object: str = Field(
        "chat.completion",
        description="The object of the conversation",
        examples=["chat.completion"],
    )
    service_tier: str | None = Field(
        None,
        description="The service tier of the conversation",
        examples=["basic", "premium"],
    )
    system_fingerprint: str | None = Field(
        None,
        description="The system fingerprint of the conversation",
        examples=["1234567890abcdef"],
    )
    usage: CompletionUsage = Field(
        CompletionUsage(),
        description="The usage of the completion",
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


# -------------------- 非流式响应模型 --------------------#


# -------------------- 流式响应模型 --------------------#
class ChoiceDelta(ChatCompletionMessage): ...


class ChatCompletionChunkChoice(BaseModel):
    index: int = Field(
        None,
        description="The index of the choice",
        examples=[0, 1, 2],
    )
    finish_reason: str | None = Field(
        None,
        description="The reason for finishing the conversation",
        examples=[None, "stop"],
    )
    logprobs: list[float] | None = Field(
        None,
        description="The log probabilities of the choices",
        examples=[-1.3862943611198906, -1.3862943611198906, -1.3862943611198906],
    )
    delta: ChoiceDelta | None = Field(
        None,
        description="The message generated by the model",
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


class ChatCompletionChunk(BaseModel):
    id: str | int | None = Field(
        None,
        description="The id of the conversation",
        examples=[123456, "abc123"],
    )
    choices: list[ChatCompletionChunkChoice] = Field(
        [],
        description="The choices generated by the model",
    )
    created: int | float | None = Field(
        None,
        description="The timestamp when the conversation was created",
    )
    model: str | None = Field(
        None,
        description="The model used for generating the response",
        examples=["gpt4o", "gpt4"],
    )
    object: str = Field(
        "chat.completion.chunk",
        description="The object of the conversation",
        examples=["chat.completion.chunk"],
    )
    service_tier: str | None = Field(
        None,
        description="The service tier of the conversation",
        examples=["basic", "premium"],
    )
    system_fingerprint: str | None = Field(
        None,
        description="The system fingerprint of the conversation",
        examples=["1234567890abcdef"],
    )
    usage: CompletionUsage = Field(
        None,
        description="The usage of the completion",
    )

    def __repr__(self) -> str:
        return self.model_dump_json()


def rag_generate(
    messages: Sequence[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    stream: bool = False,
) -> StreamingResponse | ChatCompletion:
    content: str = messages[-1].get("content", "")
    content_len: int = len(content)

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
    messages[-1]["content"] = prompt

    session_id = random_uuid_int()

    if stream:

        async def generate():
            response_len = 0
            for response_str in infer_engine.chat_stream(
                messages,
                None,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                session_id,
            ):
                response_len += len(response_str)
                response = ChatCompletionChunk(
                    id=session_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            finish_reason=None,
                            delta=ChoiceDelta(
                                content=response_str,
                                role="assistant",
                            ),
                        )
                    ],
                    created=time.time(),
                    usage=CompletionUsage(
                        prompt_tokens=content_len,
                        completion_tokens=response_len,
                        total_tokens=content_len + response_len,
                    ),
                )
                print(response)
                # openai api returns \n\n as a delimiter for messages
                yield f"data: {response.model_dump_json()}\n\n"

            response = ChatCompletionChunk(
                id=session_id,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        finish_reason="stop",
                        delta=ChoiceDelta(
                            references=references,
                        ),
                    )
                ],
                created=time.time(),
                usage=CompletionUsage(
                    prompt_tokens=content_len,
                    completion_tokens=response_len,
                    total_tokens=content_len + response_len,
                ),
            )
            print(response)
            # openai api returns \n\n as a delimiter for messages
            yield f"data: {response.model_dump_json()}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate())

    # 生成回复
    response_str = infer_engine.chat(
        messages,
        None,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        session_id,
    )

    # 非流式响应
    response = ChatCompletion(
        id=session_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    content=response_str,
                    references=references,
                    role="assistant",
                ),
            ),
        ],
        created=time.time(),
        usage=CompletionUsage(
            prompt_tokens=content_len,
            completion_tokens=len(response_str),
            total_tokens=content_len + len(response_str),
        ),
    )
    return response


# 将请求体作为 JSON 读取
# 在函数内部，你可以直接访问模型对象的所有属性
# http://127.0.0.1:8000/docs
@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat(request: ChatRequest) -> StreamingResponse | ChatCompletion:
    print("request: ", request)

    messages = request.messages
    print("messages: ", messages)

    if not messages or len(messages) == 0:
        raise HTTPException(status_code=400, detail="No messages provided")

    role = messages[-1].get("role", "")
    if role not in ["user", "assistant"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    content = messages[-1].get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="content is empty")

    return rag_generate(
        messages,
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

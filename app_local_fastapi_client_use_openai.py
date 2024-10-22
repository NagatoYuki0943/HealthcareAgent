# https://internlm.intern-ai.org.cn/api/document
# https://platform.moonshot.cn/docs/api/chat
import os
from openai import OpenAI, Stream
from openai.types.chat.chat_completion import ChatCompletion


"""
设置临时变量

linux:
    export API_KEY="your token"

powershell:
    $env:API_KEY = "your token"
"""
api_key = os.getenv("API_KEY", "I AM AN API_KEY")


client = OpenAI(
    api_key=api_key,  # 此处传token，不带Bearer
    base_url="http://localhost:8000/v1/",
)


messages = [{"role": "user", "content": "维生素C有什么作用?"}]


chat_completions: ChatCompletion = client.chat.completions.create(
    messages=messages,
    model="internlm/internlm2_5-7b-chat",
    max_tokens=1024,
    n=1,  # 为每条输入消息生成多少个结果，默认为 1
    presence_penalty=0.0,  # 存在惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇是否出现在文本中来进行惩罚，增加模型讨论新话题的可能性
    frequency_penalty=0.0,  # 频率惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇在文本中现有的频率来进行惩罚，减少模型一字不差重复同样话语的可能性
    stream=False,
    temperature=0.8,
    top_p=0.8,
)
print(chat_completions)
# ChatCompletion(
#     id=15246409379408058547,
#     choices=[
#         Choice(
#             finish_reason='stop',
#             index=0,
#             logprobs=None,
#             message=ChatCompletionMessage(
#                 content='[ 2 84 42 60 92  0 53 58 44 24]',
#                 refusal=False,
#                 role='assistant',
#                 function_call=None,
#                 tool_calls=None
#             )
#         )
#     ],
#     created=1727509965.4057014,
#     model=None,
#     object='chat.completion',
#     service_tier=None,
#     system_fingerprint=None,
#     usage=CompletionUsage(completion_tokens=31, prompt_tokens=5, total_tokens=36, completion_tokens_details=None)
# )


for choice in chat_completions.choices:
    print(choice)
    # Choice(
    #     finish_reason='stop',
    #     index=0,
    #     logprobs=None,
    #     message=ChatCompletionMessage(
    #         content='[ 2 84 42 60 92  0 53 58 44 24]',
    #         refusal=False,
    #         role='assistant',
    #         function_call=None,
    #         tool_calls=None
    #     )
    # )
    print(choice.message.content)
    # [ 2 84 42 60 92  0 53 58 44 24]


chat_completions: Stream = client.chat.completions.create(
    messages=messages,
    # model="moonshot-v1-8k",
    model="internlm/internlm2_5-7b-chat",
    max_tokens=1024,
    n=1,  # 为每条输入消息生成多少个结果，默认为 1
    presence_penalty=0.0,  # 存在惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇是否出现在文本中来进行惩罚，增加模型讨论新话题的可能性
    frequency_penalty=0.0,  # 频率惩罚，介于-2.0到2.0之间的数字。正值会根据新生成的词汇在文本中现有的频率来进行惩罚，减少模型一字不差重复同样话语的可能性
    stream=True,
    temperature=0.8,
    top_p=0.8,
)
print(chat_completions)
# <openai.Stream object at 0x000002294D483AF0>


responses = []
# print("response: ", end="", flush=True)
for idx, chunk in enumerate(chat_completions):
    print(chunk)
    # ChatCompletionChunk(
    #     id=15531134710371620732,
    #     choices=[
    #         Choice(
    #             delta=ChoiceDelta(content=']', function_call=None, refusal=False, role='assistant', tool_calls=None),
    #             finish_reason=None,
    #             index=0,
    #             logprobs=None
    #         )
    #     ],
    #     created=1727510160.8153822,
    #     model=None,
    #     object='chat.completion.chunk',
    #     service_tier=None,
    #     system_fingerprint=None,
    #     usage=CompletionUsage(completion_tokens=31, prompt_tokens=5, total_tokens=36, completion_tokens_details=None)
    # )
    # ChatCompletionChunk(
    #     id=15531134710371620732,
    #     choices=[
    #         Choice(
    #             delta=ChoiceDelta(content=None, function_call=None, refusal=False, role=None, tool_calls=None),
    #             finish_reason='stop',
    #             index=0,
    #             logprobs=None
    #         )
    #     ],
    #     created=1727510160.8153822,
    #     model=None,
    #     object='chat.completion.chunk',
    #     service_tier=None,
    #     system_fingerprint=None,
    #     usage=CompletionUsage(completion_tokens=31, prompt_tokens=5, total_tokens=36, completion_tokens_details=None)
    # )

    chunk_message = chunk.choices[0].delta
    if not chunk_message.content:
        continue
    content = chunk_message.content

    # print(content, end="", flush=True)
    responses.append(content)

print("\ncomplete response: ", "".join(responses))

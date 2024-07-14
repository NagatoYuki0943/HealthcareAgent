import os
import base64
from io import BytesIO
import requests
from PIL import Image
import uuid
from typing import Literal, Sequence
import random
from loguru import logger


# 可以传递一个提示语句 + 一张或者多张 PIL.Image.Image 的图片
# 或者传递一个提示语句 + 一张或者多张图片的url地址或者本地地址,后面 tuple 中的第二个 str 或者 list[str] 指的就是图片地址
VLQueryType = tuple[str, Image.Image] | tuple[str, list[Image.Image]] | tuple[str, str] | tuple[str, list[str]]


def random_uuid(dtype: Literal['int', 'str', 'bytes', 'time'] = 'int') -> int | str | bytes:
    """生成随机uuid
    reference: https://github.com/vllm-project/vllm/blob/main/vllm/utils.py
    """
    assert dtype in ['int', 'str', 'bytes', 'time'], f"unsupported dtype: {dtype}, should be in ['int', 'str', 'bytes', 'time']"

    # uuid4: 由伪随机数得到，有一定的重复概率，该概率可以计算出来。
    uid = uuid.uuid4()
    if dtype == 'int':
        return uid.int
    elif dtype == 'str':
        return uid.hex
    elif dtype == 'bytes':
        return uid.bytes
    else:
        return uid.time


def random_uuid_int() -> int:
    """random_uuid 生成的 int uuid 会超出int64的范围,lmdeploy使用会报错"""
    return random.getrandbits(64)


# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/utils.py#L11-L38
def encode_image_base64(image: str | Image.Image) -> str:
    """encode raw date to base64 format."""
    res = ''
    if isinstance(image, str):
        url_or_path = image
        if url_or_path.startswith('http'):
            FETCH_TIMEOUT = int(os.environ.get('LMDEPLOY_FETCH_TIMEOUT', 10))
            headers = {
                'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            try:
                response = requests.get(url_or_path,
                                        headers=headers,
                                        timeout=FETCH_TIMEOUT)
                response.raise_for_status()
                res = base64.b64encode(response.content).decode('utf-8')
            except Exception:
                pass
        elif os.path.exists(url_or_path):
            with open(url_or_path, 'rb') as image_file:
                res = base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        res = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return res



# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/templates.py#L25-L69
def convert_to_openai_history(
    history: Sequence[Sequence],
    query: str | VLQueryType | None,
) -> list:
    """
    将历史记录转换为openai格式

    Args:
        history (Sequence[Sequence]):聊天历史记录
            example: [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        query (str | VLQueryType | None): 查询语句

    Returns:
        list: a chat history in OpenAI format or a list of chat history.
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                },
                {
                    "role": "assistant",
                    "content": "The capital of France is Paris."
                },
                {
                    "role": "user",
                    "content": "Thanks!"
                },
                {
                    "role": "assistant",
                    "content": "You are welcome."
                }
            ]
    """
    # 将历史记录转换为openai格式
    messages = []
    for prompt, response in history:
        if isinstance(prompt, str):
            content = [{
                'type': 'text',
                'text': prompt,
            }]
        else:
            prompt, images = prompt
            content = [{
                'type': 'text',
                'text': prompt,
            }]
            # image: PIL.Image.Image
            images = images if isinstance(images, (list, tuple)) else [images]
            for image in images:
                # 'image_url': means url or local path to image.
                # 'image_data': means PIL.Image.Image object.
                if isinstance(image, str):
                    image_base64_data = encode_image_base64(image)
                    if image_base64_data == '':
                        logger.error(f'failed to load file {image}')
                        continue
                    item = {
                        'type': 'image_url',
                        'image_url': {
                            'url':
                            f'data:image/jpeg;base64,{image_base64_data}'
                        }
                    }
                elif isinstance(image, Image.Image):
                    item = {
                        'type': 'image_data',
                        'image_data': {
                            'data': image
                        }
                    }
                else:
                    raise ValueError(
                        'image should be a str(url/path) or PIL.Image.Image')

                content.append(item)
        messages.append({
            "role": "user",
            "content": content
        })

        if response is not None:
            messages.append({
                "role": "assistant",
                # assistant 的回答必须是字符串,不能是数组
                "content": response
            })

    # 添加当前的query
    if query is not None:
        if isinstance(query, str):
            content = [{
                'type': 'text',
                'text': query,
            }]
        else:
            query, images = query
            content = [{
                'type': 'text',
                'text': query,
            }]
            images = images if isinstance(images, (list, tuple)) else [images]
            for image in images:
                # 'image_url': means url or local path to image.
                # 'image_data': means PIL.Image.Image object.
                if isinstance(image, str):
                    image_base64_data = encode_image_base64(image)
                    if image_base64_data == '':
                        logger.error(f'failed to load file {image}')
                        continue
                    item = {
                        'type': 'image_url',
                        'image_url': {
                            'url':
                            f'data:image/jpeg;base64,{image_base64_data}'
                        }
                    }
                elif isinstance(image, Image.Image):
                    item = {
                        'type': 'image_data',
                        'image_data': {
                            'data': image
                        }
                    }
                else:
                    raise ValueError(
                        'image should be a str(url/path) or PIL.Image.Image')

                content.append(item)
        messages.append({
            "role": "user",
            "content": content
        })

    return messages


if __name__ == '__main__':
    print(random_uuid('int'))
    print(random_uuid_int())

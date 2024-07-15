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
# 或者传递一个提示语句 + 一张或者多张图片的url地址或者本地地址, 后面 tuple 中的第二个 str 或者 list[str] 指的就是图片地址
VLQueryType = tuple[str, Image.Image] | tuple[str, list[Image.Image]] | tuple[str, str] | tuple[str, list[str]]


# gradio一轮的对话格式
# [问, 答]
# [(图片, alt_text), None]
GradioChat1TuneType = list[str, str] | list[tuple[Image.Image, str], None]


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


# https://platform.openai.com/docs/guides/text-generation
# https://platform.openai.com/docs/guides/vision
# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/templates.py#L25-L69
def convert_to_openai_history(
    history: Sequence[Sequence],
    query: str | VLQueryType | None = None,
) -> list:
    """
    将历史记录转换为openai格式

    Args:
        history (Sequence[Sequence]):聊天历史记录
            example: [
                ['What is the capital of France?', 'The capital of France is Paris.'],
                [('What is in the image?', Image1), 'There is a dog in the image.'],
            ]
        query (str | VLQueryType | None): 查询语句,str或者dict,图片支持PIL.Image.Image或者本地文件路径/url, de
            example: 你是谁?
                     ('What is in the image?', Image1)
                     ('How dare you!', [Image2, Image3])

    Returns:
        list: a chat history in OpenAI format or a list of chat history.
            [
                {'role': 'user', 'content': [{'type': 'text', 'text': 'What is the capital of France?'}]},
                {'role': 'assistant', 'content': 'The capital of France is Paris.'},
                {'role': 'user', 'content': [
                        {'type': 'text', 'text': 'What is in the image?'},
                        {'type': 'image_data', 'image_data': {'data': Image1}}
                    ]
                },
                {'role': 'assistant', 'content': 'There is a dog in the image.'},
                {'role': 'user', 'content': [
                        {'type': 'text', 'text': 'How dare you!'},
                        {'type': 'image_data', 'image_data': {'data': Image2}},
                        {'type': 'image_data', 'image_data': {'data': Image3}}
                    ]
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


def history_test(
) -> list:
    history1 = [
        ['text1', '[91 24 10 19 73]'],
        ['text2', '[85 98 95  3 25]'],
        ['text3', '[58 60 35 97 39]'],
    ]
    query = 'text4'
    messages1 = convert_to_openai_history(history1, query)
    print(messages1)
    print("\n")
    [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'text1'}]},
        {'role': 'assistant', 'content': '[91 24 10 19 73]'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'text2'}]},
        {'role': 'assistant', 'content': '[85 98 95  3 25]'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'text3'}]},
        {'role': 'assistant', 'content': '[58 60 35 97 39]'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'text4'}]}
    ]


    history2 = [
        ['你是谁', '[47  5 79  7 79]'],
        [('what is this?', Image.open('../images/logo.png')), '[58 71 49 87 10]'],
        [('这2张图片展示的什么内容?', [Image.open('../images/openxlab.png'), Image.open('../images/openxlab_model.jpg')]), '[29 86 41 26 84]'],
    ]
    query = ('how dare you!', Image.open('../images/openxlab.png'))

    messages2 = convert_to_openai_history(history2, query)
    print(messages2)
    [
        {'role': 'user', 'content': [{'type': 'text', 'text': '你是谁'}]},
        {'role': 'assistant', 'content': '[47  5 79  7 79]'},
        {'role': 'user', 'content': [
                {'type': 'text', 'text': 'what is this?'},
                {'type': 'image_data', 'image_data': {'data':' <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1792x871 at 0x290FCF14C90>'}}
            ]
        },
        {'role': 'assistant', 'content': '[58 71 49 87 10]'},
        {'role': 'user', 'content': [
                {'type': 'text', 'text': '这2张图片展示的什么内容?'},
                {'type': 'image_data', 'image_data': {'data': '<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1580x1119 at 0x290FCF7D650>'}},
                {'type': 'image_data', 'image_data': {'data': '<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1019x716 at 0x290FCF7E710>'}}
            ]
        },
        {'role': 'assistant', 'content': '[29 86 41 26 84]'},
        {'role': 'user', 'content': [
                {'type': 'text', 'text': 'how dare you!'},
                {'type': 'image_data', 'image_data': {'data': '<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1580x1119 at 0x290FCF7EBD0>'}}
            ]
        }
    ]


# https://platform.openai.com/docs/guides/text-generation
# https://platform.openai.com/docs/guides/vision
# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/templates.py#L25-L69
def convert_to_openai_history_new(
    history: Sequence[GradioChat1TuneType],
    query: str | dict | None = None,
) -> list:
    """
    将历史记录转换为openai格式

    Args:
        history (Sequence[GradioChat1TuneType]):聊天历史记录
            example: [
                ['What is the capital of France?', 'The capital of France is Paris.'],
                [(Image1, None), None],
                ['What is in the image?', 'There is a dog in the image.'],
            ]
        query (str | dict | None): 查询语句,str或者dict,图片支持PIL.Image.Image或者本地文件路径/url
            example: 你是谁?
                    {'text': 'How dare you!', 'files': [Image2, Image3]}

    Returns:
        list: a chat history in OpenAI format or a list of chat history.
            [
                {'role': 'user', 'content': [{'type': 'text', 'text': 'What is the capital of France?'}]},
                {'role': 'assistant', 'content': 'The capital of France is Paris.'},
                {'role': 'user', 'content': [{'type': 'text', 'text': ''}, {'type': 'image_data', 'image_data': {'data': Image1}}]},
                {'role': 'user', 'content': [{'type': 'text', 'text': 'What is in the image?'}]},
                {'role': 'assistant', 'content': 'There is a dog in the image.'},
                {'role': 'user', 'content': [{'type': 'text', 'text': 'How dare you!'}, {'type': 'image_data', 'image_data': {'data': Image2}}, {'type': 'image_data', 'image_data': {'data': Image3}}]}
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
            content = [{
                'type': 'text',
                'text': f'',  # text占位符
            }]
            image = prompt[0]
            # 'image_url': means url or local path to image.
            # 'image_data': means PIL.Image.Image object.
            if isinstance(image, str):
                image_base64_data = encode_image_base64(image)
                if image_base64_data == '':
                    logger.error(f'failed to load file {image}')
                    raise ValueError(f'failed to load file {image}')
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
            query_text, images = query['text'], query['files']
            content = [{
                'type': 'text',
                'text': f'{query_text}',
            }]
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


def new_history_test(
) -> list:
    history1 = [
        ['text1', '[91 24 10 19 73]'],
        ['text2', '[85 98 95  3 25]'],
        ['text3', '[58 60 35 97 39]'],
    ]
    query = 'text4'
    messages1 = convert_to_openai_history_new(history1, query)
    print(messages1)
    print("\n")
    [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'text1'}]},
        {'role': 'assistant', 'content': '[91 24 10 19 73]'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'text2'}]},
        {'role': 'assistant', 'content': '[85 98 95  3 25]'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'text3'}]},
        {'role': 'assistant', 'content': '[58 60 35 97 39]'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'text4'}]}
    ]


    history2 = [
        ['你是谁', '[47  5 79  7 79]'],
        [(Image.open('../images/logo.png'), None), None],
        ['what is this?', '[58 71 49 87 10]'],
        [(Image.open('../images/openxlab.png'), None), None],
        [(Image.open('../images/openxlab_model.jpg'), None), None],
        ['这2张图片展示的什么内容?', '[29 86 41 26 84]'],
    ]
    query = {
        'text': 'how dare you!',
        'files': [
            Image.open('../images/openxlab.png'),
            Image.open('../images/openxlab_model.jpg')
        ]
    }

    messages2 = convert_to_openai_history_new(history2, query)
    print(messages2)
    [
        {'role': 'user', 'content': [{'type': 'text', 'text': '你是谁'}]},
        {'role': 'assistant', 'content': '[47  5 79  7 79]'},
        {'role': 'user', 'content': [{'type': 'text', 'text': ''}, {'type': 'image_data', 'image_data': {'data': '<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1792x871 at 0x23B1E20EB90>'}}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'what is this?'}]},
        {'role': 'assistant', 'content': '[58 71 49 87 10]'},
        {'role': 'user', 'content': [{'type': 'text', 'text': ''}, {'type': 'image_data', 'image_data': {'data': '<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1580x1119 at 0x23B1E288510>'}}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': ''}, {'type': 'image_data', 'image_data': {'data': '<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1019x716 at 0x23B1E288750>'}}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': '这2张图片展示的什么内容?'}]},
        {'role': 'assistant', 'content': '[29 86 41 26 84]'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'how dare you!'}, {'type': 'image_data', 'image_data': {'data': '<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1580x1119 at 0x23B1E288C10>'}}, {'type': 'image_data', 'image_data': {'data': '<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1019x716 at 0x23B1E288E90>'}}]}
    ]


if __name__ == '__main__':
    history_test()
    print("\n\n")
    # new_history_test()

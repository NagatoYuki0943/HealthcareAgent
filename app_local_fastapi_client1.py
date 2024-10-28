# copy from https://github.com/NagatoYuki0943/fastapi-learn/blob/main/xx_stream/client.py
import os
import requests
import json
from image2base64 import convert_image_to_base64


URL = "http://localhost:8000/v1/chat/completions"

api_key = os.getenv("API_KEY", "I AM AN API_KEY")

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {api_key}",
}


# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/api_client.py
def requests_chat(data: dict):
    stream: bool = data["stream"]

    response: requests.Response = requests.post(
        URL, json=data, headers=headers, timeout=60, stream=stream
    )
    response.raise_for_status()

    if not stream:
        yield response.json()
    else:
        chunk: bytes
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"\n\n"
        ):
            if chunk:
                decoded: str = chunk.decode("utf-8")
                if decoded.startswith("data: "):
                    decoded = decoded[6:]
                    if decoded.strip() == "[DONE]":
                        continue
                    yield json.loads(decoded)


if __name__ == "__main__":
    image_base64 = convert_image_to_base64("InternLM2-7b-1.png")
    print(image_base64)

    data = {
        "model": "ocr_chat",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "图片中有什么内容？"},
            {"type": "image_url", "image_url": {"url": image_base64}},
            ]}],
        "max_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 50,
        "stream": False,
    }
    data_stream = data.copy()
    data_stream["stream"] = True

    for output in requests_chat(data):
        print(output)

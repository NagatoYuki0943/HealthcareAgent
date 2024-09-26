# copy from https://github.com/NagatoYuki0943/fastapi-learn/blob/main/xx_stream/client.py

import requests
import json


URL = "http://localhost:8000/chat"

api_key = "I AM AN API_KEY"

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {api_key}",
}


# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/api_client.py
def requests_chat(data: dict):
    stream = data["stream"]

    response: requests.Response = requests.post(
        URL, json=data, headers=headers, timeout=60, stream=stream
    )
    if not stream:
        yield response.json()
    else:
        chunk: bytes
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"\n\n"
        ):
            if chunk:
                decoded: str = chunk.decode("utf-8")
                yield json.loads(decoded)


if __name__ == "__main__":
    data = {
        "messages": [{"role": "user", "content": "维生素E有什么作用"}],
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

    for output in requests_chat(data_stream):
        print(output)

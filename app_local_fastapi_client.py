# copy from https://github.com/NagatoYuki0943/fastapi-learn/blob/main/xx_stream/client.py

import requests
import json


URL = "http://localhost:8000/chat"


def requests_chat(data: dict):
    stream = data["stream"]
    response: requests.Response = requests.post(
        URL, json=data, timeout=60, stream=stream
    )
    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\n"
    ):
        if chunk:
            decoded = chunk.decode("utf-8")
            output = json.loads(decoded)
            yield output


if __name__ == "__main__":
    data = {
        "messages": [{"content": "维生素E有什么作用", "role": "user"}],
        "max_new_tokens": 1024,
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

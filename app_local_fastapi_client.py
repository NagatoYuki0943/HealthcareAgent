import requests


url = "http://localhost:8000/chat"


def requests_chat(data: dict):
    data["stream"] = False
    response: requests.Response = requests.post(url, json=data, timeout=60)
    if response.status_code!= 200:
        print(f"Error: {response.status_code} {response.text}")
    return response.json()


def requests_stream_chat(data: dict):
    data["stream"] = True
    response: requests.Response = requests.post(url, json=data, timeout=60, stream=True)
    if response.status_code!= 200:
        raise Exception(f"Error: {response.status_code} {response.text}")
    for line in response.iter_lines():
        if line:
            chunk = line.decode('utf-8')
            if chunk.startswith('data:') and chunk != 'data: [DONE]':
                delta = chunk.split('data: ')[1]
                print(delta)


if __name__ == '__main__':
    data = {
        "messages": [
            {
                "content": "维生素E有什么作用?",
                "role": "user"
            }
        ],
        "max_new_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 50,
        "stream": False,
    }

    ret = requests_chat(data)
    print(ret)
    print()

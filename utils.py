import os
import re
import uuid
from typing import Literal
from langchain_core.documents import Document


def is_used_rag(
    reject_answer: str,
    history: list,
) -> bool:
    """是否使用过rag"""
    if len(history) == 0:
        return False
    prompts, responses = zip(*history)
    # 去重回答
    responses = list(set(responses))
    # 经过去重之后只有一种回答,并且回答是拒绝回答说明没使用rag
    if len(responses) == 1 and responses[0] == reject_answer:
        return False
    else:
        return True


def get_filename(path: str):
    """
    './data\\FM docs 2024.3\\JOM_1998_13_4_06_The_Application_of_the_Hardin_Jones-Pauling-.pdf'
    ->
    ('./data\\FM docs 2024.3',
    'JOM_1998_13_4_06_The_Application_of_the_Hardin_Jones-Pauling-.pdf')
    """
    basepath, filename = os.path.split(path)
    return filename


def format_documents(documents: list[Document]) -> str:
    return "\n".join([doc.page_content for doc in documents])


def format_references(references: list[str]) -> str:
    if len(references) == 0:
        return "\n*no reference.*"
    else:
        references = [f"*{reference}*" for reference in references]
        references_str = "\n".join(references)
        return f"\nreferences: \n{references_str}"


def remove_history_references(history: list) -> list:
    new_history = []
    for prompt, response in history:
        # 按照参考说明拆分,只保留拆分前的内容
        response_no_reference = re.split(r"\n\*no reference\.\*|\nreferences:", response)[0]
        new_history.append([prompt, response_no_reference])
    return new_history


def download_dataset(
    dataset_repo='NagatoYuki0943/FMdocs',
    target_path='./data/'
):
    import os
    import openxlab
    from openxlab.dataset import get

    print("start download dataset")
    access_key = os.getenv("OPENXLAB_AK", "")
    secret_key = os.getenv("OPENXLAB_SK", "")
    print(f"access_key = {access_key}")
    print(f"secret_key = {secret_key}")
    openxlab.login(ak=access_key, sk=secret_key)
    get(dataset_repo=dataset_repo, target_path=target_path) # 数据集下载
    print("finish download dataset")


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


if __name__ == "__main__":
    download_dataset()

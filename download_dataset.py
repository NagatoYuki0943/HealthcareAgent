import os
from utils import download_dataset


# 获取环境变量
access_key = os.getenv("OPENXLAB_AK", "")
secret_key = os.getenv("OPENXLAB_SK", "")
print(f"access_key = {access_key}")
print(f"secret_key = {secret_key}")

DATA_PATH: str = "./data"

download_dataset(
    dataset_repo = 'NagatoYuki0943/FMdocs',
    target_path = DATA_PATH,
    access_key = access_key,
    secret_key = secret_key
)

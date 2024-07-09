import os
from utils import download_openxlab_dataset


# 获取环境变量
openxlab_access_key = os.getenv("OPENXLAB_AK", "")
openxlab_secret_key = os.getenv("OPENXLAB_SK", "")
print(f"{openxlab_access_key = }")
print(f"{openxlab_secret_key = }")

DATA_PATH: str = "./data"

download_openxlab_dataset(
    dataset_repo = 'NagatoYuki0943/PigDiseaseDataset',
    target_path = DATA_PATH,
    access_key = openxlab_access_key,
    secret_key = openxlab_secret_key
)

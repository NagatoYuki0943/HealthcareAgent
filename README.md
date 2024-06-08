# HealthcareAgent

在医学领域，准确、及时的信息对于诊断和治疗至关重要。为了提高医学问题回答的准确率，我们开发了一个基于RAG（Retrieval-Augmented Generation）模型的医学问题回答助手。该助手结合了先进的检索技术和生成技术，通过检索向量数据库来获取相关信息，从而生成精确、可靠的医学回答。

## 使用步骤

### clone仓库

将项目clone到本地

```sh
git clone https://github.com/NagatoYuki0943/HealthcareAgent.git
```

### 环境配置

建议创建一个虚拟环境，可以使用conda

```sh
conda create --name agent python=3.10
```

安装 `requirements.txt` 中的 python package 依赖。

```sh
cd HealthcareAgent
pip install -r requirements.txt
```

后续下载模型还需要安装 Git 和 Git lfs

```sh
# linux
sudo apt install git
sudo apt install git-lfs

# windows
# https://git-scm.com/downloads 下载安装 git
git lfs install
```

### 准备数据集

在 `HealthcareAgent`  目录建立一个 `data` 目录，可以建立多个子文件夹分类存放。

```sh
|-- HealthcareAgent/
    |-- data # 数据集放在这里，可以建立多个子文件夹分类存放
```

在 `app.py` 和 `app_local.py` 中有如下参数，可以选择需要的文件类型。可以把不需要的类型从元组中删除。

```python
ALLOW_SUFFIX: tuple[str] = (".txt", ".md", ".docx", ".doc", ".pdf")
```

###  下载模型

本项目部署需要一个语言模型，RAG检索需要一个 Emebdding 和 Reranker 模型，需要下载到本地

由于使用的 Embedding 和 Reranker 模型需要同意协议才能下载，所以需要登陆 [Huggingface](https://huggingface.co/) ，进入两个模型的页面（ [bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1) 和 [bce-reranker-base_v1](https://huggingface.co/maidalun1020/bce-reranker-base_v1)）点击同意协议，之后在 [tokens](https://huggingface.co/settings/tokens) 界面获取 token，放入环境变量中。也可以使用 [modelscope](https://www.modelscope.cn/) 下载模型，不需要 token。

运行如下命令下载模型。

```sh
# linux:
export HF_TOKEN="your token"

# powershell:
$env:HF_TOKEN="your token"

python download_hf.py
```

```sh
|-- HealthcareAgent/
    |-- data # 数据集放在这里，可以建立多个子文件夹分类存放
    |-- models # 模型会放在这里，可以自己将模型放到这个目录
```

### 启动应用

项目启动文件为 `app.py` 和 `app_local.py`，两者区别如下：

1. `app.py` 文件中会自动下载所需要的模型和数据集，而 `app_local.py` 不会下载，需要自己提前下载。
2. `app.py` 拥有医疗问答和化验报告分析两个功能，而 `app_local.py` 只有医疗问答功能。
3. `app.py` 默认使用 lmdeploy 进行推理，`app_local.py` 默认使用 transformers 进行推理。

####  `app.py` 启动

由于 `app.py` 会自动下载我们的私有数据，因此需要修改对应的代码，注释或者删除下载数据集的代码

```diff
- openxlab_access_key = os.getenv("OPENXLAB_AK", "")
- openxlab_secret_key = os.getenv("OPENXLAB_SK", "")
- print(f"{openxlab_access_key = }")
- print(f"{openxlab_secret_key = }")

...

# 下载数据集,不会重复下载
- download_openxlab_dataset(
-     dataset_repo = 'NagatoYuki0943/FMdocs',
-     target_path = DATA_PATH,
-     access_key = openxlab_access_key,
-     secret_key = openxlab_secret_key
- )
```

每次 `app.py` 启动时会尝试下载模型（已经下载好的不会重复下载），如果已经配置好本地模型，可以禁用自动下载

```diff
- hf_token = os.getenv("HF_TOKEN", "")
- print(f"{hf_token = }")

- snapshot_download(
-     repo_id = "maidalun1020/bce-embedding-base_v1",
-     local_dir = EMBEDDING_MODEL_PATH,
-     max_workers = 8,
-     token = hf_token
- )
- snapshot_download(
-     repo_id = "maidalun1020/bce-reranker-base_v1",
-     local_dir = RERANKER_MODEL_PATH,
-     max_workers = 8,
-     token = hf_token
- )

- os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-7b {MODEL_PATH}')
- os.system(f'cd {MODEL_PATH} && git lfs pull')
```

由于化验报告分析需要调用第三方接口，因此需要获取对应的 token

获取腾讯 OCR 模型的密钥：https://console.cloud.tencent.com/cam/capi

获取文心一言的密钥：https://developer.baidu.com/article/detail.html?id=1089328

获取对应 api 后要设置对应环境变量后再启动

```sh
# linux:
export HF_TOKEN="your token" # 如果已经下载好模型就不需要这个token
export OCR_SECRET_ID="OCR_SECRET_ID"
export OCR_SECRET_KEY="OCR_SECRET_KEY"
export ERNIE_API_KEY="ERNIE_API_KEY"
export ERNIE_SECRET_KEY="ERNIE_SECRET_KEY"

# powershell:
$env:HF_TOKEN="your token" # 如果已经下载好模型就不需要这个token
$env:OCR_SECRET_ID="OCR_SECRET_ID"
$env:OCR_SECRET_KEY="OCR_SECRET_KEY"
$env:ERNIE_API_KEY="ERNIE_API_KEY"
$env:ERNIE_SECRET_KEY="ERNIE_SECRET_KEY"

python app.py
```

#### `app_local.py` 启动

```sh
python app_local.py
```

### 其他功能

`download_dataset.py` 专门下载 openxlab 上的数据集，需要设置环境变量 `OPENXLAB_AK` 和 `OPENXLAB_SK`。

```sh
python download_dataset.py
```

`vector_database_create.py` 脚本用来重新建立向量数据库，在修改自己的数据之后需要执行。

```sh
python vector_database_create.py
```

 `app.py` 和 `app_local.py` 更换推理后端，transformers 兼容性好，lmdeploy 在 windows 兼容性可能有问题。

更换方式只需要修改一行代码即可

```python
backend = 'transformers', # transformers, lmdeploy, api 将这里指定为不同后端的名称即可
```

```python
from infer_engine import InferEngine, TransformersConfig, LmdeployConfig

TRANSFORMERS_CONFIG = TransformersConfig(
    pretrained_model_name_or_path = PRETRAINED_MODEL_NAME_OR_PATH,
    adapter_path = ADAPTER_PATH,
    load_in_8bit = LOAD_IN_8BIT,
    load_in_4bit = LOAD_IN_4BIT,
    model_name = 'internlm2',
    system_prompt = SYSTEM_PROMPT
)

LMDEPLOY_CONFIG = LmdeployConfig(
    model_path = PRETRAINED_MODEL_NAME_OR_PATH,
    backend = 'turbomind',
    model_name = 'internlm2',
    model_format = 'hf',
    cache_max_entry_count = 0.5,    # 调整 KV Cache 的占用比例为0.5
    quant_policy = 0,               # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
    system_prompt = SYSTEM_PROMPT,
    deploy_method = 'local',
    log_level = 'ERROR'
)

# 载入模型
infer_engine = InferEngine(
    backend = 'transformers', # transformers, lmdeploy, api 将这里指定为不同后端的名称即可
    transformers_config = TRANSFORMERS_CONFIG,
    lmdeploy_config = LMDEPLOY_CONFIG
)
```

# 说明

模型运行步骤：

1. 对输出判断，是否为无效字符。

2. 检索数据库，使用数据库检索进行初步筛选，然后使用重排序模型过滤。

   如果检索到数据，就将检索数据和问题经过格式化后一起提供给模型，得到输出；如果没有检索到数据，就只给模型提供问题，让模型输出。

2. 将问题和检索数据以及历史记录一起传递给模型得到回答。

细节说明：

1. 每次问答都会检索数据库。

2. 在使用 RAG 检索时，给模型的提示词时包含了检索数据和问题的提示词，不过最终呈现给用户时，只将用户的问题和模型的回答呈现给用户。

3. 历史记录同上，也就是说历史记录里面只包含用户的问题和模型的回答，没有保存历史的 RAG 检索数据。

   这样做的原因有2点：一是将检索数据呈现给用户体验不好。二是这些页数也会占用模型的 token 数量，模型的 token 是有上限的，过长会影响模型输出效果。

4. 在每次对话时，如果有检索到信息，会将检索的文档名字返回给用户，没有检索到时也有相应的提示，这些参考信息也是会保存在历史记录中。

   不过需要注意的是，在将过往历史记录传递给模型的时候，是会将历史记录中的参考文献删除的。

   这样做的原因有2点：一是在使用过程中发现，在有历史记录且历史记录的对话没使用检索信息，新的对话也没有使用检索的时候，有时在模型输出的结尾会打印两次 `no reference.`，我排查后发现第一次的打印是模型自己输出的，也就是模型根据历史记录学会了在合适的情况下添加`no reference.`，而这不是我们想要的结果。是这些参考文档也会占用模型的 token 数量，模型的 token 是有上限的，过长会影响模型输出效果。

# TODO

- [x] 返回参考文档
- [x] 当前使用 transformers 库进行推理，速度较慢，可以换成 lmdeploy 库进行加速
- [x] 当查询库中没有对应数据时说明找不到内容
- [x] 加上重排序模型




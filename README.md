# HealthcareAgent

在医学领域，准确、及时的信息对于诊断和治疗至关重要。为了提高医学问题回答的准确率，我们开发了一个基于RAG（Retrieval-Augmented Generation）模型的医学问题回答助手。该助手结合了先进的检索技术和生成技术，通过检索向量数据库来获取相关信息，从而生成精确、可靠的医学回答。

## 使用步骤

1. 安装依赖

   安装 `requirements.txt` 中的 python package 依赖。

   ```sh
   pip install -r requirements.txt
   ```

   安装 git 和 git-lfs 用来下载模型。

   ```sh
   # linux
   sudo apt install git
   sudo apt install git-lfs
   
   # windows
   # https://git-scm.com/downloads 下载安装 git
   git lfs install
   ```

2. 建立一个 `data` 目录，然后将自己的数据放入这个目录中，可以在 `data` 中新建目录或者直接将所有文件让如 `data` 目录中。

在 `app.py` 和 `app_local.py` 中有如下参数，可以选择需要的文件类型。

```python
ALLOW_SUFFIX: tuple[str] = (".txt", ".md", ".docx", ".doc", ".pdf")
```

3. 下载模型（可选）

运行如下命令下载模型。

由于使用的 Embedding 和 Reranker 模型需要同意协议才能下载，所以需要登陆 [Huggingface](https://huggingface.co/) ，进入两个模型的页面（ [bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1) 和 [bce-reranker-base_v1](https://huggingface.co/maidalun1020/bce-reranker-base_v1)）点击同意协议，之后在 [tokens](https://huggingface.co/settings/tokens) 界面获取token，放入环境变量中。也可以使用 [modelscope](https://www.modelscope.cn/) 下载模型，不需要 token。

```sh
# linux:
export HF_TOKEN="your token"

# powershell:
$env:HF_TOKEN="your token"

python download_hf.py
```

4. 运行

使用如下命令启动应用。

```sh
python app.py # 使用 lmdeploy 部署，会自动下载模型和数据集，同样需要设置 HF_TOKEN

python app_local.py # 使用 transformers 部署，支持 windows，不会自动下载模型和数据集，需要步骤3
```

5. 远程连接

```sh
ssh -CNg -L 7860:127.0.0.1:7860 root@ssh.intern-ai.org.cn -p 你自己的端口号
```

6. 其他

`download_dataset.py` 专门下载 openxlab 上的数据集

`vector_database_create.py` 用来重新创建向量数据库

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




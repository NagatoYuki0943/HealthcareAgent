# medical-rag

使用rag对医学数据进行检索

## 使用步骤

1. 安装依赖，安装 `requirements.txt` 中的依赖，可能不全，如果有需要按照的包里面没包含请告诉我

2. 下载模型

运行 `download_hf.py` 文件下载向量化模型和 internlm2 模型

也可以自己下载放到 models 目录下

3. 建立一个 `data` 目录，然后将群里的压缩包解压到data内，最终路径为 `data/FM docs 2024.3/*.pdf`

4. 建立向量数据库

运行`create_db.py`或者 `create_db.ipynb`建立数据库

5. 运行

`run_langchain.py` 是命令行运行模型，功能不全

`run_langchain_gradio.py`是使用 langchain 运行模型，功能不全

`run_custom_gradio.py` 会议中展示的，功能最全

`app.py` 内容和 `run_custom_gradio.py` 相同，不同点是会下载数据集，创建数据库，下载模型并运行，并使用了 lmdeploy 加速

## 注意

当前模型在推理时只有第一次对话会进行rag检索，后续聊天是在之前基础上做的。

## TODO

- [x] 返回参考文档
- [x] 当前使用 transformers 库进行推理，速度较慢，可以换成 lmdeploy 库进行加速
- [ ] 当查询库中没有对应数据时说明找不到内容
- [ ] 加上重排序模型




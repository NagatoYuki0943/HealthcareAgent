#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   llm.py
@Time    :   2023/10/16 18:53:26
@Author  :   Logan Zou
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于InternLM模型自定义 LLM 类
'''

from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

class InternLM_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str, load_in_8bit: bool = True):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map='auto',
            low_cpu_mem_usage=True, # 是否使用低CPU内存,使用 device_map 参数必须为 True
            quantization_config=quantization_config if load_in_8bit else None
        )
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
                        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
                        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
                        """
        # messages = [(system_prompt, '')]
        # response, history = self.model.chat(self.tokenizer, prompt , history=messages)
        # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1149
        # chat 调用的 generate
        history = []
        response, history = self.model.chat(
            tokenizer = self.tokenizer,
            query = prompt,
            history = history,
            streamer = None,
            max_new_tokens = 1024,
            do_sample = True,
            temperature = 0.8,
            top_p = 0.8,
            meta_instruction = system_prompt,
        )
        return response

    @property
    def _llm_type(self) -> str:
        return "InternLM"

if __name__ == "__main__":
    # 测试代码
    llm = InternLM_LLM(model_path = "../models/internlm2-chat-1_8b")
    # print(llm.predict("你是谁"))
    print(llm.invoke("你是谁"))

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
from typing import Any, List, Optional, Iterator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch


class InternLM_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None
    system_prompt: str = None

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        adapter_dir: str = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        system_prompt: str = """You are an AI assistant whose name is InternLM (书生·浦语).
                        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
                        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
                        """
    ):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()

        self.system_prompt = system_prompt

        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        # 量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = load_in_4bit,                # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
            load_in_8bit = False if load_in_4bit else load_in_8bit,
            llm_int8_threshold = 6.0,
            llm_int8_has_fp16_weight = False,
            bnb_4bit_compute_dtype = torch.float16,     # 4位精度计算的数据类型。这里设置为torch.float16，表示使用半精度浮点数。
            bnb_4bit_quant_type = 'nf4',                # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。 nf4: 4bit-NormalFloat
            bnb_4bit_use_double_quant = True,           # 是否使用双精度量化。如果设置为True，则使用双精度量化。
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map='auto',
            low_cpu_mem_usage=True, # 是否使用低CPU内存,使用 device_map 参数必须为 True
            quantization_config=quantization_config if load_in_8bit or load_in_4bit else None,
        )

        if adapter_dir:
            print(f"load adapter: {adapter_dir}")
            # 2种加载adapter的方式
            # 1. load adapter https://huggingface.co/docs/transformers/main/zh/peft
            # self.model.load_adapter(adapter_dir)

            # 2. https://huggingface.co/docs/peft/main/en/package_reference/peft_model#peft.PeftModel.from_pretrained
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)

        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(
        self,
        prompt : str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
        ) -> str:

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
            max_new_tokens = kwargs.get('max_new_tokens', 1024),
            do_sample = True,
            temperature = kwargs.get('temperature', 0.8),
            top_p = kwargs.get('top_p', 0.8),
            top_k = kwargs.get('top_k', 40),
            meta_instruction = self.system_prompt,
        )
        return response

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        length = 0
        for response, history in self.model.stream_chat(
            tokenizer = self.tokenizer,
            query = prompt,
            history = history,
            max_new_tokens = kwargs.get('max_new_tokens', 1024),
            do_sample = True,
            temperature = kwargs.get('temperature', 0.8),
            top_p = kwargs.get('top_p', 0.8),
            top_k = kwargs.get('top_k', 40),
            meta_instruction = self.system_prompt,
        ):
            if response is not None:
                yield GenerationChunk(text=response[length:])
                length = len(response)

    @property
    def _llm_type(self) -> str:
        return "InternLM2"


if __name__ == "__main__":
    # 测试代码
    llm = InternLM_LLM(model_path = "./models/internlm2-chat-1_8b")
    # print(llm.predict("你是谁"))
    print(llm.invoke("你是谁"))

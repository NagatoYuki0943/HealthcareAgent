from dataclasses import dataclass
from typing import Generator, Any


@dataclass
class TransformersConfig:
    pretrained_model_name_or_path: str
    adapter_dir: str = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    system_prompt: str = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """


@dataclass
class LmdeployConfig:
    model_path: str
    backend: str = 'turbomind' # turbomind, pytorch
    model_format: str = 'hf'
    model_name: str = 'internlm2'
    custom_model_name: str = 'internlm2_chat_1_8b'
    system_prompt: str = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """


class InferEngine:
    def __init__(
        self,
        backend = 'transformers', # transformers, lmdeploy
        transformers_config: TransformersConfig = None,
        lmdeploy_config: LmdeployConfig = None,
    ) -> None:
        assert backend in ['transformers', 'lmdeploy'], f"backend must be 'transformers' or 'lmdeploy', but got {backend}"
        self.backend = backend

        self.transformers_config = transformers_config
        self.lmdeploy_config = lmdeploy_config

        if backend == 'transformers':
            assert transformers_config is not None, "transformers_config must not be None when backend is 'transformers'"
            self.load_transformers_model(
                transformers_config.pretrained_model_name_or_path,
                transformers_config.adapter_dir,
                transformers_config.load_in_8bit,
                transformers_config.load_in_4bit
            )
            print("system_prompt: ", transformers_config.system_prompt)
            print("transformers model loaded")
        elif backend == 'lmdeploy':
            assert lmdeploy_config is not None, "lmdeploy_config must not be None when backend is 'lmdeploy'"
            self.load_lmdeploy_model(
                lmdeploy_config.model_path,
                lmdeploy_config.backend,
                lmdeploy_config.model_format,
                lmdeploy_config.model_name,
                lmdeploy_config.custom_model_name,
                lmdeploy_config.system_prompt
            )
            print("system_prompt: ", lmdeploy_config.system_prompt)
            print("lmdeploy model loaded")

    def load_transformers_model(
        self,
        pretrained_model_name_or_path: str,
        adapter_dir: str = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print("torch version: ", torch.__version__)
        print("transformers version: ", transformers.__version__)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code = True)

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

        # 创建模型
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype = torch.float16,
            trust_remote_code = True,
            device_map = 'auto',
            low_cpu_mem_usage = True,   # 是否使用低CPU内存,使用 device_map 参数必须为 True
            quantization_config = quantization_config if load_in_8bit or load_in_4bit else None,
        )

        if adapter_dir:
            print(f"load adapter: {adapter_dir}")
            # 2种加载adapter的方式
            # 1. load adapter https://huggingface.co/docs/transformers/main/zh/peft
            # self.model.load_adapter(adapter_dir)
            # 2. https://huggingface.co/docs/peft/main/en/package_reference/peft_model#peft.PeftModel.from_pretrained
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)

        self.model.eval()

        # print(model.__class__.__name__) # InternLM2ForCausalLM

        print(f"model.device: {self.model.device}, model.dtype: {self.model.dtype}")

    def load_lmdeploy_model(
        self,
        model_path: str,
        backend: str = 'turbomind', # turbomind, pytorch
        model_format: str = 'hf',
        model_name: str = 'internlm2',
        custom_model_name: str = 'internlm2_chat_1_8b',
        system_prompt: str = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """
    ):
        import lmdeploy
        from lmdeploy import pipeline, PytorchEngineConfig, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig

        print("lmdeploy version: ", lmdeploy.__version__)

        assert backend in ['turbomind', 'pytorch'], f"backend must be 'turbomind' or 'pytorch', but got {backend}"
        assert model_format in ['hf', 'llama', 'awq'], f"model_format must be 'hf' or 'llama' or 'awq', but got {model_format}"

        if backend == 'turbomind':
            # 可以直接使用transformers的模型,会自动转换格式
            # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#turbomindengineconfig
            backend_config = TurbomindEngineConfig(
                model_name = model_name,
                model_format = model_format, # The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by awq. Default: None. Type: str
                tp = 1,
                session_len = 8192,
                max_batch_size = 128,
                cache_max_entry_count = 0.8,    # 调整KV Cache的占用比例为0.8
                cache_block_seq_len = 64,
                quant_policy = 0,               # 4 表示 kv int4 量化, 8 表示 kv int8 量化
                rope_scaling_factor = 0.0,
                use_logn_attn = False,
                download_dir = None,
                revision = None,
                max_prefill_token_num = 8192,
            )
        else:
            # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#pytorchengineconfig
            backend_config = PytorchEngineConfig(
                model_name = model_name,
                tp = 1,
                session_len = 8192,
                max_batch_size = 128,
                cache_max_entry_count = 0.8, # 调整KV Cache的占用比例为0.8
                eviction_type = 'recompute',
                prefill_interval = 16,
                block_size = 64,
                num_cpu_blocks = 0,
                num_gpu_blocks = 0,
                adapters = None,
                max_prefill_token_num = 8192,
                thread_safe = False,
                download_dir = None,
                revision = None,
            )

        # https://lmdeploy.readthedocs.io/zh-cn/latest/_modules/lmdeploy/model.html#ChatTemplateConfig
        chat_template_config = ChatTemplateConfig(
            model_name = model_name,
            system = None,
            meta_instruction = system_prompt,
        )

        # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/async_engine.py
        self.pipe = pipeline(
            model_path = model_path,
            model_name = custom_model_name,
            backend_config = backend_config,
            chat_template_config = chat_template_config,
        )

        # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#generationconfig
        self.gen_config = GenerationConfig(
            n = 1,
            max_new_tokens = 1024,
            top_p = 0.8,
            top_k = 40,
            temperature = 0.8,
            repetition_penalty = 1.0,
            ignore_eos = False,
            random_seed = None,
            stop_words = None,
            bad_words = None,
            min_new_tokens = None,
            skip_special_tokens = True,
        )

    def transformers_chat(
        self,
        query: str,
        history: list = [],  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 40,
        temperature: float = 0.8,
        **kwargs,
    ) -> tuple[str, list]:
        print({
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
        })

        # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1149
        # chat 调用的 generate
        response, history = self.model.chat(
            tokenizer = self.tokenizer,
            query = query,
            history = history,
            streamer = None,
            max_new_tokens = max_new_tokens,
            do_sample = True,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            meta_instruction = self.transformers_config.system_prompt,
        )
        return response, history

    def convert_history(
        self,
        query: str,
        history: list
    ) -> list:
        """
        将历史记录转换为openai格式

        Args:
            query (str): query
            history (list): [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]

        Returns:
            list: prompts (List[str] | str | List[Dict] | List[Dict]): a batch of
                    prompts. It accepts: string prompt, a list of string prompts,
                    a chat history in OpenAI format or a list of chat history.
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
                    },
                    {
                        "role": "assistant",
                        "content": "The capital of France is Paris."
                    },
                    {
                        "role": "user",
                        "content": "Thanks!"
                    },
                    {
                        "role": "assistant",
                        "content": "You are welcome."
                    }
                ]
        """
        # 将历史记录转换为openai格式
        prompts = []
        for user, assistant in history:
            prompts.append(
                {
                    "role": "user",
                    "content": user
                }
            )
            prompts.append(
                {
                    "role": "assistant",
                    "content": assistant
                })
        # 需要添加当前的query
        prompts.append(
            {
                "role": "user",
                "content": query
            }
        )
        return prompts

    def lmdeploy_chat(
        self,
        query: str,
        history: list = [],  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 40,
        temperature: float = 0.8,
        **kwargs,
    ) -> tuple[str, list]:
        # 将历史记录转换为openai格式
        prompts = self.convert_history(query, history)

        # 更新生成config
        self.gen_config.max_new_tokens = max_new_tokens
        self.gen_config.top_p = top_p
        self.gen_config.top_k = top_k
        self.gen_config.temperature = temperature
        print("gen_config: ", self.gen_config)

        # 放入 [{},{}] 格式返回一个response
        # 放入 [] 或者 [[{},{}]] 格式返回一个response列表
        response = self.pipe(prompts=prompts, gen_config=self.gen_config).text
        history.append([query, response])
        return response, history

    def chat(
        self,
        query: str,
        history: list = [],  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 40,
        temperature: float = 0.8,
        **kwargs,
    ) -> tuple[str, list]:
        """对话

        Args:
            query (str): 问题
            history (list, optional): 对话历史. Defaults to [].
                example: [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
            max_new_tokens (int, optional): 单次对话返回最大长度. Defaults to 1024.
            top_p (float, optional): top_p. Defaults to 0.8.
            top_k (int, optional): top_k. Defaults to 40.
            temperature (float, optional): temperature. Defaults to 0.8.

        Returns:
            tuple[str, list]: 回答和历史记录
        """

        if self.backend == 'transformers':
            return self.transformers_chat(query, history, max_new_tokens, top_p, top_k, temperature, **kwargs)
        elif self.backend == 'lmdeploy':
            return self.lmdeploy_chat(query, history, max_new_tokens, top_p, top_k, temperature, **kwargs)

    def transformers_chat_stream(
        self,
        query: str,
        history: list = [],  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 40,
        temperature: float = 0.8,
        **kwargs,
    ) -> Generator[Any, Any, Any]:
        print({
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
        })

        # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1185
        # stream_chat 返回的句子长度是逐渐边长的,length的作用是记录之前的输出长度,用来截断之前的输出
        for response, history in self.model.stream_chat(
                tokenizer = self.tokenizer,
                query = query,
                history = history,
                max_new_tokens = max_new_tokens,
                do_sample = True,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                meta_instruction = self.transformers_config.system_prompt,
            ):
            if response is not None:
                yield response, history

    def lmdeploy_chat_stream(
        self,
        query: str,
        history: list = [],  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 40,
        temperature: float = 0.8,
    ) -> Generator[Any, Any, Any]:
        # 将历史记录转换为openai格式
        prompts = self.convert_history(query, history)

        # 更新生成config
        self.gen_config.max_new_tokens = max_new_tokens
        self.gen_config.top_p = top_p
        self.gen_config.top_k = top_k
        self.gen_config.temperature = temperature
        print("gen_config: ", self.gen_config)

        response = ""
        # 放入 [{},{}] 格式返回一个response
        # 放入 [] 或者 [[{},{}]] 格式返回一个response列表
        for _response in self.pipe.stream_infer(
            prompts = prompts,
            gen_config = self.gen_config,
            do_preprocess = True,
            adapter_name = None
        ):
            # print(_response)
            # Response(text='很高兴', generate_token_len=10, input_token_len=111, session_id=0, finish_reason=None)
            # Response(text='认识', generate_token_len=11, input_token_len=111, session_id=0, finish_reason=None)
            # Response(text='你', generate_token_len=12, input_token_len=111, session_id=0, finish_reason=None)
            response += _response.text
            yield response, history + [[query, response]]

    def chat_stream(
        self,
        query: str,
        history: list = [],
        max_new_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 40,
        temperature: float = 0.8,
        **kwargs,
    ) -> Generator[Any, Any, Any]:
        """流式返回对话

        Args:
            query (str): 问题
            history (list, optional): 对话历史. Defaults to [].
                example: [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
            max_new_tokens (int, optional): 单次对话返回最大长度. Defaults to 1024.
            top_p (float, optional): top_p. Defaults to 0.8.
            top_k (int, optional): top_k. Defaults to 40.
            temperature (float, optional): temperature. Defaults to 0.8.

        Yields:
            Generator[Any, Any, Any]: 回答和历史记录
        """
        if self.backend == 'transformers':
            return self.transformers_chat_stream(query, history, max_new_tokens, top_p, top_k, temperature, **kwargs)
        elif self.backend == 'lmdeploy':
            return self.lmdeploy_chat_stream(query, history, max_new_tokens, top_p, top_k, temperature, **kwargs)

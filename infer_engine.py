from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generator, AsyncGenerator, Literal, Sequence
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer
from threading import Thread
from queue import Empty, Queue
import asyncio
import random
from loguru import logger
from templates import get_prompt_template
from utils import random_uuid_int


@dataclass
class TransformersConfig:
    pretrained_model_name_or_path: str
    adapter_path: str = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    model_name: str = 'internlm2'  # 用于查找对应的对话模板
    system_prompt: str = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """


@dataclass
class LmdeployConfig:
    model_path: str
    backend: Literal['turbomind', 'pytorch'] = 'turbomind'
    model_format: Literal['hf', 'llama', 'awq'] = 'hf'
    cache_max_entry_count: float = 0.8  # 调整 KV Cache 的占用比例为0.8
    quant_policy: int = 0               # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
    model_name: str = 'internlm2'
    system_prompt: str = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
    """
    log_level: Literal['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'] = 'ERROR'
    deploy_method: Literal['local', 'serve'] = 'local'
    server_name: str = '0.0.0.0'
    server_port: int = 23333
    api_keys: list[str] | str | None = None
    ssl: bool = False


def convert_history(
    query: str,
    history: Sequence
) -> list:
    """
    将历史记录转换为openai格式

    Args:
        query (str): query
        history (list): [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]

    Returns:
        list: a chat history in OpenAI format or a list of chat history.
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
    prompt = []
    for user, assistant in history:
        prompt.append(
            {
                "role": "user",
                "content": user
            }
        )
        prompt.append(
            {
                "role": "assistant",
                "content": assistant
            })
    # 需要添加当前的query
    prompt.append(
        {
            "role": "user",
            "content": query
        }
    )
    return prompt


class DeployEngine(ABC):

    @abstractmethod
    def chat(
        self,
        *args,
        **kwargs,
    ) -> tuple[str, Sequence]:
        """对话

        Returns:
            tuple[str, Sequence]: 回答和历史记录
        """
        pass

    @abstractmethod
    def chat_stream(
        self,
        *args,
        **kwargs,
    ) -> Generator[tuple[str, Sequence], None, None]:
        """流式返回对话

        Yields:
            Generator[tuple[str, Sequence], None, None]: 回答和历史记录
        """
        pass


class TransfomersEngine(DeployEngine):
    def __init__(self, config: TransformersConfig) -> None:
        from transformers import BitsAndBytesConfig
        from peft import PeftModel

        logger.info(f"torch version: {torch.__version__}")
        logger.info(f"transformers version: {transformers.__version__}")
        logger.info(f"transformers config: {config}")

        # transformers config
        self.config = config

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path, trust_remote_code = True)

        # 量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = config.load_in_4bit,                # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
            load_in_8bit = False if config.load_in_4bit else config.load_in_8bit,
            llm_int8_threshold = 6.0,
            llm_int8_has_fp16_weight = False,
            bnb_4bit_compute_dtype = torch.float16,     # 4位精度计算的数据类型。这里设置为torch.float16，表示使用半精度浮点数。
            bnb_4bit_quant_type = 'nf4',                # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。 nf4: 4bit-NormalFloat
            bnb_4bit_use_double_quant = True,           # 是否使用双精度量化。如果设置为True，则使用双精度量化。
        )

        # 创建模型
        self.model = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model_name_or_path,
            torch_dtype = torch.float16,
            trust_remote_code = True,
            device_map = 'auto',
            low_cpu_mem_usage = True,   # 是否使用低CPU内存,使用 device_map 参数必须为 True
            quantization_config = quantization_config if config.load_in_8bit or config.load_in_4bit else None,
        )

        if config.adapter_path:
            logger.info(f"load adapter: {config.adapter_path}")
            # 2种加载adapter的方式
            # 1. load adapter https://huggingface.co/docs/transformers/main/zh/peft
            # self.model.load_adapter(adapter_path)
            # 2. https://huggingface.co/docs/peft/main/en/package_reference/peft_model#peft.PeftModel.from_pretrained
            self.model = PeftModel.from_pretrained(self.model, config.adapter_path)

        self.model.eval()

        logger.info(f"model.device: {self.model.device}, model.dtype: {self.model.dtype}")

        # 获取对话模板
        self.prompt_template: dict = get_prompt_template(config.model_name)
        # 停止词
        self.stop_words = [self.tokenizer.eos_token] + self.prompt_template.get('STOP_WORDS', [])
        logger.info(f"stop_words: {self.stop_words}")
        # 停止id
        self.stop_ids = self.tokenizer.convert_tokens_to_ids(self.stop_words)
        logger.info(f"stop_ids: {self.stop_ids}")

    # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1136-L1146
    def build_inputs(
        self,
        tokenizer,
        query: str,
        history: list[tuple[str, str]] | None = None,
        meta_instruction = ""
    ) -> tuple[str, Sequence]:
        history = [] if history is None else list(history)

        if tokenizer.add_bos_token:
            prompt = ""
        else:
            prompt = tokenizer.bos_token
        # 系统指令
        if meta_instruction:
            prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
        # 历史对话
        for record in history:
            # 拼接问题和答案
            prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
        # 用户最新的问题
        prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
        return prompt, tokenizer([prompt], return_tensors="pt")

    def build_inputs_advanced(
        self,
        tokenizer,
        query: str,
        history: list[tuple[str, str]] | None = None, # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        meta_instruction = ""
    ) -> tuple[str, Sequence]:
        """支持多个模型的对话模板"""
        history = [] if history is None else list(history)

        # 对话模板的各个部分
        system_template = self.prompt_template.get('SYSTEM', '<|System|>:{system}\n')
        instruction_template = self.prompt_template.get('INSTRUCTION', '<|User|>:{input}\n<|Bot|>:')
        suffix = self.prompt_template.get('SUFFIX', "")
        sep = self.prompt_template.get('SEP', '\n')

        if tokenizer.add_bos_token:
            prompt = ""
        else:
            prompt = tokenizer.bos_token
        # 系统指令
        if meta_instruction:
            prompt += system_template.format(system=meta_instruction)
        # 历史对话
        for record in history:
            # 拼接问题和答案
            prompt += instruction_template.format(input=record[0]) + record[1] + suffix + sep
        # 用户最新的问题
        prompt += instruction_template.format(input=query)
        # logger.info(f"prompt_template: \n{prompt}")
        return prompt, tokenizer([prompt], return_tensors="pt")

    # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1148-L1182
    @torch.no_grad()
    def __chat(
        self,
        tokenizer,
        query: str,
        history: Sequence | None = None,
        streamer: BaseStreamer | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        meta_instruction: str = "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.",
        **kwargs,
    ) -> tuple[str, Sequence]:
        history = [] if history is None else list(history)
        # _, inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
        _, inputs = self.build_inputs_advanced(tokenizer, query, history, meta_instruction)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if torch.is_tensor(v)}
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        # eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.stop_ids, # eos_token_id,
            **kwargs,
        )
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split("<|im_end|>")[0]
        history = history + [(query, response)]
        return response, history

    # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1184-L1268
    @torch.no_grad()
    def __stream_chat(
        self,
        tokenizer,
        query: str,
        history: Sequence | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        **kwargs,
    ) -> Generator[tuple[str, Sequence], None, None]:
        """
        Return a generator in format: (response, history)
        Eg.
        ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')])
        ('你好，有什么可以帮助您的吗？', [('你好', '你好，有什么可以帮助您的吗？')])
        """
        history = [] if history is None else list(history)

        response_queue = Queue(maxsize=20)

        stop_words = self.stop_words

        class ChatStreamer(BaseStreamer):
            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer
                self.queue = response_queue
                self.query = query
                self.history = history
                self.response = ""
                self.cache = []
                self.received_inputs = False
                self.queue.put((self.response, history + [(self.query, self.response)]))

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("ChatStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]

                if not self.received_inputs:
                    # The first received value is input_ids, ignore here
                    self.received_inputs = True
                    return

                self.cache.extend(value.tolist())
                token = self.tokenizer.decode(self.cache, skip_special_tokens=True)
                # if token.strip() != "<|im_end|>":
                if token.strip() not in stop_words:
                    self.response = self.response + token
                    history = self.history + [(self.query, self.response)]
                    self.queue.put((self.response, history))
                    self.cache = []
                else:
                    self.end()

            def end(self):
                self.queue.put(None)

        def stream_producer():
            return self.__chat(
                tokenizer=tokenizer,
                query=query,
                streamer=ChatStreamer(tokenizer=tokenizer),
                history=history,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        def consumer():
            producer = Thread(target=stream_producer)
            # 启动多线程
            producer.start()
            # 下面的命令是在主线程中运行的
            while True:
                res = response_queue.get()
                if res is None:
                    return
                yield res

        return consumer()

    def chat(
        self,
        query: str,
        history: Sequence | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 40,
        session_id: int | None = None,
        **kwargs,
    ) -> tuple[str, Sequence]:
        # session_id
        logger.info(f"{session_id = }")

        logger.info("gen_config: {}".format({
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }))

        logger.info(f"query: {query}")
        # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1149
        # response, history = self.model.chat( # only for internlm2
        response, history = self.__chat(
            tokenizer = self.tokenizer,
            query = query,
            history = history,
            streamer = None,
            max_new_tokens = max_new_tokens,
            do_sample = True,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            meta_instruction = self.config.system_prompt,
        )
        logger.info(f"response: {response}")
        logger.info(f"history: {history}")
        return response, history

    def chat_stream(
        self,
        query: str,
        history: Sequence | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 40,
        session_id: int | None = None,
        **kwargs,
    ) -> Generator[tuple[str, Sequence], None, None]:
        # session_id
        logger.info(f"{session_id = }")

        logger.info("gen_config: {}".format({
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }))

        logger.info(f"query: {query}")
        # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1185
        # stream_chat 返回的句子长度是逐渐边长的,length的作用是记录之前的输出长度,用来截断之前的输出
        # for response, history in self.model.stream_chat( # only for internlm2
        for response, history in self.__stream_chat(
                tokenizer = self.tokenizer,
                query = query,
                history = history,
                max_new_tokens = max_new_tokens,
                do_sample = True,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                meta_instruction = self.config.system_prompt,
            ):
            logger.info(f"response: {response}")
            if response is not None:
                yield response, history
        logger.info(f"history: {history}")


class LmdeployEngine(DeployEngine):
    def __init__(self, config: LmdeployConfig) -> None:
        import lmdeploy
        from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, ChatTemplateConfig

        logger.info(f"lmdeploy version: {lmdeploy.__version__}")
        logger.info(f"lmdeploy config: {config}")

        assert config.backend in ['turbomind', 'pytorch'], \
            f"backend must be 'turbomind' or 'pytorch', but got {config.backend}"
        assert config.model_format in ['hf', 'llama', 'awq'], \
            f"model_format must be 'hf' or 'llama' or 'awq', but got {config.model_format}"
        assert config.cache_max_entry_count >= 0.0 and config.cache_max_entry_count <= 1.0, \
            f"cache_max_entry_count must be >= 0.0 and <= 1.0, but got {config.cache_max_entry_count}"
        assert config.quant_policy in [0, 4, 8], f"quant_policy must be 0, 4 or 8, but got {config.quant_policy}"

        self.config = config

        if config.backend == 'turbomind':
            # 可以直接使用transformers的模型,会自动转换格式
            # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#turbomindengineconfig
            # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
            self.backend_config = TurbomindEngineConfig(
                model_name = config.model_name,
                model_format = config.model_format, # The format of input model. `hf` meaning `hf_llama`, `llama` meaning `meta_llama`, `awq` meaning the quantized model by awq. Default: None. Type: str
                tp = 1,
                session_len = None,                 # the max session length of a sequence, default to None
                max_batch_size = 128,
                cache_max_entry_count = config.cache_max_entry_count,
                cache_block_seq_len = 64,
                enable_prefix_caching = False,
                quant_policy = config.quant_policy, # KV Cache 量化, 0 代表禁用, 4 代表 4bit 量化, 8 代表 8bit 量化
                rope_scaling_factor = 0.0,
                use_logn_attn = False,
                download_dir = None,
                revision = None,
                max_prefill_token_num = 8192,
                num_tokens_per_iter = 0,
                max_prefill_iters = 1,
            )
        else:
            # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#pytorchengineconfig
            # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
            self.backend_config = PytorchEngineConfig(
                model_name = config.model_name,
                tp = 1,
                session_len = None,                 # the max session length of a sequence, default to None
                max_batch_size = 128,
                cache_max_entry_count = config.cache_max_entry_count,
                eviction_type = 'recompute',
                prefill_interval = 16,
                block_size = 64,
                num_cpu_blocks = 0,
                num_gpu_blocks = 0,
                adapters = None,
                max_prefill_token_num = 4096,
                thread_safe = False,
                enable_prefix_caching = False,
                download_dir = None,
                revision = None,
            )
        logger.info(f"lmdeploy backend_config: {self.backend_config}")

        # https://lmdeploy.readthedocs.io/zh-cn/latest/_modules/lmdeploy/model.html#ChatTemplateConfig
        self.chat_template_config = ChatTemplateConfig(
            model_name = config.model_name, # All the chat template names: `lmdeploy list`
            system = None,
            meta_instruction = config.system_prompt,
            eosys = None,
            user = None,
            eoh = None,
            assistant = None,
            eoa = None,
            separator = None,
            capability = None,
            stop_words = None,
        )
        logger.info(f"lmdeploy chat_template_config: {self.chat_template_config}")


class LmdeployLocalEngine(LmdeployEngine):
    def __init__(self, config: LmdeployConfig) -> None:
        super().__init__(config)

        from lmdeploy import pipeline

        # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/async_engine.py
        self.pipe = pipeline(
            model_path = config.model_path,
            model_name = None,
            backend_config = self.backend_config,
            chat_template_config = self.chat_template_config,
            log_level = config.log_level
        )

    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/async_engine.py#L453-L528
    def __stream_infer(
        self,
        prompts: list[str] | str | list[dict] | list[list[dict]],
        session_ids: int | list[int],
        gen_config = None,
        do_preprocess: bool = True,
        adapter_name: str | None = None,
        **kwargs
    ) -> Generator:
        """Inference a batch of prompts with stream mode.

        Args:
            prompts (List[str] | str | List[Dict] | List[Dict]): a batch of
                prompts. It accepts: string prompt, a list of string prompts,
                a chat history in OpenAI format or a list of chat history.
            session_ids (List[int] | int): a batch of session ids.
            gen_config (GenerationConfig | None): a instance of or a list of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
        """
        from lmdeploy.messages import GenerationConfig, Response
        from lmdeploy.serve.async_engine import _get_event_loop

        need_list_wrap = isinstance(prompts, str) or isinstance(
            prompts[0], dict)
        prompts = [prompts] if need_list_wrap else prompts
        need_list_wrap = isinstance(session_ids, int)
        session_ids = [session_ids] if need_list_wrap else session_ids

        assert isinstance(prompts, list), 'prompts should be a list'
        assert len(prompts) == len(session_ids), 'the length of prompts and session_ids should be the same'

        if gen_config is None:
            gen_config = GenerationConfig()
        # set random if it is not set
        if not isinstance(gen_config, list) and gen_config.random_seed is None:
            gen_config.random_seed = random.getrandbits(64)
        if not isinstance(gen_config, list):
            gen_config = [gen_config] * len(prompts)
        assert len(prompts) == len(gen_config),\
                'input gen_confg length differs from the length of prompts' # noqa
        outputs = Queue()
        generators = []
        # for i, prompt in enumerate(prompts):
        for prompt, session_id, gen_conf in zip(prompts, session_ids, gen_config):
            generators.append(
                self.pipe.generate(prompt,
                              session_id,   # i
                              gen_config=gen_conf,  # gen_config[i]
                              stream_response=True,
                              sequence_start=True,
                              sequence_end=True,
                              do_preprocess=do_preprocess,
                              adapter_name=adapter_name,
                              **kwargs))

        async def _inner_call(i, generator):
            async for out in generator:
                outputs.put(
                    Response(out.response, out.generate_token_len,
                             out.input_token_len, i, out.finish_reason,
                             out.token_ids, out.logprobs))

        async def gather():
            await asyncio.gather(
                # *[_inner_call(i, generators[i]) for i in range(len(prompts))])
                *[_inner_call(session_id, generator) for session_id, generator in zip(session_ids, generators)])
            outputs.put(None)

        loop = _get_event_loop()
        proc = Thread(target=lambda: loop.run_until_complete(gather()))
        # 启动多线程
        proc.start()

        while True:
            try:
                out = outputs.get(timeout=0.001)
                if out is None:
                    break
                yield out
            except Empty:
                pass

        # 等待子线程执行完毕
        proc.join()

    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/async_engine.py#L453-L528
    def __stream_infer_single(
        self,
        prompt: str | list[dict],
        session_id: int,
        gen_config = None,
        do_preprocess: bool = True,
        adapter_name: str | None = None,
        **kwargs
    ) -> Generator:
        """Inference a batch of prompts with stream mode.
        将输入的promot限制在一条

        Args:
            prompt (str | list[dict]): a prompt. It accepts: string prompt,
            a chat history in OpenAI format.
            session_id (int): a session id.
            gen_config (GenerationConfig | None): a instance of or a list of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
        """
        from lmdeploy.messages import GenerationConfig, Response
        from lmdeploy.serve.async_engine import _get_event_loop

        if gen_config is None:
            gen_config = GenerationConfig()
        # set random if it is not set
        if gen_config.random_seed is None:
            gen_config.random_seed = random.getrandbits(64)

        outputs = Queue()
        generator = self.pipe.generate(prompt,
                              session_id,
                              gen_config=gen_config,
                              stream_response=True,
                              sequence_start=True,
                              sequence_end=True,
                              do_preprocess=do_preprocess,
                              adapter_name=adapter_name,
                              **kwargs)

        async def _inner_call(i, generator):
            async for out in generator:
                outputs.put(
                    Response(out.response, out.generate_token_len,
                             out.input_token_len, i, out.finish_reason,
                             out.token_ids, out.logprobs))

        async def gather():
            await asyncio.gather(
                _inner_call(session_id, generator))
            outputs.put(None)

        loop = _get_event_loop()
        proc = Thread(target=lambda: loop.run_until_complete(gather()))
        # 启动多线程
        proc.start()

        while True:
            try:
                out = outputs.get(timeout=0.001)
                if out is None:
                    break
                yield out
            except Empty:
                pass

        # 等待子线程执行完毕
        proc.join()

    # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/gradio/turbomind_coupled.py#L21-L67
    async def chat_stream_local(
        self,
        prompt: str | list[dict],
        session_id: int,
        gen_config = None,
        do_preprocess: bool = True,
        adapter_name: str | None = None,
        **kwargs
    ) -> AsyncGenerator:
        """stream chat 异步实现

        Args:
            prompt (str | list[dict]): a prompt. It accepts: string prompt,
            a chat history in OpenAI format.
            session_id (int): a session id.
            gen_config (GenerationConfig | None): a instance of or a list of
                GenerationConfig. Default to None.
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
            adapter_name (str): the adapter name of slora for pytorch backend.
                Pick one from adapters. Default to None, using the base model.
        """
        from lmdeploy.messages import GenerationConfig
        from lmdeploy.serve.async_engine import GenOut
        from lmdeploy.messages import Response

        if gen_config is None:
            gen_config = GenerationConfig()
        # set random if it is not set
        if gen_config.random_seed is None:
            gen_config.random_seed = random.getrandbits(64)

        output: GenOut
        async for output in self.pipe.generate(
                prompt,
                session_id,
                gen_config=gen_config,
                stream_response=True,
                sequence_start=True,
                sequence_end=True,
                do_preprocess=do_preprocess,
                adapter_name=adapter_name,
            ):
            yield Response(
                text=output.response,
                generate_token_len=output.generate_token_len,
                input_token_len=output.input_token_len,
                session_id=session_id,
                finish_reason=output.finish_reason,
                token_ids=output.token_ids,
                logprobs=output.logprobs,
            )

    def chat(
        self,
        query: str,
        history: Sequence | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 40,
        session_id: int | None = None,
        **kwargs,
    ) -> tuple[str, Sequence]:
        from lmdeploy import GenerationConfig
        from lmdeploy.messages import Response

        # session_id
        logger.info(f"{session_id = }")

        # 将历史记录转换为openai格式
        prompt = convert_history(query, history)

        # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#generationconfig
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
        gen_config = GenerationConfig(
            n = 1,
            max_new_tokens = max_new_tokens,
            top_p = top_p,
            top_k = top_k,
            temperature = temperature,
            repetition_penalty = 1.0,
            ignore_eos = False,
            random_seed = None,
            stop_words = None,
            bad_words = None,
            min_new_tokens = None,
            skip_special_tokens = True,
            logprobs = None,
        )
        logger.info(f"gen_config: {gen_config}")

        logger.info(f"query: {query}")
        # 放入 [{},{}] 格式返回一个response
        # 放入 [] 或者 [[{},{}]] 格式返回一个response列表
        response: Response
        response = self.pipe(
            prompts = prompt,
            gen_config = gen_config,
            do_preprocess = True,
            adapter_name = None
        )
        logger.info(f"response: {response}")
        response_text = response.text
        logger.info(f"response_text: {response_text}")
        history.append([query, response_text])
        logger.info(f"history: {history}")

        return response_text, history

    def chat_stream(
        self,
        query: str,
        history: Sequence | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 40,
        session_id: int | None = None,
        **kwargs,
    ) -> Generator[tuple[str, Sequence], None, None]:
        from lmdeploy import GenerationConfig
        from lmdeploy.messages import Response

        # session_id
        session_id = random_uuid_int() if session_id is None else session_id
        logger.info(f"{session_id = }")

        # 将历史记录转换为openai格式
        prompt = convert_history(query, history)

        # https://lmdeploy.readthedocs.io/zh-cn/latest/api/pipeline.html#generationconfig
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
        gen_config = GenerationConfig(
            n = 1,
            max_new_tokens = max_new_tokens,
            top_p = top_p,
            top_k = top_k,
            temperature = temperature,
            repetition_penalty = 1.0,
            ignore_eos = False,
            random_seed = None,
            stop_words = None,
            bad_words = None,
            min_new_tokens = None,
            skip_special_tokens = True,
            logprobs = None,
        )
        logger.info(f"gen_config: {gen_config}")

        logger.info(f"query: {query}")
        response_text = ""
        # 放入 [{},{}] 格式返回一个response
        # 放入 [] 或者 [[{},{}]] 格式返回一个response列表
        response: Response
        # for response in self.pipe.stream_infer(
        for response in self.__stream_infer_single(
        # async for response in self.chat_stream_local(
            prompt = prompt,
            session_id = session_id,
            gen_config = gen_config,
            do_preprocess = True,
            adapter_name = None
        ):
            logger.info(f"response: {response}")
            # Response(text='很高兴', generate_token_len=10, input_token_len=111, session_id=0, finish_reason=None)
            # Response(text='认识', generate_token_len=11, input_token_len=111, session_id=0, finish_reason=None)
            # Response(text='你', generate_token_len=12, input_token_len=111, session_id=0, finish_reason=None)
            response_text += response.text
            yield response_text, history + [[query, response_text]]

        logger.info(f"response_text: {response_text}")
        logger.info(f"history: {history + [[query, response_text]]}")


class LmdeployServeEngine(LmdeployEngine):
    def __init__(self, config: LmdeployConfig) -> None:
        super().__init__(config)

        from lmdeploy import serve, client

        # 启动服务
        # https://lmdeploy.readthedocs.io/zh-cn/latest/serving/api_server.html
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/api_server.py
        serve(
            model_path = config.model_path,
            model_name = None,
            backend = config.backend,
            backend_config = self.backend_config,
            chat_template_config = self.chat_template_config,
            server_name = config.server_name,
            server_port = config.server_port,
            log_level = config.log_level,
            api_keys = config.api_keys,
            ssl = config.ssl,
        )

        self.api_server_url = f'http://{config.server_name}:{config.server_port}'
        self.api_key = config.api_keys

        # 启动一个 client,所有访问共同使用一个 client,不清楚是否有影响
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/api.py
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/api_client.py
        self.api_client = client(
            api_server_url = self.api_server_url,
            api_key = self.api_key
        )

    def chat_interactive_v1(
        self,
        query: str,
        history: Sequence | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 40,
        session_id: int | None = None,
        stream: bool = True,
        **kwargs,
    ) -> Generator[tuple[str, Sequence], None, None]:
        # session_id
        session_id = random_uuid_int() if session_id is None else session_id
        logger.info(f"{session_id = }")

        # 将历史记录转换为openai格式
        prompt = convert_history(query, history)

        logger.info("gen_config: {}".format({
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }))

        logger.info(f"query: {query}")
        response_text = ""
        response: dict
        for response in self.api_client.chat_interactive_v1(
            prompt = prompt,
            session_id = session_id,
            interactive_mode = False,
            stream = stream,                     # 是否使用流式传输
            stop = None,
            request_output_len = max_new_tokens, # 不确定是不是同一个参数
            top_p = top_p,
            top_k = top_k,
            temperature = temperature,
            repetition_penalty = 1.0,
            ignore_eos = False,
            skip_special_tokens = True,
            adapter_name = None,
        ):
            logger.info(f"response: {response}")
            response_text += response.get("text", "")
            yield response_text, history + [[query, response_text]]

        logger.info(f"response_text: {response_text}")
        logger.info(f"history: {history + [[query, response_text]]}")

    def chat(
        self,
        query: str,
        history: Sequence | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 40,
        session_id: int | None = None,
        **kwargs,
    ) -> tuple[str, Sequence]:
        # 将 generator 转换为 list,返回第一次输出
        return list(self.chat_interactive_v1(
            query = query,
            history = history,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            session_id = session_id,
            stream = False, # don't use stream
            **kwargs,
        ))[0]

    def chat_stream(
        self,
        query: str,
        history: Sequence | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 40,
        session_id: int | None = None,
        **kwargs,
    ) -> Generator[tuple[str, Sequence], None, None]:
        return self.chat_interactive_v1(
            query = query,
            history = history,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            session_id = session_id,
            stream = True,  # use stream
            **kwargs,
        )


class InferEngine(DeployEngine):
    def __init__(
        self,
        backend: Literal['transformers', 'lmdeploy'] = 'transformers',
        transformers_config: TransformersConfig = None,
        lmdeploy_config: LmdeployConfig = None,
    ) -> None:
        assert backend in ['transformers', 'lmdeploy'], f"backend must be 'transformers' or 'lmdeploy', but got {backend}"
        self.backend = backend

        if backend == 'transformers':
            assert transformers_config is not None, "transformers_config must not be None when backend is 'transformers'"
            self.engine = TransfomersEngine(transformers_config)
            logger.info("transformers model loaded")
        elif backend == 'lmdeploy':
            assert lmdeploy_config is not None, "lmdeploy_config must not be None when backend is 'lmdeploy'"
            assert lmdeploy_config.deploy_method in ['local', 'serve'], f"deploy_method must be 'local' or 'serve', but got {lmdeploy_config.deploy_method}"
            if lmdeploy_config.deploy_method == 'local':
                self.engine = LmdeployLocalEngine(lmdeploy_config)
            elif lmdeploy_config.deploy_method == 'serve':
                self.engine = LmdeployServeEngine(lmdeploy_config)
            logger.info("lmdeploy model loaded")

    def chat(
        self,
        query: str,
        history: Sequence | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 40,
        session_id: int | None = None,
        **kwargs,
    ) -> tuple[str, Sequence]:
        """对话

        Args:
            query (str): 问题
            history (Sequence, optional): 对话历史. Defaults to [].
                example: [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
            max_new_tokens (int, optional): 单次对话返回最大长度. Defaults to 1024.
            temperature (float, optional): temperature. Defaults to 0.8.
            top_p (float, optional): top_p. Defaults to 0.8.
            top_k (int, optional): top_k. Defaults to 40.
            session_id (int, optional): 会话id. Defaults to None.

        Returns:
            tuple[str, Sequence]: 回答和历史记录
        """
        history = [] if history is None else list(history)
        return self.engine.chat(
            query = query,
            history = history,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            session_id = session_id,
            **kwargs
        )

    def chat_stream(
        self,
        query: str,
        history: Sequence | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 40,
        session_id: int | None = None,
        **kwargs,
    ) -> Generator[tuple[str, Sequence], None, None]:
        """流式返回对话

        Args:
            query (str): 问题
            history (Sequence, optional): 对话历史. Defaults to [].
                example: [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
            max_new_tokens (int, optional): 单次对话返回最大长度. Defaults to 1024.
            temperature (float, optional): temperature. Defaults to 0.8.
            top_p (float, optional): top_p. Defaults to 0.8.
            top_k (int, optional): top_k. Defaults to 40.
            session_id (int, optional): 会话id. Defaults to None.

        Yields:
            Generator[tuple[str, Sequence], None, None]: 回答和历史记录
        """
        history = [] if history is None else list(history)
        return self.engine.chat_stream(
            query = query,
            history = history,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            session_id = session_id,
            **kwargs
        )

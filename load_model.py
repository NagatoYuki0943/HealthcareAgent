import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch


def load_model(
    pretrained_model_name_or_path: str,
    adapter_path: str = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    print("torch version: ", torch.__version__)
    print("transformers version: ", transformers.__version__)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code = True)

    # 量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,                # 是否在4位精度下加载模型。如果设置为True，则在4位精度下加载模型。
        load_in_8bit = False if load_in_4bit else load_in_8bit,
        llm_int8_threshold = 6.0,
        llm_int8_has_fp16_weight = False,
        bnb_4bit_compute_dtype = torch.bfloat16,    # 4位精度计算的数据类型。这里设置为torch.bfloat16，表示使用半精度浮点数。
        bnb_4bit_quant_type = 'nf4',                # 4位精度量化的类型。这里设置为"nf4"，表示使用nf4量化类型。 nf4: 4bit-NormalFloat
        bnb_4bit_use_double_quant = True,           # 是否使用双精度量化。如果设置为True，则使用双精度量化。
    )

    # 创建模型
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype = torch.bfloat16,
        trust_remote_code = True,
        device_map = 'auto',
        low_cpu_mem_usage = True,   # 是否使用低CPU内存,使用 device_map 参数必须为 True
        quantization_config = quantization_config if load_in_8bit or load_in_4bit else None,
    )

    if adapter_path:
        print(f"load adapter: {adapter_path}")
        # 2种加载adapter的方式
        # 1. load adapter https://huggingface.co/docs/transformers/main/zh/peft
        # model.load_adapter(adapter_path)
        # 2. https://huggingface.co/docs/peft/main/en/package_reference/peft_model#peft.PeftModel.from_pretrained
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # print(model.__class__.__name__) # InternLM2ForCausalLM

    print(f"model.device: {model.device}, model.dtype: {model.dtype}")
    return tokenizer, model


if __name__ == '__main__':
    # clone 模型
    pretrained_model_name_or_path = "../models/internlm2-chat-1_8b"
    # os.system(f'git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b {pretrained_model_name_or_path}')
    # os.system(f'cd {pretrained_model_name_or_path} && git lfs pull')
    adapter_path = None

    # 量化
    load_in_8bit = False
    load_in_4bit = False

    tokenizer, model = load_model(pretrained_model_name_or_path, adapter_path, load_in_8bit, load_in_4bit)
    print(tokenizer)
    print(model)

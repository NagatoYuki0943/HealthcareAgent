# from https://github.com/InternLM/xtuner/blob/main/xtuner/utils/templates.py
from loguru import logger
from typing import Sequence


# - Turn 0: SYSTEM + INSTRUCTION, [output + SUFFIX], SEP
# - Turn 1: INSTRUCTION, [output + SUFFIX], SEP
# - Turn ...
# Note: [] means having supervised loss during the fine-tuning
PROMPT_TEMPLATE = dict(
    default=dict(
        SYSTEM='<|System|>:{system}\n',
        INSTRUCTION='<|User|>:{input}\n<|Bot|>:',
        SEP='\n'),
    zephyr=dict(
        SYSTEM='<|system|>\n{system}\n',
        INSTRUCTION='<|user|>\n{input}\n<|assistant|>\n',
        SEP='\n'),
    internlm_chat=dict(
        SYSTEM='<|System|>:{system}\n',
        INSTRUCTION='<|User|>:{input}<eoh>\n<|Bot|>:',
        SUFFIX='<eoa>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<eoa>']),
    internlm2_chat=dict(
        SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>']),
    moss_sft=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<|Human|>: {input}<eoh>\n',
        SEP='\n',
        STOP_WORDS=['<eoc>', '<eom>']),
    llama2_chat=dict(
        SYSTEM=(
            '[INST] <<SYS>>\n You are a helpful, respectful and honest '
            'assistant. Always answer as helpfully as possible, while being '
            'safe. Your answers should not include any harmful, unethical, '
            'racist, sexist, toxic, dangerous, or illegal content. Please '
            'ensure that your responses are socially unbiased and positive in '
            'nature.\n{system}\n<</SYS>>\n [/INST] '),
        INSTRUCTION='[INST] {input} [/INST]',
        SEP='\n'),
    code_llama_chat=dict(
        SYSTEM='{system}\n', INSTRUCTION='[INST] {input} [/INST]'),
    chatglm2=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='[Round {round}]\n\n问：{input}\n\n答：',
        SEP='\n\n'),
    chatglm3=dict(
        SYSTEM='<|system|>\n{system}',
        INSTRUCTION='<|user|>\n{input}<|assistant|>\n',
        SEP='\n'),
    qwen_chat=dict(
        SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>', '<|endoftext|>']),
    baichuan_chat=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<reserved_102>{input}<reserved_103>',
        SEP='\n'),
    baichuan2_chat=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<reserved_106>{input}<reserved_107>',
        SEP='\n'),
    wizardlm=dict(
        SYSTEM=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        INSTRUCTION=('USER: {input} ASSISTANT:'),
        SEP='\n'),
    wizardcoder=dict(
        SYSTEM=(
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n'
            '{system}\n '),
        INSTRUCTION=('### Instruction:\n{input}\n\n### Response:'),
        SEP='\n\n'),
    vicuna=dict(
        SYSTEM=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        INSTRUCTION=('USER: {input} ASSISTANT:'),
        SEP='\n'),
    deepseek_coder=dict(
        SYSTEM=('You are an AI programming assistant, utilizing '
                'the DeepSeek Coder model, developed by DeepSeek'
                'Company, and you only answer questions related '
                'to computer science. For politically sensitive '
                'questions, security and privacy issues, and '
                'other non-computer science questions, you will '
                'refuse to answer. {system}\n'),
        INSTRUCTION=('### Instruction:\n{input}\n### Response:\n'),
        SEP='\n'),
    # TODO: deprecation, v0.2.0
    deepseekcoder=dict(
        SYSTEM=('You are an AI programming assistant, utilizing '
                'the DeepSeek Coder model, developed by DeepSeek'
                'Company, and you only answer questions related '
                'to computer science. For politically sensitive '
                'questions, security and privacy issues, and '
                'other non-computer science questions, you will '
                'refuse to answer. {system}\n'),
        INSTRUCTION=('### Instruction:\n{input}\n### Response:\n'),
        SEP='\n'),
    deepseek_moe=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
    mistral=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
    mixtral=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
    gemma=dict(
        # `system` field is extended by xtuner
        SYSTEM=('<start_of_turn>system\n{system}<end_of_turn>\n'),
        INSTRUCTION=('<start_of_turn>user\n{input}<end_of_turn>\n'
                     '<start_of_turn>model\n'),
        SUFFIX='<end_of_turn>',
        SUFFIX_AS_EOS=False,
        SEP='\n',
        STOP_WORDS=['<end_of_turn>']),
    cohere_chat=dict(
        SYSTEM=('<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system}'
                '<|END_OF_TURN_TOKEN|>'),
        INSTRUCTION=(
            '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{input}<|END_OF_TURN_TOKEN|>'
            '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'),
        SUFFIX='<|END_OF_TURN_TOKEN|>',
        SUFFIX_AS_EOS=True,
        STOP_WORDS=['<|END_OF_TURN_TOKEN|>']),
    llama3_chat=dict(
        SYSTEM=('<|start_header_id|>system<|end_header_id|>\n\n'
                '{system}<|eot_id|>'),
        INSTRUCTION=(
            '<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'),
        SUFFIX='<|eot_id|>',
        SUFFIX_AS_EOS=True,
        STOP_WORDS=['<|eot_id|>']),
    phi3_chat=dict(
        SYSTEM='<|system|>\n{system}<|end|>\n',
        INSTRUCTION='<|user|>\n{input}<|end|>\n<|assistant|>\n',
        SUFFIX='<|end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|end|>']),
)

"""https://github.com/InternLM/xtuner/blob/main/xtuner/engine/hooks/evaluate_chat_hook.py#L54-L58

instruction = prompt_template.get('INSTRUCTION', '{input}')
if system != '':
    system = prompt_template.get(
        'SYSTEM', '{system}\n').format(system=system)
stop_words += prompt_template.get('STOP_WORDS', [])
"""


def get_prompt_template(model_name: str = "internlm2"):
    """根据模型名字找对话模板"""
    if model_name in PROMPT_TEMPLATE.keys():
        # 模型名字,例如 vicuna
        prompt_template = PROMPT_TEMPLATE.get(model_name)
        logger.info(f"Using prompt template: `{model_name}`")
    elif f"{model_name}_chat" in PROMPT_TEMPLATE.keys():
        # 模型名字_chat,例如 internlm2_chat
        prompt_template = PROMPT_TEMPLATE.get(f"{model_name}_chat")
        logger.info(f"Using prompt template: `{model_name}_chat`")
    else:
        # 默认值
        prompt_template = PROMPT_TEMPLATE.get("default")
        logger.warning(f"Using prompt template: `default`")
    logger.info(f"prompt_template: {prompt_template}")
    return prompt_template


# # https://huggingface.co/internlm/internlm2-chat-1_8b/blob/main/modeling_internlm2.py#L1136-L1146
def build_inputs_internlm2(
    query: str,
    history: list[tuple[str, str]] | None = None,
    meta_instruction = ""
) -> tuple[str, Sequence]:
    history = [] if history is None else list(history)

    prompt = ""
        # 系统指令
    if meta_instruction:
        prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
    # 历史对话
    for record in history:
        # 拼接问题和答案
        prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
    # 用户最新的问题
    prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
    return prompt


def build_inputs(
    prompt_template: dict,
    query: str,
    history: list[tuple[str, str]] | None = None, # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    meta_instruction = ""
) -> tuple[str, Sequence]:
    """支持多个模型的对话模板"""
    history = [] if history is None else list(history)

    system_template = prompt_template.get('SYSTEM', '<|System|>:{system}\n')
    instruction_template = prompt_template.get('INSTRUCTION', '<|User|>:{input}\n<|Bot|>:')
    suffix = prompt_template.get('SUFFIX', "")
    sep = prompt_template.get('SEP', '\n')
    stop_words = prompt_template.get('STOP_WORDS', []) # TODO: add tokenizer.eos_token

    # 对话模板的各个部分
    prompt = ""
    # 系统指令
    if meta_instruction:
        prompt += system_template.format(system=meta_instruction)
    # 历史对话
    for record in history:
        # 拼接问题和答案
        prompt += instruction_template.format(input=record[0]) + record[1] + suffix + sep
    # 用户最新的问题
    prompt += instruction_template.format(input=query)
    logger.info(f"prompt_template: \n{prompt}")
    return prompt


if __name__ == "__main__":
    prompt_template = get_prompt_template()
    get_prompt_template("llama3")
    get_prompt_template("gpt-4o")

    history = [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    query = 'Can you introduce some foods in French?'
    meta_instruction = "You are a helpful assistant."
    input1 = build_inputs_internlm2(query, history, meta_instruction)
    input2 = build_inputs(prompt_template, query, history, meta_instruction)

    print(input1 == input2) # True

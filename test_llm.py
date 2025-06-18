import os
from openai import OpenAI
from api_setting import API_KEY, API_URL, API_MODEL

client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL,
)

completion = client.chat.completions.create(
    # 使用配置文件中的模型
    model=API_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
    # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
    extra_body={"enable_thinking": False},
)
print(completion.model_dump_json())
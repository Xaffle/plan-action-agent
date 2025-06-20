# api_setting.py

# 设置 LLM API 的全局变量

import os


API_PROVIDER = "deepseek" # 可设置为 'qwen' 或 'deepseek'

if API_PROVIDER == "qwen":
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    API_MODEL = "qwen3-30b-a3b" # "qwen-plus-2025-01-25"
elif API_PROVIDER == "deepseek":
    API_KEY = os.getenv("DS_API_KEY")
    API_URL = os.getenv("DS_BASE_URL")
    API_MODEL = "DeepSeek-V3-0324"
else:
    raise ValueError(f"Unknown API_PROVIDER: {API_PROVIDER}")

# 检查必需的环境变量
if not API_KEY:
    raise ValueError(f"API key is required. Please set the appropriate environment variable for {API_PROVIDER} provider.")

# 你可以根据需要添加更多设置
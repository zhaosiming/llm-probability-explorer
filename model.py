"""
验证：
torch 成功导入
transformers 也能用了（如果你加了那行）
环境配置正确
PyCharm 正在使用正确的 llm_env 环境

"""

import warnings
warnings.filterwarnings("ignore", message=".*numpy.*")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
print(torch.__version__)

from transformers import AutoTokenizer
print("OK!")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# ✅ 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ✅ 使用新的模型名称
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"

print("正在从镜像加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    resume_download=True  # 断点续传，网络不好也不怕
)

print("正在从镜像加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=False,        # 先不用 8bit，避免兼容问题
    resume_download=True       # 断点续传
)

# 测试生成
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 运行测试
generate("你好，你是谁？")
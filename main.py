# -*- coding: utf-8 -*-
"""
LLM 概率推演演示：Qwen1.5-1.8B-Chat
-----------------------------------
目标：展示模型生成每个 token 时的 top-10 概率分布
步骤：
  1. 使用国内镜像加载最新模型
  2. 逐 token 生成文本
  3. 打印每一步的概率分布
  4. 展示 LLM 的“思考过程”
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"

print("正在从镜像加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("正在从镜像加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # 强制使用 float16
    device_map={"": "cpu"},  # 使用 CPU
    trust_remote_code=True,
    load_in_8bit=False,
    attn_implementation="eager"  # 避免使用 Flash Attention
)

model.eval()


# ================== 4. 概率推演函数 ==================
def generate_with_probs(prompt, max_new_tokens=30):
    """
    生成文本并打印每一步 top-10 概率分布
    返回：最终生成的文本
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"\n📝 提示词: {prompt}\n")

    output_content = ""  # 建议用 snake_case 风格

    for i in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]  # 最后一个 token 的 logits

            # 转为概率分布
            probs = torch.softmax(next_token_logits, dim=-1)
            sorted_probs, sorted_indices = torch.topk(probs, 10)  # 取 top-10

            # 解码 top-10 的 token
            top_tokens = tokenizer.batch_decode(sorted_indices[0], skip_special_tokens=False)
            top_probs = sorted_probs[0].cpu().numpy()

            # 打印每一步的概率
            print(f"步骤 {i + 1}:")
            for token, prob in zip(top_tokens, top_probs):
                print(f"  '{token}': {prob:.4f} ({prob * 100:.2f}%)")

            # 贪心策略：选概率最高的 token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            new_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
            print(f"  ➡️  选择: '{new_text}'\n")

            # ✅ 追加到输出内容
            output_content += new_text

            # 更新输入序列
            inputs = {
                'input_ids': torch.cat([inputs['input_ids'], next_token], dim=-1),
                'attention_mask': torch.cat([
                    inputs['attention_mask'],
                    torch.ones((1, 1), device=inputs['attention_mask'].device)
                ], dim=-1)
            }

            # 如果生成结束符，提前退出
            if next_token.item() == tokenizer.eos_token_id:
                print("✅ 遇到结束符，生成终止。\n")
                break

    # ✅ 打印最终结果
    print("✨ 最终生成结果:")
    print(f"「{output_content}」\n")

    # ✅ 返回结果，便于后续使用
    return output_content

# ================== 5. 运行推演 ==================
if __name__ == "__main__":
    # prompt = "你是一个富有同理心的回答者。请回答：爱情是什么？"
    prompt = "爱情是？"
    result = generate_with_probs(prompt, max_new_tokens=30)
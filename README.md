# llm-probability-explorer 🧠

> 一个让你“看见”大模型如何思考的工具 —— 逐 token 展示生成概率与 top-k 分布。

本项目通过加载本地大模型（如 Qwen），可视化 **语言建模过程**，揭示 LLM 不是“黑箱”，而是基于上下文对每个 token 进行打分并选择的**概率引擎**。

---

## 🚀 功能亮点

- ✅ 加载本地 LLM（支持 Hugging Face 模型）
- ✅ 逐 token 输出 **logits** 与 **top-10 概率分布**
- ✅ 可视化模型“思考过程”：它是如何一步步生成文本的
- ✅ 适合教学、研究、理解 LLM 原理

---

## 📦 安装与运行

```bash
# 1. 克隆项目
git clone https://github.com/your-username/llm-probability-explorer.git
cd llm-probability-explorer

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行（需先下载 Qwen 等模型）
python main.py

支持的模型包括：

Qwen/Qwen1.5-0.5B-Chat
Qwen/Qwen1.5-1.8B-Chat ✅ 推荐
Qwen/Qwen1.5-4B-Chat
meta-llama/Llama-2-7b-chat-hf（需申请）
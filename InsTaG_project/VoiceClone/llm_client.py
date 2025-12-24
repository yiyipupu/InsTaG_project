# llm_client.py
import re
from openai import OpenAI

# ================= 配置区 =================
# 【强烈推荐】使用 DeepSeek (国内最强开源模型之一，API 兼容性好)
# 申请地址: https://platform.deepseek.com/
API_KEY = "sk-165d09b3fd8e44c697d6393bccefb97c"  # <--- 在这里填入你的 API Key
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# ==========================================

# 初始化客户端
try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    print(f"警告: LLM 客户端初始化失败，请检查 API_KEY. 错误: {e}")
    client = None


def clean_sensevoice_output(text):
    """
    清洗 FunASR/SenseVoice 的输出
    输入: <|zh|><|NEUTRAL|>你好
    输出: 你好
    """
    if not text:
        return ""
    # 正则表达式：匹配所有 <|...|> 的标签并替换为空
    clean_text = re.sub(r'<\|.*?\|>', '', text)
    return clean_text.strip()


def chat_stream(user_text):
    """
    流式对话接口
    """
    # 1. 先清洗 ASR 的脏标签
    clean_input = clean_sensevoice_output(user_text)
    print(f"\n[发送给AI]: {clean_input}")

    if not clean_input:
        yield "没听清，请再说一遍。"
        return

    if not client:
        yield "错误：未配置 API Key，无法连接大脑。"
        return

    try:
        # 2. 发起流式请求
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                # System Prompt: 设定 AI 的人设，这对于数字人很重要
                {"role": "system",
                 "content": "请用口语化的风格回答，句子在50字左右，方便语音合成。"},
                {"role": "user", "content": clean_input},
            ],
            stream=True,  # <--- 必须开启流式
            temperature=0.7,  # 控制回答的随机性
            max_tokens=200  # 限制回答长度，防止废话太多
        )

        # 3. 逐字生成
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        print(f"\nLLM 请求出错: {e}")
        yield "抱歉，我的大脑现在有点乱。"


# === 简单的自测代码 ===
if __name__ == "__main__":
    # 模拟一段带标签的 ASR 输入
    mock_input = "<|zh|><|NEUTRAL|><|Speech|>你好，请问火星上有生命吗？"
    print(f"原始输入: {mock_input}")

    print("AI 回答: ", end="")
    for char in chat_stream(mock_input):
        print(char, end="", flush=True)
    print("\n测试结束")
# llm_client.py
import re
from openai import OpenAI

API_KEY = "sk-165d09b3fd8e44c697d6393bccefb97c"  # <--- 在这里填入你的 API Key
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"


try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    print(f"警告: LLM 客户端初始化失败，请检查 API_KEY. 错误: {e}")
    client = None


STYLE_PROMPTS = {
    "normal": (
        "你是一个普通、理性的正常人，说话自然不夸张。"
        "请用口语化的风格回答，句子在50字左右，方便语音合成。"
        "不要生成表情、符号等非语言内容。"
    ),
    "toxic": (
        "你是一个毒舌但聪明的人，说话带一点讽刺和调侃。"
        "可以轻微吐槽，但不要辱骂、不要人身攻击、不要低俗。"
        "口语化，50字左右，不要生成表情、符号等非语言内容。"
    ),
    "warm": (
        "你是一个温柔、体贴、治愈型的人，说话让人感到被理解和安慰。"
        "语气温暖、耐心，像朋友一样。"
        "口语化，50字左右，不要生成表情、符号等非语言内容。"
    ),
}


def clean_sensevoice_output(text):
    if not text:
        return ""
    clean_text = re.sub(r'<\|.*?\|>', '', text)
    return clean_text.strip()


def chat_stream(user_text, style="normal"):
    clean_input = clean_sensevoice_output(user_text)
    print(f"\n[发送给AI][style={style}]: {clean_input}")

    if not clean_input:
        yield "没听清，请再说一遍。"
        return

    if not client:
        yield "错误：未配置 API Key，无法连接大脑。"
        return

    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["normal"])

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": clean_input},
            ],
            stream=True,
            temperature=0.7 if style == "normal" else 0.9,
            max_tokens=200
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        print(f"\nLLM 请求出错: {e}")
        yield "抱歉，我的大脑现在有点乱。"

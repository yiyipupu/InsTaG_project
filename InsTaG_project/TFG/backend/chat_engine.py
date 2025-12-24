import os
import uuid
from voiceclone.main import asr_request
from voiceclone.llm_client import chat_stream
from backend.video_generator import generate_video


def chat_response_from_audio(
    wav_path: str,
    model_param: str = "",
    gpu_choice: str = "GPU0",
    style: str = "normal"   # ✅ 新增：normal | toxic | warm
):
    """
    输入：浏览器录音 wav 路径
    输出：视频路径
    流程：
    1. ASR
    2. LLM（带风格）
    3. 语音克隆 + InsTaG
    """

    # ========== 1️⃣ ASR ==========
    user_text = asr_request(wav_path)
    if not user_text or not user_text.strip():
        return {"status": "error", "msg": "语音识别失败"}

    print(f"[CHAT] ASR text: {user_text}")

    # ========== 2️⃣ LLM（✅ 带风格） ==========
    reply_text = ""
    for chunk in chat_stream(user_text, style=style):
        reply_text += chunk

    reply_text = reply_text.strip()
    if not reply_text:
        return {"status": "error", "msg": "LLM 无返回"}

    print(f"[CHAT][style={style}] LLM reply: {reply_text}")

    # ========== 3️⃣ TTS + InsTaG ==========
    result = generate_video(
        model_name="InsTaG",
        model_param=model_param,
        ref_audio="",
        target_text=reply_text,
        gpu_choice=gpu_choice,
        page_id="2"
    )

    if result["status"] == "error":
        return {"reply": "视频生成失败"}

    if result["status"] == "pending":
        return {
            "reply": "视频正在生成中，请稍后查看",
            "pending": True
        }

    return {
        "status": "success",
        "reply_text": reply_text,
        "video_path": result["video_path"]
    }

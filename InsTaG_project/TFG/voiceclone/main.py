# voiceclone/main.py
"""
Web 后端专用（WSL/Linux）
只提供：ASR / LLM / TTS 的“纯函数能力”
"""

import os
import requests
from typing import Optional

from voiceclone import llm_client

IP_ADDRESS = "124.70.54.5"

ASR_URL = f"http://{IP_ADDRESS}:8001/asr"
TTS_URL = f"http://{IP_ADDRESS}:8000/inference/zero-shot"

PROMPT_WAV = "static/audios/prompt.wav"
PROMPT_TEXT = "和所有的烦恼说拜拜，和所有的快乐说嗨嗨。"

PROXIES = {"http": None, "https": None}


def asr_request(wav_path: str, timeout_s: int = 60) -> str:
    if not wav_path or not os.path.exists(wav_path):
        return ""

    try:
        with open(wav_path, "rb") as f:
            resp = requests.post(
                ASR_URL,
                files={"audio_file": f},
                proxies=PROXIES,
                timeout=(5, timeout_s),
            )

        if resp.status_code != 200:
            return ""

        data = resp.json()
        raw_text = (data.get("text") or "").strip()
        if not raw_text:
            return ""

        if hasattr(llm_client, "clean_sensevoice_output"):
            raw_text = llm_client.clean_sensevoice_output(raw_text)

        return raw_text.strip()

    except Exception:
        return ""


def llm_reply(user_text: str, style: str = "normal", max_chars: int = 500) -> str:
    """
    user_text -> LLM 回复文本（拼成完整一句/一段）
    style: normal | toxic | warm
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return ""

    reply = ""
    try:
        for chunk in llm_client.chat_stream(user_text, style=style):
            reply += chunk
            if len(reply) >= max_chars:
                break
        return reply.strip()
    except Exception:
        return ""


def tts_request(
    tts_text: str,
    prompt_wav: str = PROMPT_WAV,
    prompt_text: str = PROMPT_TEXT,
    timeout_s: int = 180,
) -> Optional[bytes]:
    tts_text = (tts_text or "").strip()
    if not tts_text:
        return None

    if not os.path.exists(prompt_wav):
        return None

    try:
        with open(prompt_wav, "rb") as f:
            files = [("prompt_audio", ("prompt.wav", f, "audio/wav"))]
            data = {"tts_text": tts_text, "prompt_text": prompt_text}

            resp = requests.post(
                TTS_URL,
                data=data,
                files=files,
                proxies=PROXIES,
                timeout=(5, timeout_s),
            )

        if resp.status_code != 200:
            return None

        content = resp.content
        if not content or len(content) < 1024:
            return None

        return content

    except Exception:
        return None


def tts_to_file(
    tts_text: str,
    out_wav_path: str,
    **kwargs
) -> bool:
    audio_bytes = tts_request(tts_text, **kwargs)
    if not audio_bytes:
        return False

    os.makedirs(os.path.dirname(out_wav_path) or ".", exist_ok=True)
    with open(out_wav_path, "wb") as f:
        f.write(audio_bytes)

    return True

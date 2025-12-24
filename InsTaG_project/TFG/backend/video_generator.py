import os
import time
import requests
from datetime import datetime
import logging

# ======================
# 关闭底层 HTTP DEBUG 日志
# ======================
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ======================
# 语音克隆服务
# ======================
TTS_URL = "http://124.70.54.5:8000/inference/zero-shot"
PROMPT_WAV = "static/audios/prompt.wav"
PROMPT_TEXT = "和所有的烦恼说拜拜，和所有的快乐说嗨嗨。"

# ======================
# InsTaG 服务（job_id 版）
# ======================
INSTAG_URL = "http://140.210.142.177:8000"

TFG_AUDIO_DIR = "static/audios"
TFG_VIDEO_DIR = "static/videos"


def generate_video(
    model_name,
    model_param,
    ref_audio,
    target_text,
    gpu_choice,
    page_id="1"
):
    # =================================================
    # 根据 page_id 选择输出视频名（保持原逻辑）
    # =================================================
    output_name = "result_2.mp4" if str(page_id) == "2" else "result_1.mp4"
    output_path = os.path.join(TFG_VIDEO_DIR, output_name)

    os.makedirs(TFG_AUDIO_DIR, exist_ok=True)
    os.makedirs(TFG_VIDEO_DIR, exist_ok=True)

    # =================================================
    # 1️⃣ TTS（同步等待 + 重试）
    # =================================================
    tts_wav_path = os.path.join(
        TFG_AUDIO_DIR,
        f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    )

    tts_bytes = None
    MAX_RETRY = 3

    for attempt in range(1, MAX_RETRY + 1):
        try:
            print(f"[TTS] attempt {attempt}/{MAX_RETRY}")
            with open(PROMPT_WAV, "rb") as wav_f:
                resp = requests.post(
                    TTS_URL,
                    files={"prompt_audio": ("prompt.wav", wav_f, "audio/wav")},
                    data={
                        "tts_text": target_text,
                        "prompt_text": PROMPT_TEXT
                    },
                    timeout=180
                )

            if resp.status_code == 200 and resp.content and len(resp.content) > 1024:
                tts_bytes = resp.content
                break

            print(f"[TTS] warning: status={resp.status_code}")

        except requests.exceptions.Timeout:
            print("[TTS] timeout")
        except Exception as e:
            print("[TTS] exception:", e)

        time.sleep(2)

    if tts_bytes is None:
        return {"status": "error", "msg": "语音模型多次尝试失败（可能 502 / 超时）"}

    with open(tts_wav_path, "wb") as f:
        f.write(tts_bytes)

    if os.path.getsize(tts_wav_path) < 1024:
        return {"status": "error", "msg": "生成的语音文件过小，疑似失败"}

    # =================================================
    # 2️⃣ 提交到 InsTaG，拿 job_id（保持原逻辑）
    # =================================================
    try:
        with open(tts_wav_path, "rb") as f:
            up = requests.post(
                f"{INSTAG_URL}/infer/upload",
                files={"audio": ("gen.wav", f, "audio/wav")},
                data={
                    "model_param": model_param   # 🔥 关键这一行
                },
                timeout=(5, 60)
            )

        if up.status_code != 200:
            return {"status": "error", "msg": f"InsTaG upload failed: {up.status_code}"}

        up_json = up.json()
        job_id = up_json.get("job_id")
        if not job_id:
            return {"status": "error", "msg": "InsTaG 未返回 job_id"}

    except Exception as e:
        return {"status": "error", "msg": f"InsTaG upload exception: {e}"}

    # =================================================
    # 3️⃣ 轮询 job 状态（🔥指数退避，核心修改）
    # =================================================
    MAX_WAIT = 1200          # 总等待上限（20 分钟）
    waited = 0

    interval = 1             # 初始 1 秒
    MAX_INTERVAL = 15        # 最慢 15 秒一次

    last_status = None

    while waited < MAX_WAIT:
        try:
            st = requests.get(
                f"{INSTAG_URL}/infer/status/{job_id}",
                timeout=10
            )

            if st.status_code == 200:
                info = st.json()
                status = info.get("status")

                # 只在状态变化时打印
                if status != last_status:
                    print(f"[JOB {job_id}] status -> {status}")
                    last_status = status

                if status == "done":
                    break

                if status == "error":
                    return {
                        "status": "error",
                        "msg": f"InsTaG infer error: {info.get('error')}"
                    }

        except Exception:
            pass

        time.sleep(interval)
        waited += interval

        # 🔥 指数退避：逐渐放慢
        interval = min(interval * 2, MAX_INTERVAL)

    else:
        return {
            "status": "pending",
            "msg": "视频仍在生成中，请稍后刷新或再次尝试",
            "job_id": job_id
        }

    # =================================================
    # 4️⃣ 下载该 job 的视频（保持原逻辑）
    # =================================================
    try:
        vid = requests.get(
            f"{INSTAG_URL}/infer/video/{job_id}",
            timeout=60
        )

        if vid.status_code != 200 or not vid.content or len(vid.content) < 1024:
            return {
                "status": "error",
                "msg": f"拉取视频失败: {vid.status_code}"
            }

        with open(output_path, "wb") as f:
            f.write(vid.content)

        return {
            "status": "success",
            "video_path": f"/static/videos/{output_name}"
        }

    except Exception as e:
        return {"status": "error", "msg": f"下载视频异常: {e}"}

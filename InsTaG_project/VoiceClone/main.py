import requests
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time
import sys
import queue
import threading
import re
import llm_client

# ================= 配置中心 =================
IP_ADDRESS = "124.70.54.5"
ASR_URL = f"http://{IP_ADDRESS}:8001/asr"
TTS_URL = f"http://{IP_ADDRESS}:8000/inference/zero-shot"
#http://124.70.54.5:8000/inference/zero-shot
MIC_DEVICE_INDEX = 1  # 你的麦克风 ID
PROMPT_WAV = "prompt.wav"
PROMPT_TEXT = "你好，我是语音助手，很高兴为你服务。"
FS = 16000

# ================= 队列定义 =================
# 1. 文本队列: LLM -> TTS
text_queue = queue.Queue()
# 2. 音频队列: TTS -> Player
audio_queue = queue.Queue()


# ================= 功能函数 =================

def split_text_stream(char_generator):
    """
    智能分句生成器：从流式字符中提取完整的句子
    """
    buffer = ""
    # 句子结束符：句号、感叹号、问号 (中英文)
    sentence_endings = re.compile(r'[。！？.!?\n]')

    for char in char_generator:
        buffer += char
        # 如果检测到标点符号，或者缓存太长了(超过30字强行切分)
        if sentence_endings.search(char) or len(buffer) > 30:
            if buffer.strip():
                yield buffer.strip()
            buffer = ""

    # yield 剩下的部分
    if buffer.strip():
        yield buffer.strip()


def tts_worker():
    """
    后台线程：不断从队列取文字，请求 TTS，存入音频队列
    """
    while True:
        text = text_queue.get()
        if text is None:  # 结束信号
            audio_queue.put(None)
            break

        print(f"  [TTS处理中]: '{text[:10]}...'")
        start = time.time()
        try:
            payload = {"tts_text": text, "prompt_text": PROMPT_TEXT}
            with open(PROMPT_WAV, "rb") as f:
                files = [('prompt_audio', ('prompt.wav', f, 'audio/wav'))]
                resp = requests.post(TTS_URL, data=payload, files=files, timeout=30)

            if resp.status_code == 200:
                # 保存为临时片段文件
                temp_filename = f"temp_{int(time.time() * 1000)}.wav"
                with open(temp_filename, "wb") as f:
                    f.write(resp.content)
                # 放入音频队列
                audio_queue.put(temp_filename)
                print(f"  [TTS完成] 耗时 {time.time() - start:.2f}s")
            else:
                print(f"  [TTS错误] {resp.text}")
        except Exception as e:
            print(f"  [TTS异常] {e}")


def play_worker():
    """
    后台线程：不断从音频队列取文件，播放，然后删除
    """
    while True:
        wav_file = audio_queue.get()
        if wav_file is None:  # 结束信号
            break

        try:
            data, fs = sf.read(wav_file)
            sd.play(data, fs)
            sd.wait()
            # 播放完删除临时文件
            try:
                os.remove(wav_file)
            except:
                pass
        except Exception as e:
            print(f"  [播放错误] {e}")


# ... (record_audio 和 asr_request 保持不变，为了节省篇幅我省略了，请保留你原来的) ...
# 为了确保代码完整运行，我还是把 record_audio 和 asr_request 完整贴一遍吧：

# (录音队列)
q = queue.Queue()


def callback(indata, frames, time, status):
    if status: print(status, file=sys.stderr)
    q.put(indata.copy())


def record_audio(filename="input.wav"):
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except:
            pass
    with q.mutex:
        q.queue.clear()

    try:
        print(f"\n🎤 准备就绪 (设备ID: {MIC_DEVICE_INDEX})")
        input("👉 按【回车键】开始录音...")
        print("🔴 正在录音... (说完再次按回车)")
        with sd.InputStream(samplerate=FS, channels=1, dtype='int16', callback=callback, device=MIC_DEVICE_INDEX):
            input()
        print("⏹️ 录音结束")

        data_list = []
        while not q.empty(): data_list.append(q.get())
        if not data_list: return None
        sf.write(filename, np.concatenate(data_list, axis=0), FS)
        return filename
    except Exception as e:
        print(f"❌ 录音失败: {e}")
        return None


def asr_request(filename):
    print("👂 正在识别...", end="", flush=True)
    try:
        with open(filename, "rb") as f:
            resp = requests.post(ASR_URL, files={"audio_file": f}, timeout=10)
        if resp.status_code == 200:
            txt = llm_client.clean_sensevoice_output(resp.json()['text'])
            print(f" -> {txt}")
            return txt
        return None
    except:
        return None


# ================= 主循环 (修改很大) =================
if __name__ == "__main__":
    print("🚀 数字人极速版 (分句流式生成)")

    if not os.path.exists(PROMPT_WAV):
        print("❌ 缺少 prompt.wav")
        sys.exit(1)

    while True:
        try:
            # 1. 录音
            user_wav = record_audio()
            if not user_wav: continue

            # 2. ASR
            user_text = asr_request(user_wav)
            if not user_text: continue

            print("🧠 AI 思考并回复中...")

            # 3. 启动 TTS 和 播放 线程
            # 每次对话都重新启动线程，虽然消耗一点资源，但逻辑最简单安全
            t_tts = threading.Thread(target=tts_worker)
            t_play = threading.Thread(target=play_worker)
            t_tts.start()
            t_play.start()

            # 4. LLM 流式生成 -> 分句 -> 放入文本队列
            full_reply = ""
            # 获取生成器
            llm_gen = llm_client.chat_stream(user_text)

            # 使用分句器处理流
            for sentence in split_text_stream(llm_gen):
                print(f"📝 生成句子: {sentence}")
                text_queue.put(sentence)  # 丢给 TTS 线程去跑，主线程继续接下一句
                full_reply += sentence

            # 5. 发送结束信号
            text_queue.put(None)  # 告诉 TTS 没话了
            t_tts.join()  # 等 TTS 全部处理完

            audio_queue.put(None)  # 告诉播放器没音频了
            t_play.join()  # 等全部播放完

            print("\n✅ 本轮对话结束")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")
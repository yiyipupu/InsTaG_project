# backend/model_trainer.py
import requests

# ⚠️ 改成你华为云的公网 IP
INSTAG_BASE_URL = "http://140.210.142.177:8000"

def start_training(model_choice, ref_video, gpu_choice, custom_params=""):
    """
    启动 InsTaG 训练（远程华为云）
    """
    if model_choice != "InsTaG":
        return {"status": "error", "msg": "Only InsTaG supported"}

    # GPU0 / GPU1 → 0 / 1
    if not gpu_choice:
        # 默认用 GPU0，或者你想要的策略
        gpu_id = 0
    else:
        gpu_id = int(
            gpu_choice
            .replace("GPU", "")
            .replace("gpu", "")
        )
    try:
        resp = requests.post(
            f"{INSTAG_BASE_URL}/train",
            params={"gpu_id": gpu_id},
            timeout=5
        )
    except Exception as e:
        return {"status": "error", "msg": str(e)}

    if resp.status_code != 200:
        return {"status": "error", "msg": resp.text}

    return {"status": "success"}

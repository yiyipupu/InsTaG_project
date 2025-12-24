import logging
import os
import uuid
from flask import Flask, render_template, request, jsonify
from backend.model_trainer import start_training
from backend.video_generator import generate_video
from backend.chat_engine import chat_response_from_audio

logging.basicConfig(level=logging.INFO)

main_app = Flask(
    "main_app",
    template_folder="templates_main",
    static_folder="static"
)

abstract_app = Flask(
    "abstract_app",
    template_folder="templates_abstract",
    static_folder="static"
)

@main_app.route("/")
@main_app.route("/main_index")
def main_index():
    return render_template("index.html")

@main_app.route("/main/model_training")
def main_model_training():
    return render_template("model_training.html")

@main_app.route("/main/video_generation")
def main_video_generation():
    return render_template("video_generation.html")

@main_app.route("/main/chat_system")
def main_chat_system():
    return render_template("chat_system.html")

@main_app.route("/abstract_index")
def abstract_index():
    return abstract_app.jinja_env.get_or_select_template("index.html").render()

@main_app.route("/abstract/model_training")
def abstract_model_training():
    return abstract_app.jinja_env.get_or_select_template("model_training.html").render()

@main_app.route("/abstract/video_generation")
def abstract_video_generation():
    return abstract_app.jinja_env.get_or_select_template("video_generation.html").render()

@main_app.route("/abstract/chat_system")
def abstract_chat_system():
    return abstract_app.jinja_env.get_or_select_template("chat_system.html").render()

@main_app.route("/model_training", methods=["POST"])
def model_training_api():
    model_choice = request.form.get("model_choice")
    ref_video = request.form.get("ref_video")
    gpu_choice = request.form.get("gpu_choice")
    custom_params = request.form.get("custom_params", "")
    result = start_training(model_choice, ref_video, gpu_choice, custom_params)
    return jsonify(result)

@main_app.route("/video_generation", methods=["POST"])
def video_generation_api():
    model_name = request.form.get("model_name")
    model_param = request.form.get("model_param")
    ref_audio = request.form.get("ref_audio", "")
    target_text = request.form.get("target_text", "")
    gpu_choice = request.form.get("gpu_choice")
    result = generate_video(model_name, model_param, ref_audio, target_text, gpu_choice, page_id="1")
    return jsonify(result)

@main_app.route("/chat/start", methods=["POST"])
def chat_start():
    if "audio" not in request.files:
        return jsonify({"status": "error", "msg": "no audio"}), 400

    audio_file = request.files["audio"]
    os.makedirs("static/audios", exist_ok=True)
    wav_path = f"static/audios/chat_{uuid.uuid4().hex}.wav"
    audio_file.save(wav_path)

    model_param = request.form.get("model_param", "")
    gpu_choice = request.form.get("gpu_choice", "GPU0")

    # ✅ 新增：风格参数
    style = request.form.get("style", "normal")  # normal | toxic | warm

    result = chat_response_from_audio(
        wav_path,
        model_param=model_param,
        gpu_choice=gpu_choice,
        style=style
    )
    return jsonify(result)

if __name__ == "__main__":
    main_app.run(host="0.0.0.0", port=5000, debug=False)

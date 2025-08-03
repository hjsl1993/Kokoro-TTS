# coding:utf-8

"""
https://github.com/thewh1teagle/kokoro-onnx
模型列表：
https://huggingface.co/onnx-community/Kokoro-82M-v1.1-zh-ONNX/tree/main/voices
"""
import gradio as gr
import soundfile as sf
from kokoro_onnx import Kokoro
from misaki import en, espeak, zh

ZH_VOICES = ['af_maple', 'af_sol', 'bf_vale', 'zf_001', 'zf_002', 'zf_003', 'zf_004', 'zf_005', 'zf_006', 'zf_007',
             'zf_008', 'zf_017', 'zf_018', 'zf_019', 'zf_021', 'zf_022', 'zf_023', 'zf_024', 'zf_026', 'zf_027',
             'zf_028', 'zf_032', 'zf_036', 'zf_038', 'zf_039', 'zf_040', 'zf_042', 'zf_043', 'zf_044', 'zf_046',
             'zf_047', 'zf_048', 'zf_049', 'zf_051', 'zf_059', 'zf_060', 'zf_067', 'zf_070', 'zf_071', 'zf_072',
             'zf_073', 'zf_074', 'zf_075', 'zf_076', 'zf_077', 'zf_078', 'zf_079', 'zf_083', 'zf_084', 'zf_085',
             'zf_086', 'zf_087', 'zf_088', 'zf_090', 'zf_092', 'zf_093', 'zf_094', 'zf_099', 'zm_009', 'zm_010',
             'zm_011', 'zm_012', 'zm_013', 'zm_014', 'zm_015', 'zm_016', 'zm_020', 'zm_025', 'zm_029', 'zm_030',
             'zm_031', 'zm_033', 'zm_034', 'zm_035', 'zm_037', 'zm_041', 'zm_045', 'zm_050', 'zm_052', 'zm_053',
             'zm_054', 'zm_055', 'zm_056', 'zm_057', 'zm_058', 'zm_061', 'zm_062', 'zm_063', 'zm_064', 'zm_065',
             'zm_066', 'zm_068', 'zm_069', 'zm_080', 'zm_081', 'zm_082', 'zm_089', 'zm_091', 'zm_095', 'zm_096',
             'zm_097', 'zm_098', 'zm_100', '']

EN_VOICES = ["af", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael", "bf_emma", "bf_isabella",
             "bm_george", "bm_lewis"]
# ZH_VOICES = ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"]

MODELS_INSTANCE = {}  # 模型实例

# Misaki G2P with espeak-ng fallback

fallback = espeak.EspeakFallback(british=False)


def get_model(lang):
    """
    获取模型实例
     {'en': {"model": <Kokoro object at 0x7f95684c5d60>, "g2p": g2p}, 'zh': {"model": <Kokoro object at 0x7f95684c5e00>, "g2p": g2p}
    :param lang:
    :return:
    """
    if lang in MODELS_INSTANCE:
        return MODELS_INSTANCE[lang]["model"], MODELS_INSTANCE[lang]["g2p"]

    if lang == "en":
        print("111111")
        g2p = en.G2P(trf=False, british=False, fallback=fallback)
        kokoro = Kokoro("./models/kokoro-v1.0.onnx", "./models/voices-v1.0.bin")
    elif lang == "zh":
        print("2222222")

        g2p = zh.ZHG2P(version="1.1")
        kokoro = Kokoro("./models/kokoro-v1.1-zh.onnx", "./models/voices-v1.1-zh.bin",
                        vocab_config="./models/config.json")
    else:
        raise ValueError(f"Unknown language: {lang}")

    if lang not in MODELS_INSTANCE:
        MODELS_INSTANCE[lang] = {"model": kokoro, "g2p": g2p}

    print(f"MODELS_INSTANCE: {MODELS_INSTANCE}")
    return MODELS_INSTANCE[lang]["model"], MODELS_INSTANCE[lang]["g2p"]


# ---------- TTS 生成 ----------
def tts_fn(text, lang, voice, speed):
    # engine = pyttsx3.init()
    # voices = engine.getProperty("voices")
    # vid = None
    # for v in voices:
    #     if v.name == voice:
    #         vid = v.id
    #         break
    # if vid:
    #     engine.setProperty("voice", vid)
    # engine.setProperty("rate", int(engine.getProperty("rate") * speed))
    #

    # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    #     path = f.name
    # engine.save_to_file(text, path)
    # engine.runAndWait()
    # return path, gr.File(value=path, label="下载音频")

    print(f"text: {text}\tlang: {lang}\tvoice: {voice}\tspeed: {speed}")

    model, g2p = get_model(lang)
    phonemes, _ = g2p(text)
    samples, sample_rate = model.create(phonemes, voice=voice, speed=speed, is_phonemes=True)

    path = "./output/audio.wav"

    sf.write(path, samples, sample_rate)
    print("Created audio.wav")

    return path, gr.File(value=path, label="下载音频")


# ---------- 动态下拉 ----------
def update_dropdown(lang):
    opts = EN_VOICES if lang == "en" else ZH_VOICES
    return gr.update(choices=opts, value=opts[0])


# ---------- Gradio UI ----------
with gr.Blocks(title="Kokoro TTS") as demo:
    gr.Markdown("## 🎤 文本转语音")

    with gr.Row():
        # ------------- 左侧控制区 -------------
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="输入文本", lines=5, placeholder="请输入英文或中文文本…")

            lang_radio = gr.Radio(["en", "zh"], value="en", label="语言")
            voice_dd = gr.Dropdown(EN_VOICES, value=EN_VOICES[0], label="发音人")
            speed_sld = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="语速倍速")

            gen_btn = gr.Button("🚀 生成语音", variant="primary")

            lang_radio.change(update_dropdown, lang_radio, voice_dd)

        # ------------- 右侧音频区 -------------
        with gr.Column(scale=1):
            upload_audio = gr.Audio(
                label="上传音频（参考/试听）",
                sources=["upload"],
                type="filepath",
                waveform_options=gr.WaveformOptions(
                    waveform_color="#01C6FF",
                    waveform_progress_color="#0066CC",
                    skip_length=2,
                    show_controls=True,
                ),
            )

            output_audio = gr.Audio(
                label="合成结果",
                type="filepath",
                waveform_options=gr.WaveformOptions(
                    waveform_color="#FF8C00",
                    waveform_progress_color="#FF4500",
                    skip_length=1,
                    show_controls=True,
                ),
            )

            download_link = gr.File(label="下载音频")

    # 事件绑定
    gen_btn.click(
        tts_fn,
        inputs=[text_input, lang_radio, voice_dd, speed_sld],
        outputs=[output_audio, download_link],
    )

demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)

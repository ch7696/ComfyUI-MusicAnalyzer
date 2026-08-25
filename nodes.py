# -*- coding: utf-8 -*-
"""
ComfyUI-MusicAnalyzer —— 音乐理解与结构化描述节点（仅分析，无生成）

将一段音频转换为结构化的音乐信息：
  - 音乐分析器：标签 / BPM / 调性 / 音乐信息 JSON
  - 歌词转录器：逐字歌词
  - 音乐描述器：面向文生音乐模型的自然语言描述
  - 音乐信息转JSON：把各部分汇总为一个结构化 JSON

模型支持（需手动下载到本节点 models/ 目录，不会自动下载）：
  - ACE-Step-Transcriber（默认，Qwen2.5-Omni 架构，歌词/结构/声乐）
  - Qwen2-Audio-7B-Instruct
  - Qwen2.5-Omni-3B / 7B，Ke-Omni-R-3B，Qwen3-Omni-8B
  - MiDaShengLM-7B（音乐描述能力最强，另有 GPTQ 4bit 低显存版）
  - Whisper 系列（ASR 转录）、Whisper 音频描述系列
  - MERT-v1-330M（音乐特征嵌入）、AST-AudioSet（音频事件分类）

音频理解核心逻辑提取并重构自 ComfyUI-AceStep_SFT（MIT 协议）。
"""

import gc
import json
import os
import re

import torch

# ===========================================================================
# 1. 模型注册表
# ===========================================================================

# 支持的音频理解模型（HuggingFace 仓库 ID）
_ANALYSIS_MODELS = {
    "ACE-Step-Transcriber": "ACE-Step/acestep-transcriber",
    "Qwen2-Audio-7B-Instruct": "Qwen/Qwen2-Audio-7B-Instruct",
    "Qwen2.5-Omni-3B": "Qwen/Qwen2.5-Omni-3B",
    "Ke-Omni-R-3B": "KE-Team/Ke-Omni-R-3B",
    "Qwen2.5-Omni-7B": "Qwen/Qwen2.5-Omni-7B",
    "Qwen3-Omni-8B": "Qwen/Qwen3-Omni-8B",
    "MiDaShengLM-7B": "mispeech/midashenglm-7b-0804-bf16",
    "MiDaShengLM-7B-GPTQ": "mispeech/midashenglm-7b-0804-w4a16-gptq",
    "Whisper-large-v3-transcription": "openai/whisper-large-v3",
    "Whisper-large-v3-turbo-transcription": "openai/whisper-large-v3-turbo",
    "Distil-Whisper-large-v3.5-transcription": "distil-whisper/distil-large-v3.5",
    "Distil-Whisper-large-v3-transcription": "distil-whisper/distil-large-v3",
    "AST-AudioSet": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "MERT-v1-330M": "m-a-p/MERT-v1-330M",
    "Whisper-large-v2-audio-captioning": "MU-NLPC/whisper-large-v2-audio-captioning",
    "Whisper-small-audio-captioning": "MU-NLPC/whisper-small-audio-captioning",
    "Whisper-tiny-audio-captioning": "MU-NLPC/whisper-tiny-audio-captioning",
}

# 默认模型
_NATIVE_ANALYSIS_MODEL = "ACE-Step-Transcriber"

# 模型类别判断
def _is_whisper_captioning_model(model_key):
    return "audio-captioning" in model_key.lower()

def _is_whisper_asr_model(model_key):
    return model_key.endswith("-transcription")

def _is_acestep_transcriber_model(model_key):
    return model_key == "ACE-Step-Transcriber"


# ===========================================================================
# 2. 模型加载与缓存（单例，切换模型自动卸载旧的）
# ===========================================================================

_audio_model = None
_audio_processor = None
_audio_model_name = None
_audio_tokenizer = None  # 仅 MiDaShengLM 使用


def _get_model_dir(model_key):
    """返回模型在本节点目录下的本地路径。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_key)


def _check_model_local(model_key):
    """检查模型是否已手动下载到节点目录（不做自动下载）。

    模型须手动放到 models/<模型名>/ 下（需包含 config.json）。
    """
    model_dir = _get_model_dir(model_key)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(config_path):
        return model_dir
    repo_id = _ANALYSIS_MODELS[model_key]
    raise RuntimeError(
        f"[MusicAnalyzer] 模型 {model_key} 未找到，请手动下载后放入：\n"
        f"  目录：{model_dir}\n"
        f"下载命令（国内网络建议先执行 set HF_ENDPOINT=https://hf-mirror.com）：\n"
        f"  huggingface-cli download {repo_id} --local-dir \"{model_dir}\"\n"
        f"或使用 Python：\n"
        f"  python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{repo_id}', local_dir=r'{model_dir}')\""
    )


def _get_analysis_device():
    """优先跟随 ComfyUI 当前设备，否则自动选择 cuda/cpu。"""
    try:
        from comfy import model_management
        return model_management.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_analysis_device_map():
    return {"": str(_get_analysis_device())}


def _load_audio_model(model_key, use_flash_attn=False):
    """加载音频理解模型 + 处理器（缓存单例，切换模型时先卸载旧模型）。"""
    global _audio_model, _audio_processor, _audio_model_name, _audio_tokenizer
    if _audio_model is not None and _audio_model_name == model_key:
        return _audio_model, _audio_processor
    if _audio_model is not None:
        _unload_audio_model()

    model_dir = _check_model_local(model_key)
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=_get_analysis_device_map(),
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    if use_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print(f"[MusicAnalyzer] {model_key} 使用 flash_attention_2")
    print(f"[MusicAnalyzer] 正在加载 {model_key} 到 {_get_analysis_device()} ...")

    if _is_acestep_transcriber_model(model_key):
        import warnings
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Flash Attention 2 without specifying a torch dtype.*")
            warnings.filterwarnings("ignore", message=".*Token2WavModel.*fallback.*")
            _audio_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
        _audio_model.disable_talker()
        _audio_model.eval()
        _audio_processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, use_fast=False)
    elif model_key.startswith("Qwen2.5-Omni"):
        import warnings
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Flash Attention 2 without specifying a torch dtype.*")
            warnings.filterwarnings("ignore", message=".*Token2WavModel.*fallback.*")
            _audio_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
        _audio_model.disable_talker()
        _audio_model.eval()
        _audio_processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, use_fast=False)
    elif model_key == "Qwen2-Audio-7B-Instruct":
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        _audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
        _audio_model.eval()
        _audio_processor = AutoProcessor.from_pretrained(model_dir)
    elif model_key == "Qwen3-Omni-8B":
        try:
            from transformers import Qwen3OmniForConditionalGeneration, AutoProcessor
        except ImportError:
            raise RuntimeError(
                "[MusicAnalyzer] Qwen3-Omni-8B 需要 transformers >= 4.53，请先升级：pip install -U transformers"
            ) from None
        _audio_model = Qwen3OmniForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
        _audio_model.eval()
        _audio_processor = AutoProcessor.from_pretrained(model_dir)
    elif model_key.startswith("MiDaShengLM"):
        import warnings
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Flash Attention.*")
            try:
                _audio_model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map=_get_analysis_device_map(),
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
            except Exception as e:
                if "gptq" in str(e).lower() or "quantization" in str(e).lower() or "auto_gptq" in str(e).lower():
                    raise RuntimeError(
                        "[MusicAnalyzer] MiDaShengLM GPTQ 版本需要安装 auto-gptq（pip install auto-gptq），"
                        "或改用 MiDaShengLM-7B（BF16）版本。"
                    ) from e
                raise
        _audio_model.eval()
        _audio_processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        _audio_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    elif _is_whisper_captioning_model(model_key):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        _audio_model = WhisperForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            device_map=_get_analysis_device_map(),
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        _audio_model.eval()
        _audio_processor = WhisperProcessor.from_pretrained(model_dir)
    elif _is_whisper_asr_model(model_key):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        whisper_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        asr_kwargs = {
            "torch_dtype": whisper_dtype,
            "device_map": _get_analysis_device_map(),
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        if use_flash_attn:
            asr_kwargs["attn_implementation"] = "flash_attention_2"
        _audio_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, **asr_kwargs)
        _audio_model.eval()
        _audio_processor = AutoProcessor.from_pretrained(model_dir)
    elif model_key == "Ke-Omni-R-3B":
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
        _audio_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
        _audio_model.eval()
        _audio_processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, use_fast=False)
    elif model_key == "MERT-v1-330M":
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        _audio_model = AutoModel.from_pretrained(
            model_dir, torch_dtype=torch.float32, device_map=_get_analysis_device_map(),
            trust_remote_code=True,
        )
        _audio_model.eval()
        _audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)
    elif model_key == "AST-AudioSet":
        from transformers import ASTForAudioClassification, AutoFeatureExtractor
        _audio_model = ASTForAudioClassification.from_pretrained(
            model_dir, torch_dtype=torch.float32, device_map=_get_analysis_device_map(),
        )
        _audio_model.eval()
        _audio_processor = AutoFeatureExtractor.from_pretrained(model_dir)

    _audio_model_name = model_key
    print(f"[MusicAnalyzer] {model_key} 加载完成。")
    return _audio_model, _audio_processor


def _unload_audio_model():
    """卸载音频理解模型，释放显存。"""
    global _audio_model, _audio_processor, _audio_model_name, _audio_tokenizer
    name = _audio_model_name or "音频模型"
    if _audio_model is not None:
        try:
            _audio_model.to("cpu")
        except Exception:
            pass
        del _audio_model
        _audio_model = None
    if _audio_processor is not None:
        del _audio_processor
        _audio_processor = None
    if _audio_tokenizer is not None:
        del _audio_tokenizer
        _audio_tokenizer = None
    _audio_model_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[MusicAnalyzer] {name} 已卸载，显存已释放。")


# ===========================================================================
# 3. 音频预处理
# ===========================================================================

def _prepare_audio_mono(audio_dict, target_sr, max_seconds):
    """把 ComfyUI 的音频字典转成单声道 float32 numpy 数组，并限制时长。"""
    import numpy as np

    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]

    if waveform.dim() == 3:
        y = waveform[0].mean(dim=0)
    elif waveform.dim() == 2:
        y = waveform.mean(dim=0)
    else:
        y = waveform
    y = y.cpu().numpy().astype(np.float32)

    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    max_samples = target_sr * max_seconds
    if len(y) > max_samples:
        start = (len(y) - max_samples) // 2
        y = y[start:start + max_samples]

    return y


# ===========================================================================
# 4. 标签提取（按模型分派）
# ===========================================================================

def _build_gen_kwargs(temperature, top_p, top_k, repetition_penalty, seed):
    """把界面参数组装成生成参数。"""
    kwargs = {}
    if temperature > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
    else:
        kwargs["do_sample"] = False
    if top_p < 1.0:
        kwargs["top_p"] = top_p
    if top_k > 0:
        kwargs["top_k"] = top_k
    if repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    return kwargs


# 给模型的标签指令（模型用，保持英文以保证输出稳定）
_TAG_TEMPLATE_START = "<<<INICIO_TAGS_TEMPLATE>>>"
_TAG_TEMPLATE_END = "<<<FIM_TAGS_TEMPLATE>>>"

_TAG_INSTRUCTION = (
    "Return the result only inside this exact template, with nothing before or after it:\n"
    "<<<INICIO_TAGS_TEMPLATE>>>\n"
    "tag1, tag2, tag3\n"
    "<<<FIM_TAGS_TEMPLATE>>>\n"
    "Inside the template, write only short lowercase comma-separated tags for this audio. "
    "No labels, no explanation, no sentences, no question, no closing text. "
    "Use only the final tags for rhythm, instrumentation, vocals, production effects, mood, and energy. "
    "Use specific tags such as 'punchy kick drum' instead of generic words. Max 4 words per tag."
)


def _extract_tag_template(result_text):
    """从模型输出中提取模板标记之间的标签内容（兼容模型幻觉变体标记）。"""
    # 截断模型可能幻觉出的对话轮次
    result_text = re.split(r"\n\s*(?:Human|User|Assistant)\s*:", result_text, maxsplit=1, flags=re.IGNORECASE)[0]
    # 去掉 markdown 代码块
    result_text = re.sub(r"```[a-zA-Z]*\n?", "", result_text)
    # 先尝试精确匹配
    start = result_text.find(_TAG_TEMPLATE_START)
    end = result_text.find(_TAG_TEMPLATE_END)
    if start != -1 and end != -1 and end > start:
        start += len(_TAG_TEMPLATE_START)
        return result_text[start:end].strip()
    # 模糊匹配：模型常把标记写成各种变体
    markers = list(re.finditer(r"<{2,3}\s*[^>]+\s*>{2,3}", result_text))
    if len(markers) >= 2:
        return result_text[markers[0].end():markers[1].start()].strip()
    if len(markers) == 1:
        marker_text = markers[0].group().lower()
        if "inicio" in marker_text or "start" in marker_text or "begin" in marker_text:
            return result_text[markers[0].end():].strip()
        return result_text[:markers[0].start()].strip()
    return result_text.strip()


def _extract_tags(audio_dict, model_key, max_new_tokens=200, audio_duration=30,
                  use_flash_attn=False, gen_kwargs=None):
    """用选中的模型提取音乐标签，返回逗号分隔的字符串。"""
    if gen_kwargs is None:
        gen_kwargs = {}
    model, processor = _load_audio_model(model_key, use_flash_attn=use_flash_attn)

    if _is_acestep_transcriber_model(model_key):
        return _extract_tags_acestep_transcriber(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
    elif model_key.startswith("Qwen2.5-Omni") or model_key == "Ke-Omni-R-3B":
        return _extract_tags_qwen_omni(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
    elif model_key == "Qwen3-Omni-8B":
        return _extract_tags_qwen3_omni(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
    elif model_key.startswith("MiDaShengLM"):
        return _extract_tags_midasheng(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
    elif model_key == "Qwen2-Audio-7B-Instruct":
        return _extract_tags_qwen2_audio(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
    elif model_key == "MERT-v1-330M":
        return _extract_tags_mert(audio_dict, model, processor)
    elif _is_whisper_captioning_model(model_key):
        return _extract_tags_whisper_captioning(audio_dict, model, processor, audio_duration, gen_kwargs)
    elif _is_whisper_asr_model(model_key):
        return _extract_tags_whisper_asr(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
    elif model_key == "AST-AudioSet":
        return _extract_tags_ast(audio_dict, model, processor)
    return ""


def _extract_tags_qwen_omni(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs=None):
    """Qwen2.5-Omni / Ke-Omni 标签提取（单轮对话）。"""
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)

    DEFAULT_SYS = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYS}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": y, "sampling_rate": 16000},
            {"type": "text", "text": _TAG_INSTRUCTION},
        ]},
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text_prompt, audio=[y], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[-1]
    gk = {"max_new_tokens": max_new_tokens}
    if hasattr(model, "talker"):
        gk["return_audio"] = False
        gk["use_audio_in_video"] = True
    gk.update(gen_kwargs or {})
    if "repetition_penalty" not in gk:
        gk["repetition_penalty"] = 1.5
    text_ids = model.generate(**inputs, **gk)
    new_tokens = text_ids[:, input_len:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = raw[0].strip() if raw else ""
    print(f"[MusicAnalyzer] 模型原始输出: {result[:300]}")
    return _clean_tags(_extract_tag_template(result))


def _extract_tags_qwen3_omni(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs=None):
    """Qwen3-Omni-8B 标签提取。"""
    if gen_kwargs is None:
        gen_kwargs = {}
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "audio", "audio": y, "sampling_rate": 16000},
            {"type": "text", "text": _TAG_INSTRUCTION},
        ]},
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text_prompt], audio=[y], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[-1]
    gk = {"max_new_tokens": max_new_tokens}
    gk.update(gen_kwargs)
    with torch.inference_mode():
        text_ids = model.generate(**inputs, **gk)
    new_tokens = text_ids[:, input_len:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = raw[0].strip() if raw else ""
    print(f"[MusicAnalyzer] Qwen3-Omni 原始输出: {result[:300]}")
    return _clean_tags(_extract_tag_template(result))


def _extract_tags_midasheng(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs=None):
    """MiDaShengLM 通用音频描述 -> 标签。"""
    if gen_kwargs is None:
        gen_kwargs = {}
    tokenizer = _audio_tokenizer
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": _TAG_INSTRUCTION},
            {"type": "audio", "audio": y},
        ]},
    ]
    mi = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, add_special_tokens=True, return_dict=True,
    )
    if not isinstance(mi, dict):
        from dataclasses import asdict
        mi = {k: v for k, v in asdict(mi).items() if v is not None}
    mi = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in mi.items()}
    gk = {"max_new_tokens": max_new_tokens}
    gk.update(gen_kwargs)
    with torch.inference_mode():
        generation = model.generate(**mi, **gk)
    input_ids = mi.get("input_ids")
    if input_ids is not None:
        generation = generation[:, input_ids.shape[-1]:]
    out = tokenizer.batch_decode(generation, skip_special_tokens=True)
    result = out[0].strip() if out else ""
    print(f"[MusicAnalyzer] MiDaShengLM 原始输出: {result[:300]}")
    return _clean_tags(_extract_tag_template(result))


def _extract_tags_qwen2_audio(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs=None):
    """Qwen2-Audio-7B-Instruct 标签提取（带 flash_attn 冲突自动回退 SDPA）。"""
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)

    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "__PLACEHOLDER__"},
            {"type": "text", "text": _TAG_INSTRUCTION},
        ]},
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text_prompt, audio=[y], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[-1]
    gk = {"max_new_tokens": max_new_tokens}
    if gen_kwargs:
        for key in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty"):
            if key in gen_kwargs:
                gk[key] = gen_kwargs[key]
    try:
        text_ids = model.generate(**inputs, **gk)
    except RuntimeError as e:
        if "cu_seqlens" in str(e):
            # flash_attention_2 与当前 flash-attn 版本不兼容 —— 回退到 SDPA
            print("[MusicAnalyzer] Qwen2-Audio 与 flash_attention_2 不兼容，改用 SDPA 重新加载...")
            _unload_audio_model()
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            global _audio_model, _audio_processor, _audio_model_name
            model_dir = _check_model_local("Qwen2-Audio-7B-Instruct")
            _audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16,
                device_map=_get_analysis_device_map(),
                attn_implementation="sdpa",
            )
            _audio_model.eval()
            _audio_processor = AutoProcessor.from_pretrained(model_dir)
            _audio_model_name = "Qwen2-Audio-7B-Instruct"
            model, processor = _audio_model, _audio_processor
            text_prompt2 = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text_prompt2, audio=[y], sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device).to(model.dtype)
            input_len = inputs["input_ids"].shape[-1]
            text_ids = model.generate(**inputs, **gk)
        else:
            raise
    new_tokens = text_ids[:, input_len:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = raw[0].strip() if raw else ""
    print(f"[MusicAnalyzer] 模型原始输出: {result[:300]}")
    return _clean_tags(_extract_tag_template(result))


def _extract_tags_acestep_transcriber(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs=None):
    """ACE-Step 转录器提取 -> 歌词/结构/声乐标签。"""
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)

    conversation = [
        {"role": "user", "content": [
            {"type": "text", "text": "*Task* Transcribe this audio in detail"},
            {"type": "audio", "audio": y, "sampling_rate": 16000},
        ]},
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text_prompt, audio=[y], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[-1]
    gk = {
        "max_new_tokens": max(256, max_new_tokens),
        "return_audio": False,
    }
    gk.update(gen_kwargs or {})
    if "repetition_penalty" not in gk:
        gk["repetition_penalty"] = 1.1
    text_ids = model.generate(**inputs, **gk)
    new_tokens = text_ids[:, input_len:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = raw[0].strip() if raw else ""
    print(f"[MusicAnalyzer] ACE-Step 转录: {result[:400]}")
    return _derive_tags_from_acestep_transcription(result, audio_duration)


def _extract_tags_whisper_asr(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs=None):
    """Whisper ASR 转录 -> 启发式声乐标签。"""
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)

    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(model.device)
    if torch.is_floating_point(input_features):
        input_features = input_features.to(dtype=model.dtype)

    gk = {"max_new_tokens": max(64, min(max_new_tokens, 256))}
    if hasattr(getattr(model, "generation_config", None), "task_to_id"):
        gk["task"] = "transcribe"
    gk.update(gen_kwargs or {})

    with torch.inference_mode():
        generated_ids = model.generate(input_features=input_features, **gk)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)
    transcript_text = result[0].strip() if result else ""
    print(f"[MusicAnalyzer] Whisper 转录: {transcript_text[:300]}")
    return _derive_tags_from_transcript(transcript_text, audio_duration)


def _extract_tags_whisper_captioning(audio_dict, model, processor, audio_duration, gen_kwargs=None):
    """Whisper 音频描述模型（MU-NLPC）标签提取。"""
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)

    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(model.device)
    gk = {"max_new_tokens": 200}
    gk.update(gen_kwargs or {})
    with torch.inference_mode():
        gen = model.generate(**inputs, **gk)
    result = processor.batch_decode(gen, skip_special_tokens=True)
    text = result[0] if result else ""
    # 去掉模型幻觉出的数据集名前缀
    text = _WHISPER_PREFIXES.sub("", text)
    return _clean_tags(_extract_tag_template(text))


def _extract_tags_ast(audio_dict, model, processor):
    """AST AudioSet 音频事件分类 -> 高分标签。"""
    import numpy as np

    y = _prepare_audio_mono(audio_dict, 16000, 30)

    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
    probs = torch.sigmoid(logits)
    top_indices = probs.argsort(descending=True)[:15].cpu().numpy()
    labels = model.config.id2label
    tags = [labels[int(i)] for i in top_indices if probs[i] > 0.1]
    if not tags:
        tags = [labels[int(top_indices[0])]]
    return ", ".join(tags)


def _extract_tags_mert(audio_dict, model, processor):
    """MERT-v1-330M 音乐嵌入 -> 启发式标签（无生成能力，取激活最高的维度）。"""
    import numpy as np

    _MERT_LABELS = [
        "drums", "bass", "guitar", "piano", "synth", "strings", "brass",
        "woodwind", "vocals", "male vocals", "female vocals", "choir",
        "electronic", "acoustic", "distorted", "clean",
        "fast tempo", "slow tempo", "medium tempo",
        "major key", "minor key",
        "happy", "sad", "aggressive", "calm", "dark", "bright",
        "energetic", "mellow", "groovy", "atmospheric",
        "reverb", "delay", "distortion", "compression",
        "kick drum", "snare", "hi hat", "cymbal", "percussion",
        "sub bass", "pad", "lead synth", "arpeggio",
        "pop", "rock", "jazz", "classical", "hip hop", "electronic music",
        "r&b", "folk", "metal", "funk", "latin", "reggae",
    ]

    y = _prepare_audio_mono(audio_dict, 24000, 30)

    inputs = processor(y, sampling_rate=24000, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]  # [1, T, 1024]
    features = hidden.mean(dim=1).squeeze().cpu().float().numpy()  # [1024]
    n_labels = len(_MERT_LABELS)
    chunk_size = len(features) // n_labels
    scores = np.array([
        features[i * chunk_size:(i + 1) * chunk_size].sum()
        for i in range(n_labels)
    ])
    top_indices = scores.argsort()[::-1][:15]
    tags = [_MERT_LABELS[i] for i in top_indices if scores[i] > 0]
    if not tags:
        tags = [_MERT_LABELS[int(top_indices[0])]]
    result = ", ".join(tags)
    print(f"[MusicAnalyzer] MERT 标签（启发式）: {result}")
    return result


def _clean_tags(result_text):
    """清洗模型输出为简短、去重、小写的标签列表。"""
    result_text = result_text.strip().strip('"').strip("'").strip()
    result_text = result_text.replace("\uff0c", ",").replace("\u3001", ",")
    lines = [ln.strip() for ln in result_text.splitlines() if ln.strip()]
    result_text = ", ".join(lines) if lines else ""

    seen = set()
    unique_tags = []
    for tag in result_text.split(","):
        tag = tag.strip().rstrip(".")
        # 去掉 "1)" / "1." 之类的编号前缀
        tag = re.sub(r"^\d+[).\]]\s*", "", tag).strip()
        # 规范化连字符（"drum - beat" -> "drum beat"）
        tag = re.sub(r"\s*-\s*", " ", tag).strip()
        if not tag:
            continue
        # 跳过 BPM 数字（librosa 单独检测）
        if re.match(r"^\d+\s*bpm$", tag, re.IGNORECASE):
            continue
        # 跳过填充词
        if tag in ("etc", "and more", "more", "and so on", "..."):
            continue
        # 跳过过长条目（>6 词），真实标签都很短
        if len(tag.split()) > 6:
            continue
        tag = tag.lower()
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    # 最多保留 20 个标签
    return ", ".join(unique_tags[:20])


# ===========================================================================
# 5. 转录文本 -> 启发式标签（语言检测 + 声乐特征）
# ===========================================================================

# Whisper 描述模型常加的数据集名前缀
_WHISPER_PREFIXES = re.compile(
    r"^\s*(audiosetrain|audioset\s*keywords?|clotho|audiocaps|music\s*role)\s*[,:]?\s*",
    re.IGNORECASE,
)

_TRANSCRIPT_LANGUAGE_PATTERNS = [
    ("japanese", re.compile(r"[\u3040-\u30ff]")),
    ("chinese", re.compile(r"[\u4e00-\u9fff]")),
    ("korean", re.compile(r"[\uac00-\ud7af]")),
    ("russian", re.compile(r"[\u0400-\u04ff]")),
    ("arabic", re.compile(r"[\u0600-\u06ff]")),
]

_TRANSCRIPT_LANGUAGE_HINTS = {
    "english": {"the", "and", "you", "love", "with", "baby", "night", "heart"},
    "portuguese": {"que", "você", "amor", "pra", "não", "meu", "minha", "coração"},
    "spanish": {"que", "amor", "corazón", "noche", "eres", "tengo", "para", "con"},
    "french": {"je", "tu", "amour", "avec", "pas", "dans", "pour", "coeur"},
    "german": {"ich", "du", "und", "nicht", "liebe", "nacht", "mein", "mit"},
    "italian": {"che", "amore", "notte", "con", "sei", "mio", "mia", "cuore"},
}

_LANGUAGE_CODE_TO_NAME = {
    "ar": "arabic", "de": "german", "el": "greek", "en": "english",
    "es": "spanish", "fa": "persian", "fr": "french", "he": "hebrew",
    "hi": "hindi", "id": "indonesian", "it": "italian", "ja": "japanese",
    "ko": "korean", "ms": "malay", "nl": "dutch", "pl": "polish",
    "pt": "portuguese", "ru": "russian", "th": "thai", "tl": "filipino",
    "tr": "turkish", "uk": "ukrainian", "ur": "urdu", "vi": "vietnamese",
    "zh": "chinese",
}


def _infer_transcript_language(text):
    """根据字符集和常见词推断歌词语言。"""
    for language_name, pattern in _TRANSCRIPT_LANGUAGE_PATTERNS:
        if pattern.search(text):
            return language_name

    tokens = re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower())
    if not tokens:
        return ""

    token_set = set(tokens)
    best_language = ""
    best_score = 0
    for language_name, hints in _TRANSCRIPT_LANGUAGE_HINTS.items():
        score = len(token_set & hints)
        if score > best_score:
            best_language = language_name
            best_score = score
    return best_language if best_score >= 2 else ""


def _derive_tags_from_transcript(transcript_text, audio_duration):
    """从 ASR 转录文本推导声乐相关标签。"""
    transcript_text = transcript_text.strip()
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", transcript_text.lower())
    if len(words) < 3:
        return "instrumental, no clear vocals"

    tags = ["vocals"]
    language_name = _infer_transcript_language(transcript_text)
    if language_name:
        tags.append(f"{language_name} vocals")

    effective_duration = max(float(audio_duration), 1.0)
    word_density = len(words) / effective_duration
    lexical_diversity = len(set(words)) / max(len(words), 1)

    if word_density >= 3.0:
        tags.extend(["fast vocals", "rap-like flow", "lyrical"])
    elif word_density >= 1.6:
        tags.extend(["sung vocals", "lyrical", "clear vocals"])
    else:
        tags.extend(["sparse vocals", "melodic vocals"])

    if lexical_diversity < 0.6:
        tags.append("repeated hook")
    if len(words) >= 40:
        tags.append("storytelling lyrics")
    if any(token in {"yeah", "oh", "la", "na", "hey", "woo"} for token in words):
        tags.append("hook vocals")

    return _clean_tags(", ".join(tags))


def _parse_acestep_transcription(result_text):
    """解析 ACE-Step 转录输出（Markdown 结构：# Languages / # Lyrics / [段落] 歌词行）。"""
    result_text = re.split(r"\n\s*(?:Human|User|Assistant)\s*:", result_text, maxsplit=1, flags=re.IGNORECASE)[0]
    result_text = re.sub(r"```[a-zA-Z]*\n?", "", result_text)
    result_text = result_text.replace("```", "").strip()

    language_match = re.search(r"#\s*Languages\s*\n+(.+?)(?=\n#|\Z)", result_text, flags=re.IGNORECASE | re.DOTALL)
    language_block = language_match.group(1).strip() if language_match else ""
    language_code = next((line.strip().lower() for line in language_block.splitlines() if line.strip()), "")

    lyrics_match = re.search(r"#\s*Lyrics\s*\n+(.+?)(?=\n#\s*[A-Za-z]|\Z)", result_text, flags=re.IGNORECASE | re.DOTALL)
    lyrics_block = lyrics_match.group(1).strip() if lyrics_match else result_text

    section_tags = []
    instrument_tags = []
    lyric_lines = []
    for line in lyrics_block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            inside = stripped[1:-1].strip()
            parts = [part.strip() for part in inside.split("-", 1)]
            section_name = re.sub(r"\s+\d+$", "", parts[0].lower())
            section_tags.append(section_name)
            if len(parts) > 1 and parts[1]:
                instrument_tags.append(parts[1].lower())
            continue
        lyric_lines.append(stripped)

    lyrics_text = "\n".join(lyric_lines).strip()
    return {
        "language_code": language_code,
        "lyrics_text": lyrics_text,
        "section_tags": section_tags,
        "instrument_tags": instrument_tags,
    }


def _derive_tags_from_acestep_transcription(result_text, audio_duration):
    """从 ACE-Step 转录结果推导结构/声乐/乐器标签。"""
    parsed = _parse_acestep_transcription(result_text)
    section_tags = parsed["section_tags"]
    instrument_tags = parsed["instrument_tags"]
    lyrics_text = parsed["lyrics_text"]
    language_code = parsed["language_code"]

    tags = []
    if lyrics_text:
        tags.extend([tag.strip() for tag in _derive_tags_from_transcript(lyrics_text, audio_duration).split(",") if tag.strip()])
    else:
        tags.extend(["instrumental", "no clear vocals"])

    language_name = _LANGUAGE_CODE_TO_NAME.get(language_code, language_code)
    if language_name:
        tags.append(f"{language_name} lyrics")

    normalized_sections = []
    for section in section_tags:
        cleaned = re.sub(r"\s+", " ", section).strip()
        normalized_sections.append(cleaned)

    if any("verse" in section for section in normalized_sections):
        tags.append("verse structure")
    if any("chorus" in section for section in normalized_sections):
        tags.append("chorus structure")
    if any("bridge" in section for section in normalized_sections):
        tags.append("bridge section")
    if any("intro" in section for section in normalized_sections):
        tags.append("intro section")
    if any("outro" in section for section in normalized_sections):
        tags.append("outro section")
    if any("spoken" in section for section in normalized_sections):
        tags.append("spoken section")
    if any("instrumental" in section or "interlude" in section for section in normalized_sections):
        tags.append("instrumental section")
    if normalized_sections and len(set(normalized_sections)) >= 2:
        tags.append("structured song form")

    for instrument in instrument_tags[:4]:
        instrument = re.sub(r"\s+", " ", instrument).strip()
        if instrument:
            tags.append(instrument)

    return _clean_tags(", ".join(tags))


# ===========================================================================
# 6. BPM / 调性检测（librosa 信号处理）
# ===========================================================================

def _detect_bpm_keyscale(audio_dict):
    """用 librosa 检测 BPM 和调性/音阶。"""
    try:
        import librosa
        import numpy as np
    except ImportError:
        return {"bpm": 0, "keyscale": ""}

    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]

    if waveform.dim() == 3:
        y = waveform[0].mean(dim=0)
    elif waveform.dim() == 2:
        y = waveform.mean(dim=0)
    else:
        y = waveform
    y = y.cpu().numpy().astype(np.float32)

    target_sr = 22050
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    detected_bpm = int(round(tempo))

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    pitch_names = ["C", "C#", "D", "D#", "E", "F",
                   "F#", "G", "G#", "A", "A#", "B"]

    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = chromagram.mean(axis=1)

    best_corr = -2.0
    best_key = "C"
    best_scale = "major"
    for i in range(12):
        maj_corr = float(np.corrcoef(chroma_vals, np.roll(major_profile, -i))[0, 1])
        min_corr = float(np.corrcoef(chroma_vals, np.roll(minor_profile, -i))[0, 1])
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = pitch_names[i]
            best_scale = "major"
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = pitch_names[i]
            best_scale = "minor"

    return {"bpm": detected_bpm, "keyscale": f"{best_key} {best_scale}"}


# ===========================================================================
# 7. 通用文本生成（歌词转录 / 音乐描述共用）
# ===========================================================================

# 给模型的默认指令（保持英文以保证各模型输出稳定，界面提示为中文）
_DEFAULT_TRANSCRIBE_PROMPT = (
    "Transcribe this song completely. Output ONLY the verbatim lyrics, "
    "organized by song sections (intro, verse, pre-chorus, chorus, bridge, outro). "
    "Do not add any commentary."
)

_DEFAULT_CAPTION_PROMPT = (
    "Describe this music for a text-to-music generation model. "
    "Mention genre and sub-genre, mood, tempo, key, main instruments, "
    "vocal characteristics (gender, timbre, delivery style), production style "
    "and overall arrangement energy. Write a flowing descriptive paragraph of 3-5 sentences."
)


def _generate_text(audio_dict, model_key, user_text, max_new_tokens, audio_duration,
                   use_flash_attn, gen_kwargs=None):
    """加载模型并执行一次自由文本生成，返回原始文本。

    支持 Qwen 系对话模型（ACE-Step / Qwen2.5-Omni / Ke-Omni / Qwen3-Omni）、
    Qwen2-Audio、MiDaShengLM 以及 Whisper ASR 变体。
    """
    if gen_kwargs is None:
        gen_kwargs = {}
    model, processor = _load_audio_model(model_key, use_flash_attn=use_flash_attn)
    y = _prepare_audio_mono(audio_dict, 16000, audio_duration)

    if model_key.startswith("MiDaShengLM"):
        tokenizer = _audio_tokenizer
        messages = [{"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "audio", "audio": y},
        ]}]
        mi = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, add_special_tokens=True, return_dict=True,
        )
        if not isinstance(mi, dict):
            from dataclasses import asdict
            mi = {k: v for k, v in asdict(mi).items() if v is not None}
        mi = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in mi.items()}
        gk = {"max_new_tokens": max_new_tokens}
        gk.update(gen_kwargs)
        with torch.inference_mode():
            generation = model.generate(**mi, **gk)
        input_ids = mi.get("input_ids")
        if input_ids is not None:
            generation = generation[:, input_ids.shape[-1]:]
        out = tokenizer.batch_decode(generation, skip_special_tokens=True)
        return out[0].strip() if out else ""

    if model_key == "Qwen2-Audio-7B-Instruct":
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "__PLACEHOLDER__"},
            {"type": "text", "text": user_text},
        ]}]
    elif _is_whisper_asr_model(model_key):
        # Whisper 不走对话模板，直接输入特征
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(model.device)
        if torch.is_floating_point(input_features):
            input_features = input_features.to(dtype=model.dtype)
        gk = {"max_new_tokens": max(64, min(max_new_tokens, 256))}
        if hasattr(getattr(model, "generation_config", None), "task_to_id"):
            gk["task"] = "transcribe"
        gk.update(gen_kwargs)
        with torch.inference_mode():
            generated_ids = model.generate(input_features=input_features, **gk)
        out = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return out[0].strip() if out else ""
    else:
        conversation = [{"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "audio", "audio": y, "sampling_rate": 16000},
        ]}]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text_prompt, audio=[y], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[-1]
    gk = {"max_new_tokens": max_new_tokens}
    gk.update(gen_kwargs)
    if hasattr(model, "talker"):
        gk.setdefault("return_audio", False)
        gk.setdefault("use_audio_in_video", True)
    with torch.inference_mode():
        text_ids = model.generate(**inputs, **gk)
    new_tokens = text_ids[:, input_len:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return raw[0].strip() if raw else ""


# ===========================================================================
# 8. 标签分类词表（供「音乐信息转JSON」做尽力分类）
# ===========================================================================

_GENRE_WORDS = [
    "pop", "mandopop", "cpop", "rock", "alternative rock", "punk", "metal",
    "jazz", "blues", "r&b", "soul", "hip hop", "rap", "electronic", "edm",
    "house", "techno", "trance", "dubstep", "drum and bass", "folk", "country",
    "classical", "orchestral", "reggae", "latin", "salsa", "funk", "disco",
    "indie", "ballad", "anime", "lo-fi", "ambient", "synthwave", "city pop",
]
_MOOD_WORDS = [
    "happy", "upbeat", "joyful", "sad", "melancholic", "melancholy", "romantic",
    "dreamy", "intimate", "aggressive", "dark", "bright", "energetic", "calm",
    "mellow", "groovy", "atmospheric", "nostalgic", "angry", "tense", "peaceful",
    "somber", "wistful", "playful", "triumphant", "epic", "hypnotic", "relaxed",
]
_INSTRUMENT_WORDS = [
    "piano", "electric piano", "guitar", "acoustic guitar", "electric guitar",
    "bass", "sub bass", "drums", "kick drum", "snare", "hi hat", "hi-hat",
    "cymbal", "percussion", "strings", "violin", "cello", "brass", "trumpet",
    "saxophone", "flute", "woodwind", "organ", "synth", "synth pad", "lead synth",
    "arpeggio", "harpsichord", "harp", "accordion", "choir", "vocals",
    "male vocals", "female vocals", "harmony", "backing vocals",
]
_STRUCTURE_WORDS = [
    "intro", "verse", "pre-chorus", "chorus", "bridge", "outro", "breakdown",
    "solo", "interlude", "drop", "hook", "coda", "instrumental break",
]


def _match_tags(tags, words):
    """返回标签文本中命中的词表条目（长条目优先，避免子串误匹配，去重保序）。"""
    remaining = tags.lower()
    matched = []
    for w in sorted(words, key=len, reverse=True):
        if w in remaining:
            matched.append(w)
            remaining = remaining.replace(w, " ")
    return matched


def _detect_vocal_gender(tags):
    """从标签文本尽力推断人声性别。"""
    text_lower = tags.lower()
    if any(w in text_lower for w in ("female", "woman", "girl", "femenino", "femme")):
        return "female"
    if any(w in text_lower for w in ("male", "man", "guy", "masculine", "baritone")):
        return "male"
    if "duet" in text_lower or "male and female" in text_lower:
        return "mixed"
    return ""


# ===========================================================================
# 9. 节点定义
# ===========================================================================

class MusicAnalyzer:
    """音乐分析器：提取标签、BPM、调性，输出音乐信息 JSON。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO", {"tooltip": "要分析的音频。"}),
                "提取标签": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "用选中的模型提取描述性标签（曲风/情绪/乐器/声乐等）。",
                }),
                "提取BPM": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "用 librosa 检测 BPM。",
                }),
                "提取调性": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "用 librosa 检测调性/音阶（如 G minor）。",
                }),
            },
            "optional": {
                "最大生成长度": ("INT", {
                    "default": 256, "min": 64, "max": 2000, "step": 16,
                    "tooltip": "标签生成的最大 token 数。",
                }),
                "音频时长": ("INT", {
                    "default": 60, "min": 10, "max": 300, "step": 5,
                    "tooltip": "最多分析的音频秒数（居中截取）。",
                }),
                "用后卸载模型": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "分析完成后卸载模型，释放显存。",
                }),
                "使用Flash注意力": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用 FlashAttention-2（需已安装 flash-attn，更快更省显存）。",
                }),
                "温度": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "采样温度，0 = 确定性输出。",
                }),
                "核采样top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "核采样，一般保持 1.0。",
                }),
                "top_k": ("INT", {
                    "default": 0, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Top-K 采样，0 = 不启用。",
                }),
                "重复惩罚": ("FLOAT", {
                    "default": 1.1, "min": 1.0, "max": 3.0, "step": 0.05,
                    "tooltip": "重复 token 惩罚，1.1 为温和的防循环。",
                }),
                "随机种子": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "采样种子。",
                }),
                "模型": (list(_ANALYSIS_MODELS.keys()), {
                    "default": _NATIVE_ANALYSIS_MODEL,
                    "tooltip": "用于标签提取的音频理解模型，需提前手动下载到本节点 models/ 目录。",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("标签", "BPM", "调性", "音乐信息JSON")
    FUNCTION = "analyze"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = (
        "分析音频并提取描述性标签、BPM 与调性/音阶。"
        "标签由选中的音频理解模型生成（默认 ACE-Step Transcriber，"
        "也支持 Qwen2-Audio、Qwen2.5-Omni、MiDaShengLM、Qwen3-Omni、Whisper、MERT、AST 等），"
        "BPM 与调性由 librosa 信号处理检测。"
    )

    def analyze(self, 音频, 提取标签=True, 提取BPM=True, 提取调性=True,
                最大生成长度=256, 音频时长=60, 用后卸载模型=True, 使用Flash注意力=False,
                温度=0.0, 核采样top_p=1.0, top_k=0, 重复惩罚=1.1, 随机种子=0,
                模型=_NATIVE_ANALYSIS_MODEL):
        tags = ""
        detected_bpm = 0
        keyscale = ""
        model_key = 模型

        torch.manual_seed(随机种子)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(随机种子)
        gen_kwargs = _build_gen_kwargs(温度, 核采样top_p, top_k, 重复惩罚, 随机种子)

        if 提取标签:
            try:
                tags = _extract_tags(音频, model_key, 最大生成长度, 音频时长,
                                     use_flash_attn=使用Flash注意力, gen_kwargs=gen_kwargs)
                print(f"[MusicAnalyzer] 提取到的标签: {tags}")
            except Exception as e:
                print(f"[MusicAnalyzer] 标签提取失败: {e}")

        if 提取BPM or 提取调性:
            try:
                dsp = _detect_bpm_keyscale(音频)
                if 提取BPM:
                    detected_bpm = dsp["bpm"]
                if 提取调性:
                    keyscale = dsp["keyscale"]
                print(f"[MusicAnalyzer] BPM: {dsp['bpm']} | 调性: {dsp['keyscale']}")
            except Exception as e:
                print(f"[MusicAnalyzer] librosa 检测失败: {e}")

        if 用后卸载模型 and 提取标签:
            _unload_audio_model()

        music_infos = json.dumps({
            "tags": tags,
            "bpm": f"{detected_bpm}bpm",
            "keyscale": keyscale,
        }, ensure_ascii=False, indent=4)

        return (tags, detected_bpm, keyscale, music_infos)


class MusicTranscriber:
    """歌词转录器：把音频转成逐字歌词文本。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO", {"tooltip": "要转录歌词的音频。"}),
            },
            "optional": {
                "模型": (list(_ANALYSIS_MODELS.keys()), {
                    "default": "ACE-Step-Transcriber",
                    "tooltip": "转录模型，Whisper 系列最省显存。需提前手动下载到本节点 models/ 目录。",
                }),
                "提示词": ("STRING", {
                    "default": _DEFAULT_TRANSCRIBE_PROMPT,
                    "multiline": True,
                    "tooltip": "发给模型的指令，可自行修改调整转录风格。",
                }),
                "最大生成长度": ("INT", {
                    "default": 512, "min": 64, "max": 4000, "step": 16,
                    "tooltip": "生成歌词的最大 token 数。",
                }),
                "音频时长": ("INT", {
                    "default": 60, "min": 10, "max": 300, "step": 5,
                    "tooltip": "最多分析的音频秒数（居中截取）。",
                }),
                "用后卸载模型": ("BOOLEAN", {"default": True}),
                "使用Flash注意力": ("BOOLEAN", {"default": False}),
                "温度": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "核采样top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
                "重复惩罚": ("FLOAT", {
                    "default": 1.1, "min": 1.0, "max": 3.0, "step": 0.05,
                }),
                "随机种子": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("歌词",)
    FUNCTION = "transcribe"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "把音频转录为逐字歌词。配合「音乐分析器」和「音乐信息转JSON」组成完整的音乐理解链路。"

    def transcribe(self, 音频, 模型="ACE-Step-Transcriber", 提示词=_DEFAULT_TRANSCRIBE_PROMPT,
                   最大生成长度=512, 音频时长=60, 用后卸载模型=True, 使用Flash注意力=False,
                   温度=0.0, 核采样top_p=1.0, top_k=0, 重复惩罚=1.1, 随机种子=0):
        torch.manual_seed(随机种子)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(随机种子)
        gen_kwargs = _build_gen_kwargs(温度, 核采样top_p, top_k, 重复惩罚, 随机种子)
        try:
            lyrics = _generate_text(音频, 模型, 提示词, 最大生成长度, 音频时长,
                                    使用Flash注意力, gen_kwargs)
            print(f"[MusicAnalyzer] 歌词转录: {lyrics[:400]}")
        except Exception as e:
            print(f"[MusicAnalyzer] 歌词转录失败: {e}")
            lyrics = ""
        if 用后卸载模型:
            _unload_audio_model()
        return (lyrics,)


class MusicCaptioner:
    """音乐描述器：生成适合喂给文生音乐模型的自然语言描述。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO", {"tooltip": "要描述的音频。"}),
            },
            "optional": {
                "模型": (list(_ANALYSIS_MODELS.keys()), {
                    "default": "ACE-Step-Transcriber",
                    "tooltip": "描述模型，MiDaShengLM-7B 的音乐描述质量最佳。需提前手动下载到本节点 models/ 目录。",
                }),
                "提示词": ("STRING", {
                    "default": _DEFAULT_CAPTION_PROMPT,
                    "multiline": True,
                    "tooltip": "发给模型的指令，可自行修改调整描述风格。",
                }),
                "最大生成长度": ("INT", {
                    "default": 256, "min": 64, "max": 2000, "step": 16,
                }),
                "音频时长": ("INT", {
                    "default": 60, "min": 10, "max": 300, "step": 5,
                }),
                "用后卸载模型": ("BOOLEAN", {"default": True}),
                "使用Flash注意力": ("BOOLEAN", {"default": False}),
                "温度": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "核采样top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
                "重复惩罚": ("FLOAT", {
                    "default": 1.1, "min": 1.0, "max": 3.0, "step": 0.05,
                }),
                "随机种子": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("描述",)
    FUNCTION = "caption"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "生成一段自然的音乐描述，适合直接喂给文生音乐模型（如 MiniMax Music 3）。"

    def caption(self, 音频, 模型="ACE-Step-Transcriber", 提示词=_DEFAULT_CAPTION_PROMPT,
                最大生成长度=256, 音频时长=60, 用后卸载模型=True, 使用Flash注意力=False,
                温度=0.0, 核采样top_p=1.0, top_k=0, 重复惩罚=1.1, 随机种子=0):
        torch.manual_seed(随机种子)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(随机种子)
        gen_kwargs = _build_gen_kwargs(温度, 核采样top_p, top_k, 重复惩罚, 随机种子)
        try:
            caption_text = _generate_text(音频, 模型, 提示词, 最大生成长度, 音频时长,
                                          使用Flash注意力, gen_kwargs)
            print(f"[MusicAnalyzer] 音乐描述: {caption_text[:400]}")
        except Exception as e:
            print(f"[MusicAnalyzer] 音乐描述失败: {e}")
            caption_text = ""
        if 用后卸载模型:
            _unload_audio_model()
        return (caption_text,)


class MusicInfoToJSON:
    """音乐信息转JSON：把歌词、标签、BPM、调性、描述汇总为一个结构化 JSON。

    标签会被尽力分类到 genre / mood / instruments / structure 字段，
    也可传入附加 JSON（例如 LLM 改写的结果）合并进输出。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "歌词": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "来自「歌词转录器」的歌词。",
                }),
                "标签": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "来自「音乐分析器」的逗号分隔标签。",
                }),
                "描述": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "来自「音乐描述器」的自然语言描述。",
                }),
                "BPM": ("INT", {
                    "default": 0, "forceInput": True,
                    "tooltip": "来自「音乐分析器」的 BPM。",
                }),
                "调性": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "来自「音乐分析器」的调性/音阶。",
                }),
                "附加JSON": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "可选的 JSON 对象，会合并进输出（例如 LLM 改写的结果）。",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("JSON",)
    FUNCTION = "to_json"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "把歌词、标签、BPM、调性、描述汇总为一个结构化 JSON，方便交给 LLM 或文生音乐后端。"

    def to_json(self, 歌词="", 标签="", 描述="", BPM=0, 调性="", 附加JSON=""):
        tag_list = [t.strip() for t in re.split(r"[,，、;；]", 标签) if t.strip()]
        info = {
            "lyrics": 歌词 or "",
            "bpm": int(BPM or 0),
            "key": 调性 or "",
            "tags": tag_list,
            "genre": _match_tags(标签, _GENRE_WORDS),
            "mood": _match_tags(标签, _MOOD_WORDS),
            "instruments": _match_tags(标签, _INSTRUMENT_WORDS),
            "structure": _match_tags(标签, _STRUCTURE_WORDS),
            "vocal": {
                "gender": _detect_vocal_gender(标签),
                "timbre": "",
                "style": "",
            },
            "caption": 描述 or "",
        }
        if 附加JSON and 附加JSON.strip():
            try:
                extra = json.loads(附加JSON)
                if isinstance(extra, dict):
                    info.update(extra)
            except Exception as e:
                print(f"[MusicAnalyzer] 附加JSON 解析失败，已忽略: {e}")
        return (json.dumps(info, ensure_ascii=False, indent=2),)


# ===========================================================================
# 10. 节点注册
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "MusicAnalyzer": MusicAnalyzer,
    "MusicTranscriber": MusicTranscriber,
    "MusicCaptioner": MusicCaptioner,
    "MusicInfoToJSON": MusicInfoToJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MusicAnalyzer": "音乐分析器",
    "MusicTranscriber": "歌词转录器",
    "MusicCaptioner": "音乐描述器",
    "MusicInfoToJSON": "音乐信息转JSON",
}

# -*- coding: utf-8 -*-
"""
ComfyUI-MusicAnalyzer —— 音乐理解与结构化描述节点（仅分析，无生成）

将一段音频转换为结构化的音乐信息：
  - 音乐分析器：标签 / BPM / 调性 / 音乐信息 JSON
  - 歌词转录器：逐字歌词
  - 音乐描述器：面向文生音乐模型的自然语言描述
  - 音乐信息转JSON：把各部分汇总为一个结构化 JSON

模型支持（需手动下载到 ComfyUI 官方共用目录 models/audio_encoders/<模型名>/，不会自动下载）：
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
    "MiDaShengLM-7B-FP8": "mispeech/midashenglm-7b-0804-fp8",
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


def _patch_qwen_omni_padding(model, processor):
    """补齐 Transformers 5.x 与 Qwen2.5-Omni talker 配置之间的兼容字段。

    部分 Qwen2.5-Omni 检查点的 talker_config 没有 pad_token_id，
    但新版 generate() 会无条件读取它，导致三类分析请求在生成前直接失败。
    """
    tokenizer = getattr(processor, "tokenizer", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        # Qwen tokenizer 的常见 EOS/PAD fallback；只在 tokenizer 未提供时使用。
        pad_id = 151643
    for config in (
        getattr(model, "config", None),
        getattr(model, "generation_config", None),
        getattr(getattr(model, "talker", None), "config", None),
        getattr(getattr(model, "talker", None), "generation_config", None),
    ):
        if config is not None:
            try:
                setattr(config, "pad_token_id", int(pad_id))
            except Exception:
                pass
    print(f"[MusicAnalyzer] Qwen2.5-Omni pad_token_id={pad_id}")


def _qwen_pad_id(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    return getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 151643


def _get_model_search_dirs():
    """模型查找目录列表：ComfyUI 官方 audio_encoders 目录优先，插件本地 models/ 回退。"""
    dirs = []
    try:
        import folder_paths
        # ComfyUI 官方注册的音频模型目录：models/audio_encoders/
        dirs.extend(folder_paths.get_folder_paths("audio_encoders"))
    except Exception:
        pass
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if local_dir not in dirs:
        dirs.append(local_dir)
    return dirs


def _find_model_dir(model_key):
    """在官方共用目录与插件本地目录中查找模型，返回第一个存在的目录。"""
    for base in _get_model_search_dirs():
        candidate = os.path.join(base, model_key)
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return None


def _check_model_local(model_key):
    """检查模型是否已手动下载（不做自动下载）。

    优先在 ComfyUI 官方共用目录 models/audio_encoders/<模型名>/ 查找，
    其次回退到插件本地 models/<模型名>/。
    """
    model_dir = _find_model_dir(model_key)
    if model_dir is not None:
        return model_dir
    repo_id = _ANALYSIS_MODELS[model_key]
    search_dirs = "\n  ".join(_get_model_search_dirs())
    raise RuntimeError(
        f"[MusicAnalyzer] 模型 {model_key} 未找到，已查找以下目录：\n"
        f"  {search_dirs}\n"
        f"请手动下载后放入其中任一目录（子目录名须为 {model_key}）：\n"
        f"下载命令（国内网络建议先执行 set HF_ENDPOINT=https://hf-mirror.com）：\n"
        f"  huggingface-cli download {repo_id} --local-dir \"<上面的任一目录>\\{model_key}\""
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


# 本地没有任何模型时下拉列表的占位项
_PLACEHOLDER_MODEL = "（请先下载模型到 models/audio_encoders/）"


def _get_local_model_keys():
    """只返回本地已下载（存在 config.json）的模型 key。

    下载新模型后，在 ComfyUI 里重新添加节点或刷新页面即可看到更新。
    """
    keys = [k for k in _ANALYSIS_MODELS if _find_model_dir(k) is not None]
    if not keys:
        keys = [_PLACEHOLDER_MODEL]
    return keys


def _model_default():
    """默认模型：优先 ACE-Step-Transcriber，本地没有则取列表第一个。"""
    keys = _get_local_model_keys()
    if _NATIVE_ANALYSIS_MODEL in keys:
        return _NATIVE_ANALYSIS_MODEL
    return keys[0]


def _is_placeholder(model_key):
    return model_key == _PLACEHOLDER_MODEL


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
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig
        omni_config = Qwen2_5OmniConfig.from_pretrained(model_dir)
        # Transformers 5.x expects this field while constructing the talker,
        # but older Qwen2.5-Omni checkpoints omit it from talker_config.
        if getattr(omni_config.talker_config, "pad_token_id", None) is None:
            # Talker vocab is 8448; the main Qwen tokenizer PAD (151643) is
            # outside that range and would fail nn.Embedding construction.
            omni_config.talker_config.pad_token_id = getattr(
                omni_config.talker_config, "tts_codec_pad_token_id", 8292
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Flash Attention 2 without specifying a torch dtype.*")
            warnings.filterwarnings("ignore", message=".*Token2WavModel.*fallback.*")
            _audio_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_dir, config=omni_config, **load_kwargs)
        _audio_model.disable_talker()
        _audio_model.eval()
        _audio_processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, use_fast=False)
        _patch_qwen_omni_padding(_audio_model, _audio_processor)
    elif model_key.startswith("Qwen2.5-Omni"):
        import warnings
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig
        omni_config = Qwen2_5OmniConfig.from_pretrained(model_dir)
        if getattr(omni_config.talker_config, "pad_token_id", None) is None:
            omni_config.talker_config.pad_token_id = getattr(
                omni_config.talker_config, "tts_codec_pad_token_id", 8292
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Flash Attention 2 without specifying a torch dtype.*")
            warnings.filterwarnings("ignore", message=".*Token2WavModel.*fallback.*")
            _audio_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_dir, config=omni_config, **load_kwargs)
        _audio_model.disable_talker()
        _audio_model.eval()
        _audio_processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, use_fast=False)
        _patch_qwen_omni_padding(_audio_model, _audio_processor)
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
        # BF16 版显式用 bf16；GPTQ / FP8 量化版不指定 dtype，交给量化配置决定
        midasheng_kwargs = dict(
            trust_remote_code=True,
            device_map=_get_analysis_device_map(),
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        if "GPTQ" not in model_key and "FP8" not in model_key:
            midasheng_kwargs["torch_dtype"] = torch.bfloat16
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Flash Attention.*")
            try:
                _audio_model = AutoModelForCausalLM.from_pretrained(model_dir, **midasheng_kwargs)
            except Exception as e:
                msg_lower = str(e).lower()
                if "gptq" in msg_lower or "quantization" in msg_lower or "auto_gptq" in msg_lower:
                    raise RuntimeError(
                        "[MusicAnalyzer] MiDaShengLM GPTQ 版本需要安装 auto-gptq（pip install auto-gptq），"
                        "或改用 MiDaShengLM-7B（BF16）版本。"
                    ) from e
                if "fp8" in msg_lower or "float8" in msg_lower:
                    raise RuntimeError(
                        "[MusicAnalyzer] MiDaShengLM FP8 版本加载失败（FP8 需要较新的 PyTorch/GPU 支持），"
                        "可改用 MiDaShengLM-7B（BF16）或 GPTQ 版本。"
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
    # Transformers 4.x uses return_audio; generation_mode is a Transformers 5
    # argument and is rejected by the Qwen2.5-Omni model on this environment.
    gk["return_audio"] = False
    gk["use_audio_in_video"] = True
    gk["pad_token_id"] = int(_qwen_pad_id(processor))
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
    gk["pad_token_id"] = int(_qwen_pad_id(processor))
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
    "in singing order with natural line breaks. Do not infer or add section labels, "
    "metadata, or commentary."
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
    if model_key.startswith("Qwen2.5-Omni") or _is_acestep_transcriber_model(model_key):
        # The plugin disables the talker to save VRAM; force text-only mode.
        gk.setdefault("return_audio", False)
        gk.setdefault("use_audio_in_video", True)
        gk.setdefault("pad_token_id", int(_qwen_pad_id(processor)))
    elif hasattr(model, "talker"):
        gk.setdefault("return_audio", False)
        gk.setdefault("use_audio_in_video", True)
        gk.setdefault("pad_token_id", int(_qwen_pad_id(processor)))
    with torch.inference_mode():
        text_ids = model.generate(**inputs, **gk)
    new_tokens = text_ids[:, input_len:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return raw[0].strip() if raw else ""


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
                    "default": 512, "min": 64, "max": 2000, "step": 16,
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
                "模型": (_get_local_model_keys(), {
                    "default": _model_default(),
                    "tooltip": "用于标签提取的音频理解模型（只显示本地已下载的，下载新模型后刷新页面或重建节点）。",
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
        if _is_placeholder(model_key):
            print("[MusicAnalyzer] 未检测到任何已下载模型，请先下载到 models/audio_encoders/ 目录。")

        torch.manual_seed(随机种子)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(随机种子)
        gen_kwargs = _build_gen_kwargs(温度, 核采样top_p, top_k, 重复惩罚, 随机种子)

        if 提取标签 and not _is_placeholder(model_key):
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
                "模型": (_get_local_model_keys(), {
                    "default": _model_default(),
                    "tooltip": "转录模型（只显示本地已下载的）。Whisper 系列最省显存。",
                }),
                "提示词": ("STRING", {
                    "default": _DEFAULT_TRANSCRIBE_PROMPT,
                    "multiline": True,
                    "tooltip": "发给模型的指令，可自行修改调整转录风格。",
                }),
                "最大生成长度": ("INT", {
                    "default": 2048, "min": 64, "max": 4000, "step": 16,
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
        if _is_placeholder(模型):
            print("[MusicAnalyzer] 未检测到任何已下载模型，请先下载到 models/audio_encoders/ 目录。")
            return ("",)
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
                "模型": (_get_local_model_keys(), {
                    "default": _model_default(),
                    "tooltip": "描述模型（只显示本地已下载的）。MiDaShengLM-7B 的音乐描述质量最佳。",
                }),
                "提示词": ("STRING", {
                    "default": _DEFAULT_CAPTION_PROMPT,
                    "multiline": True,
                    "tooltip": "发给模型的指令，可自行修改调整描述风格。",
                }),
                "最大生成长度": ("INT", {
                    "default": 768, "min": 64, "max": 2000, "step": 16,
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
        if _is_placeholder(模型):
            print("[MusicAnalyzer] 未检测到任何已下载模型，请先下载到 models/audio_encoders/ 目录。")
            return ("",)
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


class MusicInfoToMusic3:
    """音乐信息转Music3：把分析结果格式化为 MiniMax Music 3 需要的双输入。

    MiniMax Music 3 接收两个文本输入（分别进各自的 CLIPTextEncode）：
      1. 结构化描述（Structured Caption，按 [Genre]/[BPM]/[Key]/[Instruments]/[Arrangement] 组织）
      2. 分段歌词（带 [Verse]/[Chorus] 等段落标签）

    注意：JSON 不能直接进 CLIP，必须先经过本节点（或 LLM）二次转换为以上两种文本。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "歌词": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "来自「歌词转录器」的歌词。建议已带 [Verse]/[Chorus] 等段落标签。",
                }),
                "标签": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "来自「音乐分析器」的逗号分隔标签。",
                }),
                "描述": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "来自「音乐描述器」的自然语言描述，会写入 [Arrangement]。",
                }),
                "BPM": ("INT", {
                    "default": 0, "forceInput": True,
                    "tooltip": "来自「音乐分析器」的 BPM，会写入 [BPM]。",
                }),
                "调性": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "来自「音乐分析器」的调性/音阶，会写入 [Key]。",
                }),
                "风格改写": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "可选：填目标风格（如「改成赛博朋克摇滚」），将覆盖 [Genre]；留空则用分析出的曲风。",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("结构化描述", "歌词")
    FUNCTION = "to_music3"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "把分析结果格式化为 MiniMax Music 3 的 Structured Caption + 分段歌词双输入，输出直接接两个 CLIPTextEncode。"

    def to_music3(self, 歌词="", 标签="", 描述="", BPM=0, 调性="", 风格改写=""):
        lines = []
        if 风格改写 and 风格改写.strip():
            lines.append(f"[Genre] {风格改写.strip()}")
        else:
            genre = ", ".join(_match_tags(标签, _GENRE_WORDS))
            if genre:
                lines.append(f"[Genre] {genre}")
        if int(BPM or 0) > 0:
            lines.append(f"[BPM] {int(BPM)}")
        if 调性 and 调性.strip():
            lines.append(f"[Key] {调性.strip()}")
        instruments = ", ".join(_match_tags(标签, _INSTRUMENT_WORDS))
        if instruments:
            lines.append(f"[Instruments] {instruments}")
        if 描述 and 描述.strip():
            lines.append(f"[Arrangement] {描述.strip()}")
        caption = "\n".join(lines).strip()
        return (caption, 歌词 or "")


# 风格化输出中「结构化描述 / 歌词」两段的分隔标记
_REWRITE_SPLIT_MARKER = "<<<LYRICS>>>"

def _split_rewrite_output(text):
    """把 LLM 输出按标记拆成（结构化描述, 歌词）两段。"""
    if _REWRITE_SPLIT_MARKER in text:
        cap, lyr = text.split(_REWRITE_SPLIT_MARKER, 1)
        return cap.strip(), lyr.strip()
    return text.strip(), ""


class MusicInfoToLLMPrompt:
    """音乐信息转LLM指令：把分析结果 + 风格提示词组装成给 LLM 的完整指令文本。

    配合 ComfyUI 官方文本生成节点使用（如 TextGenerate / TextGenerateLTX2Prompt，
    输入 CLIP + prompt 即可让 CLIP（LLM 底座）直接输出文本 STRING）：
      本节点输出「指令文本」→ 接官方节点的 prompt 输入 → 官方节点生成风格化文本。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "结构化描述": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "来自「音乐信息转Music3」的结构化描述。",
                }),
                "歌词": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "来自「音乐信息转Music3」的歌词。",
                }),
                "风格提示词": ("STRING", {
                    "default": "改成赛博朋克摇滚风格，女声，更快节奏", "multiline": True,
                    "tooltip": "目标风格要求，例如：改成赛博朋克摇滚 / 换成男声 / 加快到 130 BPM。",
                }),
            },
            "optional": {
                "原始标签": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "（可选）来自「音乐分析器」的标签输出，把 omni 识别的完整原始信息也带给 LLM 参考。留空则不附加。",
                }),
                "附加信息": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "（可选）额外附加信息（如音乐信息JSON / 描述），留空则不附加。",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("指令文本",)
    FUNCTION = "build"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "把结构化描述 + 歌词 + 风格提示词（及可选的 omni 原始标签/附加信息）组装成一段完整指令文本，接 ComfyUI 官方 TextGenerate 节点（prompt 输入）生成风格化提示词。"

    def build(self, 结构化描述="", 歌词="", 风格提示词="", 原始标签="", 附加信息=""):
        extras = []
        if 原始标签 and 原始标签.strip():
            extras.append(f"Raw tags detected by the audio understanding model:\n{原始标签.strip()}")
        if 附加信息 and 附加信息.strip():
            extras.append(f"Additional info:\n{附加信息.strip()}")
        extra_block = ("\n\n" + "\n\n".join(extras)) if extras else ""
        text = (
            "You are a music producer preparing a MiniMax Music 3 remake.\n"
            "Rewrite ONLY the Structured Caption according to the style request. Do not output lyrics.\n"
            "Output exactly three labeled paragraphs and nothing else. This is a value-filling task: write the finished caption itself, using concrete source-specific musical facts and the requested transformation. Never copy the field names, their definitions, or the wording of this instruction as the answer, and never output placeholders.\n"
            "Required labels and content:\n"
            "Global Metadata: one complete sentence containing the actual genre/subgenre, BPM, key/scale, mood, use case, and production texture.\n"
            "Vocal Details: one complete sentence containing the actual lead vocal gender/register/timbre/delivery, harmonies, vocal effects, and vocal density.\n"
            "Arrangement: one complete sentence containing the actual instruments, rhythm section, section development, transitions, and ending.\n"
            "Hard rules:\n"
            "- Change only musical attributes explicitly requested by the user.\n"
            "- Preserve the source key unless a key change is explicitly requested.\n"
            "- If faster/slower is requested without a target BPM, adjust the source BPM moderately.\n"
            "- Vocal gender, register, and timbre must be internally consistent.\n"
            "- Preserve the source section sequence unless restructuring is explicitly requested.\n"
            "- Never output, translate, summarize, or rewrite the lyrics.\n"
            "- Never copy phrases such as 'genre, subgenre' or 'lead vocal gender/register'; replace them with actual musical content.\n"
            "- Forbidden output example: 'Global Metadata: genre, subgenre, BPM, key/scale...' or any sentence that merely lists the requested fields.\n"
            "- Required output example style: 'Global Metadata: warm lo-fi pop ballad, 82 BPM, D-flat major, intimate late-night mood, dusty tape texture.'\n"
            "- No title, preface, explanation, Markdown fence, or commentary.\n\n"
            f"Original structured caption:\n{结构化描述 or '(none)'}\n\n"
            "Lyrics are handled by a separate lossless path and are intentionally omitted here."
            f"{extra_block}\n\n"
            f"Style request:\n{风格提示词.strip() or '(keep original style)'}\n\n"
            "Now write the three completed paragraphs using the source facts above."
        )
        return (text,)


class MusicLyricsRepairPrompt:
    """生成歌词轻量纠错提示词，交给文本 LLM 处理。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "歌词": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
            "optional": {
                "纠错要求": ("STRING", {
                    "default": "修复明显的同音字、漏字和不通顺词语，使歌词语义自然。",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("纠错指令",)
    FUNCTION = "build"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "让文本模型轻量纠错歌词，并为翻唱主动设计适合 Music 3 的段落结构。"

    def build(self, 歌词="", 纠错要求="修复明显的同音字、漏字和不通顺词语，使歌词语义自然。"):
        text = (
            "You are a lyric editor and song-structure arranger for MiniMax Music 3.\n"
            "Treat the ASR transcript as unlabeled source lyrics for a new cover arrangement. Actively design a clear, musically useful section map even when the source contains no section tags.\n"
            "Freely choose section boundaries and assign [Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge], [Instrumental], and [Outro] according to lyric meaning, repeated hooks, emotional peaks, blank lines, and the song arc you infer. The cover arrangement does not have to preserve an unknown original section map.\n"
            "A memorable or emotionally central passage may be labeled [Chorus] even if it appears only once in a short or incomplete transcript. You may also split an unlabeled block into multiple sections when that produces a better cover structure.\n"
            "Use only the labels that improve the arrangement; not every label is required. Do not add fake lyric lines merely to fill a section.\n"
            "Correct obvious ASR homophones, missing characters, or semantically broken phrases when strongly supported, but do not translate, paraphrase, beautify, censor, or invent lyrics.\n"
            "Preserve every real lyric line and its order whenever possible; do not duplicate, omit, or merge lyric lines.\n"
            "Output ONLY valid MiniMax Music 3 lyrics, with no explanation, title, metadata, or Markdown fence.\n"
            "Section tags must be alone on a line and may use only: [Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge], [Instrumental], [Outro].\n"
            "Parenthetical backing vocals or sound cues are allowed only as standalone lines when present or strongly supported.\n"
            "When a word is uncertain, keep the original wording.\n\n"
            f"Correction focus: {纠错要求.strip() or 'minimal semantic correction'}\n\n"
            f"SOURCE LYRICS:\n{歌词 or '(empty)'}"
        )
        return (text,)


class Music3LyricsFormatter:
    """确定性清洗歌词，使 LLM 输出符合 Music 3 的分段文本格式。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"歌词": ("STRING", {"forceInput": True, "multiline": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("歌词",)
    FUNCTION = "format_lyrics"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "清理代码围栏和说明文字，规范 Music 3 段落标签；不改写歌词内容。"

    _TAGS = {
        "intro": "Intro", "verse": "Verse", "pre-chorus": "Pre-Chorus",
        "pre chorus": "Pre-Chorus", "prechorus": "Pre-Chorus",
        "chorus": "Chorus", "hook": "Chorus", "refrain": "Chorus",
        "bridge": "Bridge", "instrumental": "Instrumental", "interlude": "Instrumental",
        "solo": "Instrumental", "outro": "Outro",
    }

    def format_lyrics(self, 歌词=""):
        import re

        raw = str(歌词 or "").replace("\r\n", "\n").replace("\r", "\n")
        raw = re.sub(r"^\s*```(?:text|txt|lyrics)?\s*\n?", "", raw, flags=re.I)
        raw = re.sub(r"\n?\s*```\s*$", "", raw)
        lines = []
        saw_tag = False
        for source_line in raw.split("\n"):
            line = source_line.strip()
            if not line:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            line = line.replace("［", "[").replace("］", "]")
            match = re.match(r"^\[\s*([^\]]+)\s*\](.*)$", line)
            if match:
                key = re.sub(r"\s+", " ", match.group(1).strip().lower())
                tag = self._TAGS.get(key)
                if tag:
                    saw_tag = True
                    if lines and lines[-1] != "":
                        lines.append("")
                    lines.append(f"[{tag}]")
                    trailing = match.group(2).strip()
                    if trailing:
                        lines.append(trailing)
                    continue
                # Unknown bracket labels are not valid Music 3 section tags.
                # Keep any trailing lyric text, but remove the unsupported label itself.
                trailing = match.group(2).strip()
                if trailing:
                    lines.append(trailing)
                continue
            if not lines and re.match(r"^(lyrics?|歌词)\s*[:：]?$", line, flags=re.I):
                continue
            lines.append(line)

        while lines and lines[0] == "":
            lines.pop(0)
        while lines and lines[-1] == "":
            lines.pop()
        if not saw_tag and lines:
            lines.insert(0, "[Verse]")
        return ("\n".join(lines),)


class LyricsDurationEstimator:
    """歌词时长估算：以「基准时长」为中心，按歌词句数做温和修正。

    设计思路（以正常歌曲 2 分半为基准）：
      秒数 = 基准时长 + (实际歌词句数 - 参考句数) × 每句修正秒数
    歌词正好 24 句（一首歌的正常量）就用基准时长 150 秒；
    歌词多几句就多几秒、少几句就少几秒，并限制在 [最小时长, 最大时长] 内，
    不会因为歌词太短就生成几十秒的残歌，也不会因为歌词太长而无脑拉长。

    输出 FLOAT 秒数直接接 MiniMaxMusic3TextEncode 的 max_duration 输入；
    所有参数均可手动调节（想固定时长就把「每句修正秒数」调成 0）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "歌词": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "歌词文本（含 [Verse]/[Chorus] 等段落标签）。来自「音乐信息转Music3」或 LLM 接入后的歌词。",
                }),
            },
            "optional": {
                "基准时长": ("FLOAT", {
                    "default": 150.0, "min": 30.0, "max": 300.0, "step": 5.0,
                    "tooltip": "歌曲基准时长（秒），默认 150 秒 = 2 分半。歌词句数等于「参考句数」时就用这个值。",
                }),
                "参考句数": ("INT", {
                    "default": 24, "min": 1, "max": 200, "step": 1,
                    "tooltip": "正常歌曲的歌词句数，达到这个数就用基准时长。",
                }),
                "每句修正秒数": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "歌词每多/少一句，时长增减的秒数。调成 0 = 完全固定用基准时长。",
                }),
                "最小时长": ("FLOAT", {
                    "default": 90.0, "min": 30.0, "max": 300.0, "step": 5.0,
                    "tooltip": "结果下限（秒）。歌词再短也不会低于这个值。",
                }),
                "最大时长": ("FLOAT", {
                    "default": 240.0, "min": 60.0, "max": 600.0, "step": 10.0,
                    "tooltip": "结果上限（秒）。歌词再长也不会超过这个值（也受 Music 3 支持范围约束）。",
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("秒数", "整秒")
    FUNCTION = "estimate"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "以基准时长（默认150秒/2分半）为中心，按歌词句数温和修正生成时长（多句+秒、少句-秒，限制在上下限内），输出接 MiniMaxMusic3TextEncode 的 max_duration。所有参数可手动调节。"

    def estimate(self, 歌词="", 基准时长=150.0, 参考句数=24, 每句修正秒数=2.0,
                 最小时长=90.0, 最大时长=240.0):
        lines = (歌词 or "").splitlines()
        # 跳过空行与段落标签行（[Verse]、[Chorus] 等）
        lyric_lines = [ln.strip() for ln in lines
                       if ln.strip() and not ln.strip().startswith("[")]
        count = len(lyric_lines)
        seconds = 基准时长 + (count - 参考句数) * 每句修正秒数
        seconds = round(max(最小时长, min(最大时长, seconds)), 1)
        print(f"[MusicAnalyzer·歌词时长估算] 歌词 {count} 句（参考 {参考句数}）-> {seconds}s（基准 {基准时长}s，每句 ±{每句修正秒数}s，范围 {最小时长}-{最大时长}s）")
        return (seconds, int(seconds))


class AudioDuration:
    """读取输入音频真实时长，并限制到 Music 3 一条龙支持的范围。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO", {"forceInput": True}),
            },
            "optional": {
                "最小时长": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 600.0, "step": 1.0}),
                "最大时长": ("FLOAT", {"default": 240.0, "min": 1.0, "max": 600.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("秒数", "整秒")
    FUNCTION = "measure"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "从 LoadAudio 的 waveform 和 sample_rate 读取真实时长；默认限制为 60–240 秒并接入 Music3。"

    def measure(self, 音频, 最小时长=60.0, 最大时长=240.0):
        waveform = 音频.get("waveform") if isinstance(音频, dict) else None
        sample_rate = 音频.get("sample_rate") if isinstance(音频, dict) else None
        if waveform is None or not sample_rate:
            raise ValueError("[MusicAnalyzer] 无法从 AUDIO 输入读取 waveform/sample_rate。")
        samples = int(waveform.shape[-1])
        actual = samples / float(sample_rate)
        lower = min(float(最小时长), float(最大时长))
        upper = max(float(最小时长), float(最大时长))
        seconds = round(max(lower, min(upper, actual)), 1)
        print(f"[MusicAnalyzer·自动时长] 音频实际 {actual:.2f}s -> Music3 时长 {seconds:.1f}s（范围 {lower:.1f}-{upper:.1f}s）")
        return (seconds, int(round(seconds)))


class AnalysisOverview:
    """分析结果总览：把 omni 识别出的全部信息汇总成一段易读文本。

    输入来自「音乐分析器 / 歌词转录器 / 音乐描述器」的输出（均可选），
    汇总后打印到控制台并原样输出，让你一眼看到模型到底识别出了什么。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "标签": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "来自「音乐分析器」的标签输出。",
                }),
                "BPM": ("INT", {
                    "default": 0, "forceInput": True,
                    "tooltip": "来自「音乐分析器」的 BPM。",
                }),
                "调性": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "来自「音乐分析器」的调性（如 G minor）。",
                }),
                "歌词": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "来自「歌词转录器」的歌词。",
                }),
                "描述": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "来自「音乐描述器」的描述。",
                }),
                "音乐信息JSON": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "来自「音乐信息转JSON」的结构化 JSON（可选）。",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("总览文本",)
    FUNCTION = "overview"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "把 omni 识别出的全部信息（标签/BPM/调性/歌词/描述/JSON）汇总成一段易读文本并打印到控制台，方便检查识别质量。"

    def overview(self, 标签="", BPM=0, 调性="", 歌词="", 描述="", 音乐信息JSON=""):
        parts = []
        parts.append("【音乐分析结果·omni 识别】")
        parts.append(f"标签: {标签 or '(空)'}")
        parts.append(f"BPM: {BPM if BPM else '(未检测)'}")
        parts.append(f"调性: {调性 or '(未检测)'}")
        if 歌词 and 歌词.strip():
            parts.append(f"\n歌词:\n{歌词.strip()}")
        if 描述 and 描述.strip():
            parts.append(f"\n描述:\n{描述.strip()}")
        if 音乐信息JSON and 音乐信息JSON.strip():
            parts.append(f"\n音乐信息JSON:\n{音乐信息JSON.strip()}")
        text = "\n".join(parts)
        print(f"[MusicAnalyzer·分析结果总览]\n{text}\n{'=' * 60}")
        return (text,)


class MusicLLMToMusic3:
    """LLM输出接入Music3：把官方文本生成节点输出的文本按 <<<LYRICS>>> 拆成 caption/lyrics 两段。

    输入是 ComfyUI 官方 TextGenerate / TextGenerateLTX2Prompt 等节点的 generated_text 输出，
    拆分失败时回退到兜底输入（可接「音乐信息转Music3」的输出）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LLM输出文本": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "官方 TextGenerate / TextGenerateLTX2Prompt 的 generated_text 输出。",
                }),
            },
            "optional": {
                "兜底描述": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "LLM 输出无法拆分时回退的结构化描述（可接「音乐信息转Music3」）。",
                }),
                "兜底歌词": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "LLM 输出无法拆分时回退的歌词。",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("caption", "lyrics")
    FUNCTION = "connect"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "把官方文本生成节点的输出按 <<<LYRICS>>> 拆成 caption/lyrics 两段，直接接 MiniMaxMusic3TextEncode 输入。"

    def connect(self, LLM输出文本="", 兜底描述="", 兜底歌词=""):
        cap, lyr = _split_rewrite_output(LLM输出文本 or "")
        if not cap:
            cap = 兜底描述 or ""
        if not lyr:
            lyr = 兜底歌词 or ""
        return (cap, lyr)


class Music3PromptAdapter:
    """音乐信息接入Music3：把两段提示词接到 MiniMax Music 3 的输入。

    输入的两段文本（结构化描述 + 歌词）可以来自「音乐信息转Music3」的格式化输出，
    也可以来自其他 CLIP（本地 LLM 底座）参考风格提示转写的结果。
    输出端口命名 caption / lyrics，与官方 MiniMaxMusic3TextEncode（或官方工作流
    子图节点）的输入一一对应，直接连线即可，生成部分无需再管。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "结构化描述": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "第一段：结构化描述（对应 caption）。可来自「音乐信息转Music3」，或 CLIP/LLM 参考风格提示转写的结果。",
                }),
                "歌词": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "第二段：歌词，含 [Verse]/[Chorus] 等段落标签（对应 lyrics）。",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("caption", "lyrics")
    FUNCTION = "connect"
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "把两段提示词（结构化描述 + 歌词）原样透传，输出端口命名为 caption/lyrics，直接连线到 MiniMax Music 3 的对应输入。"

    def connect(self, 结构化描述="", 歌词=""):
        return (结构化描述 or "", 歌词 or "")


class TextPreview:
    """文本预览：把链路中任意一段文本原样透传，同时打印到控制台。

    可插在任意 STRING 链路的中间（输入输出都是 STRING，不影响连线），
    用于查看「音乐信息转LLM指令」生成的指令、LLM 扩写结果、或最终
    caption/lyrics 的内容；输出也可直接接显示节点查看。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "音频/音乐分析"
    DESCRIPTION = "把输入的文本原样透传并打印到控制台，用于预览链路中的提示词/歌词等文本内容。可串在任意 STRING 链路中间。"

    def preview(self, text, unique_id=None, extra_pnginfo=None):
        if isinstance(text, list):
            text = "\n".join(str(item) for item in text)
        text = str(text or "")
        print(f"[MusicAnalyzer·文本预览]\n{text}\n{'=' * 50}")
        return {"ui": {"text": text}, "result": (text,)}


# ===========================================================================
# 10. 节点注册
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "MusicAnalyzer": MusicAnalyzer,
    "MusicTranscriber": MusicTranscriber,
    "MusicCaptioner": MusicCaptioner,
    "MusicInfoToJSON": MusicInfoToJSON,
    "MusicInfoToMusic3": MusicInfoToMusic3,
    "MusicInfoToLLMPrompt": MusicInfoToLLMPrompt,
    "MusicLyricsRepairPrompt": MusicLyricsRepairPrompt,
    "Music3LyricsFormatter": Music3LyricsFormatter,
    "MusicLLMToMusic3": MusicLLMToMusic3,
    "Music3PromptAdapter": Music3PromptAdapter,
    "LyricsDurationEstimator": LyricsDurationEstimator,
    "AudioDuration": AudioDuration,
    "AnalysisOverview": AnalysisOverview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MusicAnalyzer": "音乐分析器",
    "MusicTranscriber": "歌词转录器",
    "MusicCaptioner": "音乐描述器",
    "MusicInfoToJSON": "音乐信息转JSON",
    "MusicInfoToMusic3": "音乐信息转Music3",
    "MusicInfoToLLMPrompt": "音乐信息转LLM指令",
    "MusicLyricsRepairPrompt": "歌词语义纠错指令",
    "Music3LyricsFormatter": "Music3歌词格式化",
    "MusicLLMToMusic3": "LLM输出接入Music3",
    "Music3PromptAdapter": "音乐信息接入Music3",
    "LyricsDurationEstimator": "歌词时长估算",
    "AudioDuration": "音频自动时长",
    "AnalysisOverview": "分析结果总览",
}

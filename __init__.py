"""
ComfyUI-MusicAnalyzer —— 音乐理解与结构化描述节点包

专注一件事：把一段音频转成结构化的音乐信息（歌词、BPM、调性、曲风、
情绪、乐器、段落结构、自然语言描述），输出统一的 JSON，供 LLM 改写或
接文本生音乐后端（如 MiniMax Music 3）使用。

本包只包含「分析」能力，不含任何音频生成逻辑。

音频理解核心代码提取并重构自 ComfyUI-AceStep_SFT（MIT 协议），
并新增了 MiDaShengLM / Qwen3-Omni 支持与结构化 JSON 输出节点。

模型存放（ComfyUI 官方共用目录）：ComfyUI/models/audio_encoders/<模型名>/
"""

import os

import folder_paths

# 使用 ComfyUI 官方注册的 audio_encoders 目录（models/audio_encoders/）。
# 老版本 ComfyUI 若尚未注册，则补注册一次（幂等，路径相同，其他插件同样可共用）。
if not folder_paths.get_folder_paths("audio_encoders"):
    folder_paths.add_model_folder_path(
        "audio_encoders", os.path.join(folder_paths.models_dir, "audio_encoders")
    )

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

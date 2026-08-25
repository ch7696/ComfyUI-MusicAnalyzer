# ComfyUI-MusicAnalyzer

音乐理解与结构化描述节点包 —— 把一段音频变成一份**结构化的音乐信息**（歌词、BPM、调性、曲风、情绪、乐器、段落结构、自然语言描述），输出统一 JSON，方便交给 LLM 改写，或直接喂给文生音乐模型（如 MiniMax Music 3）。

> 只做「分析」，不含任何生成逻辑。
> 音频理解核心代码提取并重构自 [ComfyUI-AceStep_SFT](https://github.com/ACE-Step/ComfyUI-AceStep_SFT)（MIT 协议），并新增 MiDaShengLM / Qwen3-Omni 支持与结构化 JSON 节点。本仓库同样以 MIT 协议开源。

## 节点一览

| 节点 | 输出 | 作用 |
|---|---|---|
| **音乐分析器** (MusicAnalyzer) | 标签 / BPM / 调性 / 音乐信息JSON | 提取描述性标签（曲风、情绪、乐器、声乐），librosa 检测 BPM 和调性 |
| **歌词转录器** (MusicTranscriber) | 歌词 | 逐字歌词转录，按段落组织 |
| **音乐描述器** (MusicCaptioner) | 描述 | 生成适合喂给文生音乐模型的自然语言描述 |
| **音乐信息转JSON** (MusicInfoToJSON) | JSON | 汇总以上结果，输出结构化 JSON，并把标签尽力分类到 genre/mood/instruments/structure 字段 |

## 安装

把本目录放到 ComfyUI 的 `custom_nodes/` 下：

```bash
git clone https://your-host/ComfyUI-MusicAnalyzer.git ComfyUI/custom_nodes/ComfyUI-MusicAnalyzer
cd ComfyUI/custom_nodes/ComfyUI-MusicAnalyzer
pip install -r requirements.txt
```

重启 ComfyUI 后，在节点菜单的 `音频/音乐分析` 分类下即可找到四个节点。

## 模型支持

首次使用某个模型时，会自动从 HuggingFace 下载到本节点的 `models/` 目录（国内网络建议先设置 `HF_ENDPOINT=https://hf-mirror.com`）。

| 模型 | 用途 | 显存参考 |
|---|---|---|
| ACE-Step-Transcriber（默认） | 歌词/结构/声乐，全能 | ~7GB（bf16） |
| Qwen2.5-Omni-3B | 通用理解，均衡 | ~7GB |
| Ke-Omni-R-3B | 通用理解，推理快 | ~7GB |
| MiDaShengLM-7B | **音乐描述质量最佳**（MusicCaps 59.7 FENSE） | ~16GB（bf16）/ ~5GB（GPTQ） |
| Qwen3-Omni-8B | 最强通用音频理解 | ~16GB，需 transformers≥4.53 |
| Qwen2-Audio-7B-Instruct | 通用音频问答 | ~16GB |
| Whisper large-v3 (turbo/distil) | 纯歌词转录，最省显存 | ~3GB |
| MERT-v1-330M / AST-AudioSet | 特征嵌入/事件分类，快 | <1GB |

> 16GB 显存（如 RTX 5060 Ti）建议主力用 3B 档模型；想用 MiDaShengLM 建议选 GPTQ 4bit 版（需 `pip install auto-gptq`）。

## 典型工作流

```
LoadAudio (VHS)
   │
   ├──► 音乐分析器 ──► 标签/BPM/调性
   │
   ├──► 歌词转录器 ──► 歌词
   │
   └──► 音乐描述器 ──► 描述
                         │
                         ▼
                 音乐信息转JSON ──► 结构化 JSON
                        │
                        ▼
                LLM 改写（曲风迁移）→ 文生音乐模型
```

`example_workflows/` 目录下提供了可直接导入的示例工作流。

## 输出示例

```json
{
  "lyrics": "…完整歌词…",
  "bpm": 92,
  "key": "G minor",
  "tags": ["piano", "female vocals", "melancholic", "strings", "92bpm", "romantic"],
  "genre": ["pop", "ballad"],
  "mood": ["melancholic", "romantic"],
  "instruments": ["piano", "strings", "female vocals"],
  "structure": ["intro", "verse", "pre-chorus", "chorus", "bridge", "outro"],
  "vocal": {"gender": "female", "timbre": "", "style": ""},
  "caption": "…自然语言描述…"
}
```

## 常见问题

- **下载很慢 / 失败**：设置环境变量 `HF_ENDPOINT=https://hf-mirror.com` 后重启 ComfyUI。
- **显存不足**：改用 3B 档模型；`音频时长` 调小；用后卸载模型保持开启。
- **Qwen3-Omni 报错**：`pip install -U transformers`（需要 ≥4.53）。
- **MiDaShengLM GPTQ 报错**：`pip install auto-gptq`，或改用 BF16 版。

## 协议

MIT License，详见 [LICENSE](LICENSE)。音频理解核心逻辑源自 [ComfyUI-AceStep_SFT](https://github.com/ACE-Step/ComfyUI-AceStep_SFT)（MIT）。

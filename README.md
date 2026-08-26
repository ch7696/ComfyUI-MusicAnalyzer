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
| **音乐信息转Music3** (MusicInfoToMusic3) | 结构化描述 / 歌词 | 把分析结果格式化为 MiniMax Music 3 的 Structured Caption + 分段歌词双输入 |
| **音乐信息转LLM指令** (MusicInfoToLLMPrompt) | 指令文本 | 把结构化描述 + 歌词 + 你的风格提示词组装成一段给 LLM 的改写指令（零推理，纯文本组装）；可选接入「音乐分析器」的原始标签让 LLM 参考更全的 omni 识别信息 |
| **音乐LLM输出接入** (MusicLLMToMusic3) | caption / lyrics | 把 LLM（你的 CLIP/TextGenerate 等生成的文本）输出按 `<<<LYRICS>>>` 标记拆成两段；LLM 输出异常时自动用原分析结果兜底 |
| **音乐信息接入Music3** (Music3PromptAdapter) | caption / lyrics | 把两段提示词透传，端口命名与 Music 3 输入一致，直接连线 |
| **文本预览** (TextPreview) | 文本 | 把链路中任意 STRING 原样透传并打印到控制台，方便查看「转LLM指令」生成的指令、LLM 扩写结果、caption/lyrics 等内容（可串在链路中间） |
| **歌词时长估算** (LyricsDurationEstimator) | 秒数 / 整秒 | 以基准时长（默认150秒/2分半）为中心，按歌词句数温和修正（多句+秒、少句-秒，限制在上下限内），输出直接接 `MiniMaxMusic3TextEncode.max_duration`；想固定时长就把每句修正调成0 |
| **分析结果总览** (AnalysisOverview) | 总览文本 | 把 omni 识别出的全部信息（标签/BPM/调性/歌词/描述/JSON）汇总成一段易读文本并打印到控制台，一眼看清识别质量 |

> 节点里所有「模型」下拉框**只显示本地已下载的模型**。下载新模型后，在 ComfyUI 里重新添加节点或刷新页面即可看到更新。

## 一键翻唱完整流程

```
LoadAudio
   │
   ├──► 音乐分析器 ──── 标签 / BPM / 调性
   ├──► 歌词转录器 ──── 歌词
   └──► 音乐描述器 ──── 描述
                          │
                          ▼
                   音乐信息转Music3 ──► 结构化描述 + 歌词
                          │
                          ▼
             音乐信息转LLM指令 ◄── 风格提示词（"改成赛博朋克摇滚"）
                          │
                          ▼
        CLIPLoader(你的LLM底座CLIP) + TextGenerate（官方节点，CLIP 直接生成文本）
                          │
                          ▼
                   音乐LLM输出接入 ◄── 原分析结果（兜底）
                          │
                          ▼
             音乐信息接入Music3 ──► caption / lyrics
                          │
                          ▼
   MiniMaxMusic3TextEncode → KSampler → VAEDecode(可选 tiled) → SaveAudio
```

> 「音乐信息转LLM指令」只做文本组装（零推理）：把分析出的结构化描述、歌词和你填的风格提示词拼成一段改写指令，交给你的 CLIP（如 `TextGenerate` 官方节点，CLIP 直接输出文本而非 condition）。「音乐LLM输出接入」再把 LLM 生成的文本按 `<<<LYRICS>>>` 标记拆回 caption / lyrics 两段——LLM 没输出或输出异常时自动回退到原分析结果，流程不会断。

`example_workflows/` 提供两个现成示例：
- `翻唱流程_分析到改写.json` —— 分析 → 格式化 → 转LLM指令 → CLIP/TextGenerate → LLM接入（输出 caption/lyrics，接你的 Music 3 工作流）
- `翻唱全流程_分析到Music3.json` —— **完整一键翻唱**：分析 + LLM 风格化 + Music 3 生成（UNETLoader/CLIPLoader/VAELoader/KSampler/解码/SaveAudio 已全部接好，含 tiled 低显存开关）

## 安装

把本目录放到 ComfyUI 的 `custom_nodes/` 下：

```bash
git clone https://your-host/ComfyUI-MusicAnalyzer.git ComfyUI/custom_nodes/ComfyUI-MusicAnalyzer
cd ComfyUI/custom_nodes/ComfyUI-MusicAnalyzer
pip install -r requirements.txt
```

重启 ComfyUI 后，在节点菜单的 `音频/音乐分析` 分类下即可找到四个节点。

## 模型支持（手动下载，不自动下载）

节点**不会**自动下载任何模型。需要先把模型放到 ComfyUI **官方共用目录** `ComfyUI/models/audio_encoders/<模型名>/`（目录内须有 `config.json`）——这是 ComfyUI 官方注册的音频模型目录，其他插件也可以共用。

下载命令示例（国内网络建议先执行 `set HF_ENDPOINT=https://hf-mirror.com`）：

```bash
# 例：下载默认模型 ACE-Step-Transcriber
huggingface-cli download ACE-Step/acestep-transcriber --local-dir "ComfyUI/models/audio_encoders/ACE-Step-Transcriber"
```

也可以用仓库根目录的 `download_models.bat` 一键下载推荐模型（脚本同样下载到该官方目录）。

| 模型 | 仓库 ID | 用途 | 显存参考 |
|---|---|---|---|
| ACE-Step-Transcriber（默认） | `ACE-Step/acestep-transcriber` | 歌词/结构/声乐，全能 | ~7GB（bf16） |
| Qwen2.5-Omni-3B | `Qwen/Qwen2.5-Omni-3B` | 通用理解，均衡 | ~7GB |
| Ke-Omni-R-3B | `KE-Team/Ke-Omni-R-3B` | 通用理解，推理快 | ~7GB |
| MiDaShengLM-7B | `mispeech/midashenglm-7b-0804-bf16` | **音乐描述质量最佳**（MusicCaps 59.7 FENSE） | ~16GB（bf16）/ ~5GB（GPTQ）/ ~8GB（FP8） |
| MiDaShengLM-7B-GPTQ | `mispeech/midashenglm-7b-0804-w4a16-gptq` | 音乐描述，4bit 量化，省显存 | ~5GB，需 `pip install auto-gptq` |
| MiDaShengLM-7B-FP8 | `mispeech/midashenglm-7b-0804-fp8` | 音乐描述，FP8 量化 | ~8GB，需较新 PyTorch/GPU |
| Qwen3-Omni-8B | `Qwen/Qwen3-Omni-8B` | 最强通用音频理解 | ~16GB，需 transformers≥4.53 |
| Qwen2-Audio-7B-Instruct | `Qwen/Qwen2-Audio-7B-Instruct` | 通用音频问答 | ~16GB |
| Whisper large-v3 (turbo/distil) | `openai/whisper-large-v3` 等 | 纯歌词转录，最省显存 | ~3GB |
| MERT-v1-330M / AST-AudioSet | `m-a-p/MERT-v1-330M` / `MIT/ast-...` | 特征嵌入/事件分类，快 | <1GB |

> 16GB 显存（如 RTX 5060 Ti）建议主力用 3B 档模型；想用 MiDaShengLM 建议选 GPTQ 4bit 版（需 `pip install auto-gptq`）。

## 模型推荐（纯分析用途）

| 环节 | 推荐 | 理由 |
|---|---|---|
| 歌词转录 + 结构 + 声乐（默认主力） | **ACE-Step-Transcriber** | 全能、零额外依赖、~7GB 显存舒适，中文歌词支持好 |
| 音乐描述（喂给文生音乐模型） | **MiDaShengLM-7B**（能装 auto-gptq 就用 GPTQ 版） | 开源音乐描述最强（MusicCaps 59.7 FENSE，超 Qwen2.5-Omni-7B） |
| BPM / 调性 | 无需模型（librosa 内置） | 不占显存 |
| 轻量纯歌词 | Whisper-large-v3-turbo | ~3GB，速度快 |

一句话：**ACE-Step-Transcriber 当默认主力；要冲描述质量就加 MiDaShengLM-7B。**

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

- **模型在哪下载？** 本仓库不做自动下载。先执行 `set HF_ENDPOINT=https://hf-mirror.com`（国内网络），再用 `huggingface-cli download <仓库ID> --local-dir "ComfyUI/models/audio_encoders/<模型名>"` 下载，详见上文模型支持表。
- **显存不足**：改用 3B 档模型；`音频时长` 调小；用后卸载模型保持开启。
- **Qwen3-Omni 报错**：`pip install -U transformers`（需要 ≥4.53）。
- **MiDaShengLM GPTQ 报错**：`pip install auto-gptq`，或改用 BF16 版。

## 协议

MIT License，详见 [LICENSE](LICENSE)。音频理解核心逻辑源自 [ComfyUI-AceStep_SFT](https://github.com/ACE-Step/ComfyUI-AceStep_SFT)（MIT）。

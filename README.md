# ComfyUI-MusicAnalyzer

<p align="center">
  <img src="assets/logo.png" alt="野茶柿 Studio 路边E条" width="360">
</p>

<p align="center">
  <strong>ComfyUI 音频理解与音乐工作流扩展</strong><br>
  面向翻唱分析、结构化描述和 MiniMax Music 3 提示词编排。
</p>

ComfyUI-MusicAnalyzer 将音频解析为统一的结构化信息，包括**歌词、BPM、调性、曲风、情绪、乐器、声乐特征和段落结构**，并提供 JSON、自然语言描述以及 MiniMax Music 3 Structured Caption / 分段歌词接口。

本项目聚焦音频分析和文本处理，不包含音频生成模型；生成环节由 ComfyUI 中已安装的 Music 3 或其他生成节点完成。音频理解核心代码提取并重构自 [ComfyUI-AceStep_SFT](https://github.com/ACE-Step/ComfyUI-AceStep_SFT)（MIT 协议），并扩展 MiDaShengLM、Qwen3-Omni 及结构化 JSON 支持。

## 节点一览

| 节点 | 输出 | 作用 |
|---|---|---|
| **音乐分析器** (MusicAnalyzer) | 标签 / BPM / 调性 / 音乐信息 JSON | 提取曲风、情绪、乐器和声乐等描述性标签，并使用 librosa 分析 BPM 与调性 |
| **歌词转录器** (MusicTranscriber) | 歌词 | 转录歌词并按音乐段落组织文本 |
| **音乐描述器** (MusicCaptioner) | 描述 | 生成面向文生音乐模型的自然语言音乐描述 |
| **音乐信息转 JSON** (MusicInfoToJSON) | JSON | 汇总分析结果，并按 genre、mood、instruments、structure 等字段输出 |
| **音乐信息转 Music3** (MusicInfoToMusic3) | 结构化描述 / 歌词 | 将分析结果转换为 MiniMax Music 3 的 Structured Caption 与分段歌词 |
| **音乐信息转 LLM 指令** (MusicInfoToLLMPrompt) | 指令文本 | 根据结构化描述、歌词和风格要求生成 LLM 改写指令；节点本身不执行推理 |
| **音乐 LLM 输出接入** (MusicLLMToMusic3) | caption / lyrics | 按 `<<<LYRICS>>>` 标记解析 LLM 输出，并在格式异常时保留原始分析结果 |
| **音乐信息接入 Music3** (Music3PromptAdapter) | caption / lyrics | 将 caption 与 lyrics 以 Music 3 输入格式透传 |
| **文本预览** (TextPreview) | 文本 | 在界面和控制台查看链路中的文本，并以单一 STRING 继续传递 |
| **歌词时长估算** (LyricsDurationEstimator) | 秒数 / 整秒 | 根据歌词句数估算建议时长，可直接连接 `MiniMaxMusic3TextEncode.max_duration` |
| **分析结果总览** (AnalysisOverview) | 总览文本 | 汇总标签、BPM、调性、歌词、描述及 JSON，便于检查分析结果 |

> 节点中的模型列表仅展示当前 ComfyUI 模型目录中可用的模型。新增模型后，请刷新界面或重新加载节点。

## 翻唱工作流参考

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
             音乐信息转 LLM 指令 ◄── 风格转换要求
                          │
                          ▼
        CLIPLoader + TextGenerate（文本生成节点）
                          │
                          ▼
                   音乐 LLM 输出接入 ◄── 原分析结果
                          │
                          ▼
             音乐信息接入Music3 ──► caption / lyrics
                          │
                          ▼
   MiniMaxMusic3TextEncode → KSampler → VAEDecode(可选 tiled) → SaveAudio
```

> 「音乐信息转 LLM 指令」仅负责文本组装，将结构化描述、歌词和风格要求交给工作流中的文本生成节点（例如官方 `TextGenerate`）。「音乐 LLM 输出接入」再依据 `<<<LYRICS>>>` 标记拆分 caption 与 lyrics；格式不符合要求时自动保留原分析结果。

`example_workflows/` 提供两个参考工作流：
- `翻唱流程_分析到改写.json` —— 分析 → 格式化 → LLM 指令 → 文本生成 → 输出 caption/lyrics
- `翻唱全流程_分析到Music3.json` —— 分析、LLM 风格化及 Music 3 生成的完整示例，包含 UNETLoader、CLIPLoader、VAELoader、KSampler、解码和 SaveAudio 节点

## 安装

将本仓库安装至 ComfyUI 的 `custom_nodes/` 目录：

```bash
git clone https://github.com/ch7696/ComfyUI-MusicAnalyzer.git ComfyUI/custom_nodes/ComfyUI-MusicAnalyzer
cd ComfyUI/custom_nodes/ComfyUI-MusicAnalyzer
python -m pip install -r requirements.txt
```

重启 ComfyUI 后，可在节点菜单的 `音频/音乐分析` 分类中使用本扩展的全部节点。

## 模型准备与支持

模型文件由使用者另行准备，本扩展不执行自动下载。模型应放置于 ComfyUI 的共享目录 `ComfyUI/models/audio_encoders/<模型名>/`，目录中须包含 `config.json` 及完整的权重文件（含所有分片）。该目录可供其他 ComfyUI 扩展复用。

对于共享盘或集中式模型仓库，可将模型目录软链接至上述路径。软链接目标必须保持可读，并包含模型的完整文件集合。例如 Linux：

```bash
ln -s /shared/models/audio_encoders/ACE-Step-Transcriber \
  ComfyUI/models/audio_encoders/ACE-Step-Transcriber
```

分片模型还应同时具备 `model.safetensors.index.json` 及其列出的全部
`model-*.safetensors` 文件。Windows 可使用目录联接（`mklink /J`）；目录名称需与节点模型列表中的名称一致。

下载命令示例（国内网络环境可设置 `HF_ENDPOINT=https://hf-mirror.com`）：

```bash
# 例：下载默认模型 ACE-Step-Transcriber
huggingface-cli download ACE-Step/acestep-transcriber --local-dir "ComfyUI/models/audio_encoders/ACE-Step-Transcriber"
```

也可以用仓库根目录的 `download_models.bat` 一键下载推荐模型（脚本同样下载到该官方目录）。

| 模型 | 仓库 ID | 用途 | 显存参考 |
|---|---|---|---|
| ACE-Step-Transcriber（默认） | `ACE-Step/acestep-transcriber` | 歌词、段落和声乐信息提取 | ~7GB（bf16） |
| Qwen2.5-Omni-3B | `Qwen/Qwen2.5-Omni-3B` | 通用音频理解 | ~7GB |
| Ke-Omni-R-3B | `KE-Team/Ke-Omni-R-3B` | 通用音频理解 | ~7GB |
| MiDaShengLM-7B | `mispeech/midashenglm-7b-0804-bf16` | 音乐描述生成 | ~16GB（bf16） |
| MiDaShengLM-7B-GPTQ | `mispeech/midashenglm-7b-0804-w4a16-gptq` | 音乐描述生成，4bit 量化 | ~5GB，需 `auto-gptq` |
| MiDaShengLM-7B-FP8 | `mispeech/midashenglm-7b-0804-fp8` | 音乐描述生成，FP8 量化 | ~8GB，需支持 FP8 的 PyTorch/GPU |
| Qwen3-Omni-8B | `Qwen/Qwen3-Omni-8B` | 高质量通用音频理解 | ~16GB，需 `transformers>=4.53` |
| Qwen2-Audio-7B-Instruct | `Qwen/Qwen2-Audio-7B-Instruct` | 通用音频问答 | ~16GB |
| Whisper large-v3 (turbo/distil) | `openai/whisper-large-v3` 等 | 歌词转录 | ~3GB |
| MERT-v1-330M / AST-AudioSet | `m-a-p/MERT-v1-330M` / `MIT/ast-...` | 音频嵌入和事件分类 | <1GB |

> 16GB 显存环境建议优先选择 3B 规模模型；使用 MiDaShengLM 时，可根据显存情况选择 GPTQ 4bit 版本（需安装 `auto-gptq`）。

## 分析场景参考

| 场景 | 建议模型 | 说明 |
|---|---|---|
| 歌词、段落与声乐分析 | **ACE-Step-Transcriber** | 默认模型，中文歌词场景适用 |
| 音乐描述（用于文生音乐模型） | **MiDaShengLM-7B** | 可根据显存选择 BF16、GPTQ 或 FP8 版本 |
| BPM / 调性 | 无需额外模型 | 由 librosa 完成分析 |
| 轻量歌词转录 | Whisper-large-v3-turbo | 资源占用较低 |

建议以 ACE-Step-Transcriber 作为歌词与结构分析模型，并根据任务需求和显存配置选用 MiDaShengLM-7B。

## MiniMax Music 3 稳定性说明

Music 3 文本编码器包含较长的自回归采样，并使用 KV cache。部分
ComfyUI 0.35.x 与 PyTorch/CUDA 组合在启用 `comfy-aimdo` 动态编译器时，
可能在后续采样阶段出现 `aimdo memory compile error`、`device-side assert triggered`
或 `ScatterGatherKernel` 错误。遇到上述情况时，可在启动参数中加入：

```bash
python main.py --disable-comfy-compiler
```

该参数仅停用 ComfyUI 的内存编译与 CUDA Graph 优化，推理仍使用 GPU，
但可能减少部分编译优化收益。若启动器已配置 `--disable-cuda-malloc`，可与该参数同时保留。
修改启动参数后请完整重启 ComfyUI，再重新提交任务。

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

### 模型路径与加载

请确认模型目录位于 `ComfyUI/models/audio_encoders/<模型名>/`，且软链接目标可读。
对于分片模型，索引文件及其列出的全部 `safetensors` 分片必须同时存在。国内网络环境可先设置 `HF_ENDPOINT=https://hf-mirror.com`，再按模型支持表中的仓库 ID 下载。

### 显存与运行速度

优先选择与显存容量匹配的模型规模，并适当降低音频时长。分析节点支持在任务完成后卸载模型，以便为后续节点释放显存。

### 依赖版本

- Qwen3-Omni 需要 `transformers>=4.53`。
- MiDaShengLM GPTQ 版本需要安装 `auto-gptq`；也可改用 BF16 版本。

### Music 3 后续采样报错

出现 `aimdo memory compile error`、CUDA assert 或 `ScatterGatherKernel` 时，
请参照 [MiniMax Music 3 稳定性说明](#minimax-music-3-稳定性说明)，使用
`--disable-comfy-compiler` 完整重启 ComfyUI。

### 文本预览节点

`TextPreview` 接收批量输入并输出单一 STRING，适用于检查提示词、caption 和 lyrics。
如出现重复执行，请先确认扩展已更新至最新版本，并重启 ComfyUI 使节点定义重新加载。

## 合作算力平台

本项目可部署于具备 GPU 资源的云端 ComfyUI 环境。以下为合作平台入口，
适用于按需运行 ComfyUI、音频理解及 Music 3 工作流。注册链接包含对应的邀请信息，
平台价格、资源库存及服务条款以官方页面为准。

- **优云智算**：提供 GPU 云主机及镜像环境，适用于 ComfyUI、音频模型和 Music 3 工作流。
  [注册入口（邀请码）](https://passport.compshare.cn/register?referral_code=GMpN6yndJi2BFfvScmzRH2)
- **仙宫云**：提供面向 AI 工作流的弹性云端算力，适用于模型测试和持续运行任务。
  [注册入口（邀请链接）](https://www.xiangongyun.com/register/7AL3V3)

以上为合作推广入口；平台服务、价格与可用性以各平台公布的信息为准。

## 协议

MIT License，详见 [LICENSE](LICENSE)。音频理解核心逻辑源自 [ComfyUI-AceStep_SFT](https://github.com/ACE-Step/ComfyUI-AceStep_SFT)（MIT）。

# ComfyUI-WanAnimate-Enhancer-Custom

Enhanced WanAnimateToVideo with multi-dimensional control

Version: 1.2.0

## Features

- Motion strength control (body movement)
- Expression strength control (facial animation)
- Pose adherence control (follow pose video)
- Background blend control (character/background integration)
- Fully compatible with original WanAnimateToVideo parameters

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/THEman6989/ComfyUI-WanAnimate-Enhancer-Custom.git ComfyUI-WanAnimate-Enhancer-Custom
# Restart ComfyUI
```

## Nodes

### Wan Animate To Video Enhanced

Category: Wan2.2AnimateEnhancer

Enhanced version of WanAnimateToVideo with additional controls.

<img width="879" height="1122" alt="db712424c4b427f076ee1f1287489312" src="https://github.com/user-attachments/assets/33e2e239-52b5-4b56-8b55-a0cf764aaf93" />

<img width="550" height="355" alt="c888b980a0856a064c2555a0b5c32b9e" src="https://github.com/user-attachments/assets/3e037ef4-7bcf-4874-b6d1-f8786027d3aa" />


Parameters:
- motion_strength (0.0-3.0): Overall motion intensity
- expression_strength (0.0-3.0): Facial expression intensity
- pose_adherence (0.0-2.0): Pose following strength
- background_blend (0.0-1.0): Background integration level
- enable: Enable/disable all enhancements

All original WanAnimateToVideo parameters are supported.

### Wan Animate Model Enhancer

Category: Wan2.2AnimateEnhancer

Model patcher for motion strength control. Place between model loader and ToVideo node.

Parameters:
- enable: Enable/disable enhancement
- ffn_chunk_tokens: Token chunk size for the WAN transformer FFN (default `4096`; `0` disables it)

FFN chunking is activated only when the connected **MODEL itself** carries
DisTorch metadata. DisTorch CLIP or VAE loaders elsewhere in the workflow do
not activate it. Normal DynamicVRAM model branches keep the original WAN FFN.
Chunking reduces transient FFN/GELU activation peaks and does not change CLIP
or VAE execution.

### WAN FFN Chunking

Category: Wan2.2AnimateEnhancer

Standalone branch-local MODEL node for reducing WAN FFN/GELU peak VRAM. Unlike
the Model Enhancer's compatibility option, this node works with both ordinary
DynamicVRAM/standard WAN models and DisTorch WAN models.

Selectable presets:
- `disabled`
- `512`, `1024`, `2048`, `4096`, `8192`, `16384`, `32768` tokens

Smaller values reduce peak VRAM more strongly but run more FFN calls and are
therefore slower. Start with `4096`; use `2048` or `1024` for long/high-resolution
clips that still OOM.

```text
[WAN Model + LoRAs] -> [WAN FFN Chunking] -> [SageAttention] -> [KSampler]
```

## Usage

### Basic Usage (Data Layer Only)

```
[Model] -> [Wan Animate To Video Enhanced] -> [Sampler]
```

Controls expression, pose, and background. Motion control is limited.

### Full Control (Recommended)

```
[Model] -> [Wan Animate Model Enhancer] -> [Wan Animate To Video Enhanced] -> [Sampler]
```

Complete control over all parameters including motion strength.

## Parameter Guide

### Expression Strength

- 0.0-0.8: Subtle expressions
- 1.0: Original
- 1.2-2.0: Enhanced expressions
- 2.0+: Exaggerated (cartoon style)

### Motion Strength

- 0.0-0.8: Gentle movement
- 1.0: Original
- 1.2-1.8: Enhanced movement
- 1.8+: Exaggerated

### Pose Adherence

- 0.0-0.8: Loose following
- 1.0: Original
- 1.2-1.8: Strict following

### Background Blend

- 0.0-0.5: Transparent effect
- 0.8-1.0: Natural integration
- 1.0: Full background

## Examples

### Cartoon Animation
```
motion_strength: 1.8
expression_strength: 2.5
pose_adherence: 0.7
background_blend: 1.0
```

### Realistic Style
```
motion_strength: 0.9
expression_strength: 1.1
pose_adherence: 1.2
background_blend: 0.95
```

### Dance Performance
```
motion_strength: 1.4
expression_strength: 1.2
pose_adherence: 1.3
background_blend: 1.0
```

## Troubleshooting

### Nodes Not Showing

Check installation:
```bash
cd ComfyUI/custom_nodes/ComfyUI-Wan-Animate-Enhancer
ls -la
```

Should see __init__.py and wan_animate_tovideo_enhanced.py

### No Effect

- Check enable parameter is True
- Try larger parameter changes (1.5 or 2.0)
- Ensure using Wan2.2-Animate-14B model

### Artifacts in Output

- Lower all strength parameters to 1.0
- Adjust one parameter at a time
- Check input video quality

## Requirements

- ComfyUI (latest version)
- Wan2.2-Animate-14B model
- PyTorch 2.0+

## Fork note

This fork preserves the original MIT-licensed implementation and attribution
from [`wallen0322/ComfyUI-WanAnimate-Enhancer`](https://github.com/wallen0322/ComfyUI-WanAnimate-Enhancer).
It adds DisTorch-aware WAN FFN activation chunking and regression tests.

## License

MIT

## Links

- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- Wan2.2 Animate: https://huggingface.co/Wan-AI/Wan2.2-Animate-14B

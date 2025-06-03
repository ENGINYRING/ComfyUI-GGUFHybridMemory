[![ENGINYRING](https://cdn.enginyring.com/img/logo_dark.png)](https://www.enginyring.com)


# GGUF Hybrid Memory Manager for ComfyUI

A custom ComfyUI node set that implements a “hybrid” memory strategy for GGUF‐based models. It does **not** work on regular SafeTensor models.
This allows parts of a large UNet/CLIP/VAE to live in CPU RAM when VRAM is insufficient, avoiding OOMs and improving stability.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
   - [GGUFHybridMemory Node](#ggufhybridmemory-node)  
   - [ApplyGGUFHybrid Node](#applyggufhybrid-node)  
   - [“Skip CLIP” Option](#skip-clip-option)  
5. [Configuration Parameters](#configuration-parameters)  
6. [How It Works](#how-it-works)  
7. [Troubleshooting](#troubleshooting)  
8. [License](#license)  

---

## Overview

When loading large GGUF‐formatted UNet, CLIP or VAE models in ComfyUI, you may encounter VRAM exhaustion. This “GGUF Hybrid Memory” add-on:

- Estimates each model’s compressed GGUF footprint  
- Checks free VRAM (with a configurable safety buffer) plus available system RAM  
- Splits model weights between GPU and CPU when needed  
- Monkey-patches the model’s `.to()` method to proactively avoid OOMs (by falling back to CPU if VRAM is insufficient)  
- Pins CPU tensors in page-locked (pinned) memory to speed up transfers  

The result is more stable loading of very large GGUF models, even on GPUs with limited VRAM.

---

## Features

- **Automatic GPU/CPU Split**  
  Estimates how many bytes can safely live on GPU versus CPU.

- **Proactive OOM Prevention**  
  Before `.to("cuda")`, checks “free VRAM – 15% overhead” and will instead send the module to CPU if it doesn’t fit.

- **Pinned CPU Tensors**  
  After moving a submodule to CPU, its buffers and parameters are pinned so that any future GPU transfers (if triggered) happen faster.

- **“Skip CLIP” Option**  
  Optionally skip patching/loading CLIP altogether (useful if you want to leave CLIP on CPU by default).

- **Configurable Safety Buffers**  
  - VRAM overhead percentage (default 15 %)  
  - “Conservative factor” to further reduce GPU budget (default 0.7)  
  - System-RAM pad (either a fixed GB floor or 10 % of total)

---

## Installation

1. **Clone or download** this repository (or copy the `GGUFHybridMemory.py` file) and place it under:

   ```
   <Your ComfyUI folder>/
       custom_nodes/
         └── GGUFHybridMemory.py
   ```

2. **Restart ComfyUI**. You should see two new nodes under the **memory_management** category:
   - 🔧 GGUF Hybrid Memory  
   - ⚙️ Apply GGUF Hybrid  

3. Make sure you have **PyTorch 2.7+ (with CUDA)** and **psutil** installed. For example:
   ```bash
   pip install torch psutil
   ```
   (ComfyUI’s own environment should already include `torch`, but `psutil` sometimes needs manual install.)

---

## Usage

### GGUFHybridMemory Node

1. Drag “🔧 GGUF Hybrid Memory” into your flow.
2. Set **Target GPU** (e.g. `cuda:0`) and **Model Type** (e.g. `gguf_unet`, `gguf_clip`, `gguf_vae` or `auto`).  
3. (Optional) Tweak:
   - **Conservative factor** (0.5 – 0.9; default 0.7)  
   - **GGUF overhead %** (10 – 30; default 15)  
   - **System RAM pad (GB)** (or leave blank to auto = max(2 GB, 10 % of total))  
   - **Skip CLIP** (ON/OFF; default OFF)  
4. The node outputs a single data blob named **`gguf_config`**.

### ApplyGGUFHybrid Node

1. Connect your **`gguf_config`** output into the **ApplyGGUFHybrid** node’s **gguf_config** input.  
2. Select which submodule to patch by setting **`model_attr`** (choose one of:
   - `model` (= UNet)  
   - `cond_stage_model` (= CLIP)  
   - `first_stage_model` (= VAE)  
   )
3. Plug in the actual **UNet**, **CLIP**, or **VAE** model handle into the matching input port.  
4. The node will:
   - Estimate the model’s GGUF footprint  
   - Compute how many bytes can fit on GPU  vs  CPU  
   - Monkey-patch that submodule’s `.to()` method  
   - Return the same model handle (now patched) and a short **status string**  

### “Skip CLIP” Option

If **Skip CLIP** = `True`, then when you run **ApplyGGUFHybrid** with `model_attr="cond_stage_model"`, it will do nothing and print a warning. Use this if you want CLIP to always remain on CPU without patching.

---

## Configuration Parameters

| Parameter                  | Node           | Type     | Default | Description                                                                                                                                      |
| :------------------------- | :------------- | :------- | :------ | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| **target_gpu**             | Required       | `combo`  | `"cuda:0"` (or your first GPU) | Which CUDA device to target.                                                                                                     |
| **model_type**             | Required       | `combo`  | `"gguf_unet"` | Just for logging/GUI; can be one of `gguf_unet`, `gguf_clip`, `gguf_vae`, or `auto`.                                                             |
| **conservative_factor**    | Optional       | `float`  | `0.7`   | Multiplier on “(free VRAM – overhead)” to set GPU budget. Lower → more data forced to CPU.                                                         |
| **gguf_overhead_percent**  | Optional       | `float`  | `15.0`  | Percent of total VRAM reserved as a safety buffer for GGUF operations.                                                                           |
| **system_ram_pad_gb**      | Optional       | `float`  | `None`  | Fixed number of GB to subtract from system-RAM available. Leave blank for auto = `max(2 GB, 10 % of total RAM)`.                                    |
| **skip_clip**              | Optional       | `boolean`| `False` | If `True`, the CLIP submodule (`cond_stage_model`) will not be patched or offloaded when “Apply GGUF Hybrid” is run on that attr.                   |

---

## How It Works

1. **Estimate Model Size** – `estimate_gguf_model_size()`  
   - Sum up all parameters’ `numel() * element_size()`  
   - Multiply by 1.5× if it has `gguf_file` or `qtypes`, otherwise 1.2×  
   - Floor at 2 GB  

2. **Compute Optimal Split** – `get_gguf_optimal_split()`  
   - Poll “free VRAM” = `(total_vram – reserved)` – `(gguf_overhead)`  
   - GPU budget = `free_vram * conservative_factor`  
   - System RAM available = `(available_RAM – pad_bytes)`  
   - If `estimated_size <= gpu_budget`, do “gpu_only”  
   - Else if `estimated_size <= gpu_budget + sys_avail`, do “gguf_hybrid” (split)  
   - Else “gguf_insufficient” (best‐effort split, but shortfall flagged)  

3. **Patch `.to()`** – `patch_gguf_model_loading()`  
   - Replace `pmod.to(...)` with `hybrid_to(...)` wrapper  
   - On `hybrid_to("cuda")`, re‐check free VRAM. If not enough, force `CPU`  
   - On `hybrid_to("cpu")`, first move to CPU, then pin each **CPU** buffer/param → `.pin_memory()`  
   - On GPU move, attempt `original_to(cuda, non_blocking=True)`, and if OOM, fallback to CPU + pin again  

4. **Apply in Flow** – Connect “GGUF Hybrid Memory” → “Apply GGUF Hybrid” → your UNet/CLIP/VAE loader nodes.  

---

## Troubleshooting

- **“cannot pin 'torch.cuda.FloatTensor'”**:  
  This readme’s code already checks `buf.device.type == "cpu"` before calling `pin_memory()`. If you see that error, make sure you are using the latest patched version—older versions attempted to pin CUDA tensors.

- **TypeError: cannot create weak reference to 'GGUFHybridManager' object**:  
  Make sure your `GGUFHybridManager` class has `__slots__ = (..., "__weakref__")`, or remove/rename any conflicting attribute that prevents adding a weakref slot.

- **Unexpected validation errors in the node UI**:  
  - Ensure `model_type` is exactly one of `["gguf_unet", "gguf_clip", "gguf_vae", "auto"]` (all lowercase).  
  - Ensure `gguf_overhead_percent` is ≥ 10.0 (per the node’s widget constraints).  

- **Still get an OOM when loading CLIP**:  
  - Try toggling “Skip CLIP” to `True` (so you can manually manage CLIP placement or let ComfyUI’s default loader handle it).  
  - Increase `system_ram_pad_gb` to offload more of CLIP to CPU.  

---

## License

This project is released under the [MIT License](LICENSE). Feel free to fork and adapt for your own needs.

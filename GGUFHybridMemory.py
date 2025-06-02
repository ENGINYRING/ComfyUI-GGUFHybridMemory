# GGUF-Compatible Hybrid Memory Manager for ComfyUI
# Save as: ComfyUI/custom_nodes/GGUFHybridMemory.py
# Description:
# Further micro-optimizations: cache device objects, minimize attribute lookups,
# inline frequently used locals, and avoid string splits in the hot path.
# Added: skip_clip setting to avoid patching/loading CLIP when not needed.

from __future__ import annotations

import types
import functools
import logging
import torch
import threading
import time
import gc
import psutil
import weakref
from typing import Dict, Any, Tuple, Optional

import comfy.model_management  # noqa: F401 – imported for side-effects

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────
BYTES_PER_GB = 1024**3
MIN_SYSTEM_RAM_PAD_GB = 2.0               # at least 2 GB pad
SYSTEM_RAM_PAD_FRACTION = 0.10            # 10% of total if larger
DEFAULT_GGUF_OVERHEAD_PERCENT = 15.0      # 15% VRAM reserved for GGUF
DEFAULT_CONSERVATIVE_FACTOR = 0.7
DEFAULT_GGUF_OVERHEAD_FACTOR = 1.5        # 50% overhead for GGUF-style models
DEFAULT_GENERIC_OVERHEAD_FACTOR = 1.2     # 20% overhead for other models
MIN_ESTIMATED_MODEL_SIZE_GB = 2.0         # floor at 2 GB
MEMORY_CACHE_TTL_SECONDS = 1.0            # cache memory stats for 1 second

# ────────────────────────────────────────────────────────────────────────────────
# Memory Profiler
# ────────────────────────────────────────────────────────────────────────────────
class GGUFMemoryProfiler:
    """
    Collect GPU and system-RAM stats with caching and overhead considerations.
    Caches static GPU total memory and polls only allocated/reserved each TTL.
    """

    __slots__ = ("_cached_total", "gpu_memory", "system_memory", "_last_update_time")

    def __init__(self) -> None:
        self._cached_total: Dict[str, int] = {}
        self.gpu_memory: Dict[str, Dict[str, float]] = {}
        self.system_memory: Dict[str, float] = {}
        self._last_update_time: float = 0.0
        self._cache_static_gpu_props()

    def _cache_static_gpu_props(self) -> None:
        """Cache total_memory per CUDA device once."""
        if not torch.cuda.is_available():
            return
        ct = self._cached_total
        for i in range(torch.cuda.device_count()):
            dev = f"cuda:{i}"
            ct[dev] = torch.cuda.get_device_properties(i).total_memory

    def _update_memory_info(self, force: bool = False) -> None:
        """Update GPU (allocated/reserved) and system memory, TTL-cached."""
        now = time.monotonic()
        if not force and (now - self._last_update_time) < MEMORY_CACHE_TTL_SECONDS:
            return

        # Update GPU stats (only allocated/reserved; total from cache)
        if torch.cuda.is_available():
            gm = self.gpu_memory
            ct = self._cached_total
            # Local references to methods for speed
            alloc_fn = torch.cuda.memory_allocated
            resv_fn = torch.cuda.memory_reserved
            for dev, total in ct.items():
                idx = int(dev[5:])  # ‘cuda:’ → index
                alloc = alloc_fn(idx)
                resv = resv_fn(idx)
                gm[dev] = {
                    "total": total,
                    "allocated": alloc,
                    "reserved": resv,
                    "free": total - resv,
                }

        # Update system RAM stats
        mem = psutil.virtual_memory()
        self.system_memory = {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "free": mem.free,
            "percent": mem.percent,
        }

        self._last_update_time = now

    def get_gguf_optimal_split(
        self,
        estimated_size: int,
        target_gpu: str,
        conservative_factor: float = DEFAULT_CONSERVATIVE_FACTOR,
        gguf_overhead_percent: float = DEFAULT_GGUF_OVERHEAD_PERCENT,
        system_ram_pad_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Determine how many bytes to place on GPU vs. CPU.
        Uses an adaptive system-RAM pad: max(fixed GB, fraction of total).
        """
        self._update_memory_info()

        gm = self.gpu_memory
        if target_gpu not in gm:
            raise ValueError(f"GPU {target_gpu!r} not found or CUDA unavailable")

        gstat = gm[target_gpu]
        total_gpu = gstat["total"]
        free_gpu = gstat["free"]
        overhead = total_gpu * (gguf_overhead_percent * 0.01)
        gpu_avail = free_gpu - overhead
        if gpu_avail < 0:
            gpu_budget = 0
        else:
            gpu_budget = int(gpu_avail * conservative_factor)

        sys_mem = self.system_memory
        total_sys = sys_mem["total"]
        avail_sys = sys_mem["available"]
        if system_ram_pad_gb is None:
            pad_bytes = max(
                int(MIN_SYSTEM_RAM_PAD_GB * BYTES_PER_GB),
                int(total_sys * SYSTEM_RAM_PAD_FRACTION),
            )
        else:
            pad_bytes = int(system_ram_pad_gb * BYTES_PER_GB)
        sys_avail = avail_sys - pad_bytes
        if sys_avail < 0:
            sys_avail = 0

        if estimated_size <= gpu_budget:
            return {
                "gpu_bytes": estimated_size,
                "cpu_bytes": 0,
                "strategy": "gpu_only",
                "gguf_compatible": True,
            }
        if estimated_size <= gpu_budget + sys_avail:
            return {
                "gpu_bytes": gpu_budget,
                "cpu_bytes": estimated_size - gpu_budget,
                "strategy": "gguf_hybrid",
                "gguf_compatible": True,
            }
        return {
            "gpu_bytes": gpu_budget,
            "cpu_bytes": sys_avail,
            "strategy": "gguf_insufficient",
            "gguf_compatible": False,
            "shortfall": estimated_size - (gpu_budget + sys_avail),
        }


# ────────────────────────────────────────────────────────────────────────────────
# Layer Interceptor (for future per-layer offload)
# ────────────────────────────────────────────────────────────────────────────────
class GGUFLayerInterceptor:
    """
    Tracks layer sizes and placement for possible fine-grained CPU/GPU offload.
    Not actively used in this version; provided for future extensions.
    """

    __slots__ = ("target_device", "cpu_device", "mem_split", "gpu_used", "gpu_total", "layer_map", "_lock")

    def __init__(self, target: torch.device, split: Dict[str, Any]) -> None:
        self.target_device = target
        self.cpu_device = torch.device("cpu")
        self.mem_split = split
        self.gpu_used = 0
        self.gpu_total = split["gpu_bytes"]
        self.layer_map: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def should_place_on_gpu(self, size: int, name: str = "") -> bool:
        with self._lock:
            lname = name.lower()
            hi = "attn" in lname or "transformer" in lname or "block" in lname or "q_proj" in lname or "k_proj" in lname or "v_proj" in lname
            lo = "norm" in lname or "embed" in lname or "bias" in lname
            if hi and (self.gpu_used + size) <= self.gpu_total:
                return True
            if lo:
                return False
            return (self.gpu_used + size) <= self.gpu_total

    def track(self, name: str, size: int, device: torch.device) -> None:
        with self._lock:
            self.layer_map[name] = {"device": device, "size": size, "last_access": time.time()}
            if device.type == "cuda":
                self.gpu_used += size


# ────────────────────────────────────────────────────────────────────────────────
# Hybrid Manager
# ────────────────────────────────────────────────────────────────────────────────
class GGUFHybridManager:
    """Orchestrates model-size estimation, memory-split, and hybrid `.to()` patching."""

    __slots__ = ("profiler", "managed_models", "_lock", "__weakref__")

    def __init__(self) -> None:
        self.profiler = GGUFMemoryProfiler()
        # WeakKeyDictionary so that models can be garbage-collected
        self.managed_models: weakref.WeakKeyDictionary[Any, Dict[str, Any]] = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def estimate_gguf_model_size(self, model: Any, attr: str) -> int:
        """Estimate model footprint with GGUF-specific overhead."""
        try:
            pmod = getattr(model, attr) if attr else model
        except AttributeError:
            raise ValueError(f"Attribute {attr!r} not found on {model.__class__.__name__}")

        total_bytes = 0
        for p in pmod.parameters():
            total_bytes += p.numel() * p.element_size()

        # GGUF-style models need more buffer space
        if hasattr(pmod, "gguf_file") or hasattr(pmod, "qtypes"):
            est = int(total_bytes * DEFAULT_GGUF_OVERHEAD_FACTOR)
        else:
            est = int(total_bytes * DEFAULT_GENERIC_OVERHEAD_FACTOR)

        min_bytes = int(MIN_ESTIMATED_MODEL_SIZE_GB * BYTES_PER_GB)
        return est if est >= min_bytes else min_bytes

    def patch_gguf_model_loading(
        self,
        model: Any,
        attr: str,
        target_gpu: str,
        estimated_size: int
    ) -> Dict[str, Any]:
        """
        Replace model.to() with a hybrid-aware version:
        - Proactive OOM prevention (check free VRAM)
        - Pin CPU buffers/params after moving to CPU
        - Use non_blocking=True on GPU transfers
        """
        pmod = getattr(model, attr) if attr else model
        if hasattr(pmod, "device"):
            pmod.device = torch.device(target_gpu)

        original_to = pmod.to
        manager_ref = weakref.ref(self)

        # Cache “cpu” device object and parse GPU index once
        CPU_DEV = torch.device("cpu")
        GPU_DEV = torch.device(target_gpu)
        # If target_gpu is “cuda:0”, idx = 0; avoid splitting string repeatedly:
        gpu_index = int(target_gpu[5:]) if target_gpu.startswith("cuda") else None

        @functools.wraps(original_to)
        def hybrid_to(module_self, device=None, *args, **kwargs):
            # if called without arguments, just return self
            if device is None:
                return module_self

            # Determine target device
            tgt = GPU_DEV if (isinstance(device, str) and device.startswith("cuda")) or (device == GPU_DEV) else CPU_DEV
            mgr = manager_ref()

            # If manager was GC’d, fallback
            if mgr is None:
                return original_to(tgt, *args, **kwargs)

            # Pre-flight OOM check only if going to GPU
            if tgt.type == "cuda":
                mgr.profiler._update_memory_info(force=True)
                gstat = mgr.profiler.gpu_memory.get(target_gpu)
                if gstat is not None:
                    free_vram = gstat["free"]
                    if estimated_size > free_vram:
                        # Not enough free VRAM: force CPU
                        cleanup()
                        tgt = CPU_DEV

            # If we ended up on CPU, move first, then pin any CPU‐resident tensors
            if tgt.type == "cpu":
                # 1) Move the module to CPU
                mod_after_move = original_to(CPU_DEV, *args, **kwargs)
                # 2) Pin any CPU buffers/params
                for _, buf in module_self.named_buffers():
                    if buf is not None and buf.device.type == "cpu":
                        buf.data = buf.data.pin_memory()
                for _, param in module_self.named_parameters():
                    if param is not None and param.device.type == "cpu":
                        param.data = param.data.pin_memory()
                return mod_after_move

            # Otherwise, try GPU move with non_blocking
            try:
                return original_to(GPU_DEV, non_blocking=True, *args, **kwargs)
            except torch.cuda.OutOfMemoryError:
                # If OOM (maybe fragmentation), fallback to CPU
                cleanup()
                # 1) Move to CPU
                mod_after_move = original_to(CPU_DEV, *args, **kwargs)
                # 2) Pin CPU tensors
                for _, buf in module_self.named_buffers():
                    if buf is not None and buf.device.type == "cpu":
                        buf.data = buf.data.pin_memory()
                for _, param in module_self.named_parameters():
                    if param is not None and param.device.type == "cpu":
                        param.data = param.data.pin_memory()
                return mod_after_move

        pmod.to = types.MethodType(hybrid_to, pmod)

        with self._lock:
            self.managed_models[pmod] = {
                "original_to": original_to,
                "patched_at": time.time(),
            }

        return {"model_id": id(pmod)}

    def setup_gguf_hybrid(
        self,
        model: Any,
        attr: str,
        *,
        target_gpu: str,
        conservative_factor: float = DEFAULT_CONSERVATIVE_FACTOR,
        gguf_overhead_percent: float = DEFAULT_GGUF_OVERHEAD_PERCENT,
        system_ram_pad_gb: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        1. Estimate size
        2. Compute split (adaptive pad)
        3. Patch model.to(…) for hybrid transfers
        """
        try:
            est = self.estimate_gguf_model_size(model, attr)
            split = self.profiler.get_gguf_optimal_split(
                estimated_size=est,
                target_gpu=target_gpu,
                conservative_factor=conservative_factor,
                gguf_overhead_percent=gguf_overhead_percent,
                system_ram_pad_gb=system_ram_pad_gb,
            )

            logger.info(
                "GGUF split for '%s' → %s | GPU: %.2fGB | CPU: %.2fGB",
                attr, split["strategy"],
                split["gpu_bytes"] / BYTES_PER_GB,
                split["cpu_bytes"] / BYTES_PER_GB,
            )

            if not split["gguf_compatible"]:
                logger.warning(
                    "Insufficient memory: shortfall %.2fGB",
                    split.get("shortfall", 0) / BYTES_PER_GB,
                )
                return {"status": "failed_insufficient_memory", **split}

            patch_info = self.patch_gguf_model_loading(
                model, attr, target_gpu, est
            )

            return {
                "status": "success",
                "estimated_size_gb": est / BYTES_PER_GB,
                **split,
                **patch_info,
            }

        except Exception as e:
            logger.exception("Error in setup_gguf_hybrid")
            return {"status": "error", "message": str(e)}


# ────────────────────────────────────────────────────────────────────────────────
# ComfyUI Node Wrappers
# ────────────────────────────────────────────────────────────────────────────────
class GGUFHybridMemoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        devs = ["cpu"]
        if torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        default_dev = devs[1] if len(devs) > 1 else "cpu"
        return {
            "required": {
                "target_gpu": (devs, {"default": default_dev}),
                # normalized to lowercase so "gguf_unet" is valid
                "model_type": (["gguf_unet", "gguf_clip", "gguf_vae", "auto"], {"default": "gguf_unet"}),
            },
            "optional": {
                "conservative_factor": ("FLOAT", {"default": 0.7, "min": 0.5, "max": 0.9, "step": 0.05}),
                "gguf_overhead_percent": ("FLOAT", {"default": 15.0, "min": 10.0, "max": 30.0, "step": 1.0}),
                "system_ram_pad_gb": ("FLOAT", {"default": None}),
                "skip_clip": ("BOOLEAN", {"default": False}),  # <--- NEW: skip CLIP if True
            },
        }

    RETURN_TYPES = ("GGUF_HYBRID_CONFIG",)
    RETURN_NAMES = ("gguf_config",)
    FUNCTION = "create_gguf_config"
    CATEGORY = "memory_management"
    TITLE = "GGUF Hybrid Memory"

    def __init__(self) -> None:
        self.manager = GGUFHybridManager()

    def create_gguf_config(
        self,
        target_gpu: str,
        model_type: str = "gguf_unet",
        conservative_factor: float = DEFAULT_CONSERVATIVE_FACTOR,
        gguf_overhead_percent: float = DEFAULT_GGUF_OVERHEAD_PERCENT,
        system_ram_pad_gb: Optional[float] = None,
        skip_clip: bool = False,  # <--- NEW
    ) -> Tuple[Dict[str, Any]]:
        cfg = {
            "gguf_manager": self.manager,
            "created": time.time(),
            "target_gpu": target_gpu,
            "model_type": model_type,
            "conservative_factor": conservative_factor,
            "gguf_overhead_percent": gguf_overhead_percent,
            "system_ram_pad_gb": system_ram_pad_gb,
            "skip_clip": skip_clip,  # <--- NEW
        }
        logger.info(
            "GGUF config created for %s (factor %.2f, overhead %.1f%%, pad %s, skip_clip=%r)",
            target_gpu,
            conservative_factor,
            gguf_overhead_percent,
            f"{system_ram_pad_gb}GB" if system_ram_pad_gb is not None else "auto",
            skip_clip,
        )
        return (cfg,)


class ApplyGGUFHybrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_config": ("GGUF_HYBRID_CONFIG",),
                "model_attr": (["model", "cond_stage_model", "first_stage_model"], {"default": "model"}),
            },
            "optional": {"model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",)},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "report")
    FUNCTION = "apply_gguf_hybrid"
    CATEGORY = "memory_management"
    TITLE = "Apply GGUF Hybrid"

    def apply_gguf_hybrid(
        self,
        gguf_config: Dict[str, Any],
        model_attr: str,
        model=None,
        clip=None,
        vae=None,
    ):
        mgr: GGUFHybridManager = gguf_config["gguf_manager"]
        skip_clip = gguf_config.get("skip_clip", False)

        target_map = {"model": model, "cond_stage_model": clip, "first_stage_model": vae}
        name_map = {"model": "UNet", "cond_stage_model": "CLIP", "first_stage_model": "VAE"}
        out_model, out_clip, out_vae = model, clip, vae
        lines: list[str] = []

        tgt_model = target_map.get(model_attr)

        # If skip_clip is True and user is attempting to patch CLIP, skip it
        if skip_clip and model_attr == "cond_stage_model":
            lines.append("⚠️ CLIP patching was skipped (skip_clip=True).")
        elif tgt_model is not None:
            mname = name_map[model_attr]
            try:
                # Patch or offload memory for this model part
                res = mgr.setup_gguf_hybrid(
                    tgt_model,
                    model_attr,
                    target_gpu=gguf_config["target_gpu"],
                    conservative_factor=gguf_config["conservative_factor"],
                    gguf_overhead_percent=gguf_config["gguf_overhead_percent"],
                    system_ram_pad_gb=gguf_config.get("system_ram_pad_gb"),
                )
                if res.get("status") == "success":
                    lines.append(f"✅ {mname} patched → {res['strategy']}")
                else:
                    lines.append(f"❌ {mname} patch failed: {res.get('message','Insufficient memory')}")
            except Exception as e:
                logger.exception("Error during GGUF hybrid setup:")
                lines.append(f"❌ Exception: {e}")
        else:
            lines.append("❌ No matching model supplied for the chosen attribute.")

        # Post-operation memory report
        prof = GGUFMemoryProfiler()
        prof._update_memory_info(force=True)
        gpu_dev = gguf_config.get("target_gpu", "cuda:0")
        if gpu_dev in prof.gpu_memory:
            g = prof.gpu_memory[gpu_dev]
            lines.append(f"📊 GPU {gpu_dev}: {g['allocated']/BYTES_PER_GB:.2f}/{g['total']/BYTES_PER_GB:.2f} GB used")

        sys = prof.system_memory
        lines.append(f"📊 System RAM: {sys['used']/BYTES_PER_GB:.2f}/{sys['total']/BYTES_PER_GB:.2f} GB ({sys['percent']}%)")

        return out_model, out_clip, out_vae, "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────────
# Utilities & Node Registration
# ────────────────────────────────────────────────────────────────────────────────
def cleanup() -> None:
    """Free cached CUDA buffers and run the Python garbage collector."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


NODE_CLASS_MAPPINGS = {
    "GGUFHybridMemory": GGUFHybridMemoryNode,
    "ApplyGGUFHybrid": ApplyGGUFHybrid,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUFHybridMemory": "🔧 GGUF Hybrid Memory",
    "ApplyGGUFHybrid": "⚙️ Apply GGUF Hybrid",
}

"""Naive bf16 -> float8_e4m3fn cast for 2D weight tensors (research-only, no scale factors).

Research-only diagnostic cast, not a production quantization method.
All 2D tensors whose key ends with '.weight' are cast; other tensors are copied
as-is. That includes Linear weights but may also include other 2D parameters
(embeddings, norms stored as 2D, etc.) depending on the checkpoint layout.

Usage:
    python quant_bf16_to_fp8.py INPUT.safetensors OUTPUT.safetensors

Requirements:
    - Input must be a safetensors file readable by torch/safetensors.
    - PyTorch build must support torch.float8_e4m3fn (see requirements.txt).
    - RAM: the full state dict is held in memory while writing
      (~checkpoint size + output size). bf16 ~10 GB model -> expect >20 GB free RAM
      and enough disk for the output (~5 GB for this TRELLIS.2 DiT).
    - Output is written to OUTPUT.tmp then renamed, so a crash should not leave
      a half-written final file (you may still see OUTPUT.tmp).

Metadata written into the output (method provenance):
    quant_method = naive_fp8_e4m3fn_no_scale
    fp8_max = 448.0
    source = <input filename>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0  # float8_e4m3fn max finite value
QUANT_METHOD = "naive_fp8_e4m3fn_no_scale"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="source bf16 safetensors")
    p.add_argument("output", type=Path, help="destination fp8 safetensors")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src: Path = args.input
    dst: Path = args.output

    if not src.is_file():
        print(f"[error] input not found: {src}", file=sys.stderr)
        print(
            "[hint] download e.g. Comfy-Org TRELLIS.2 bf16 diffusion model "
            "into your ComfyUI models/diffusion_models/ folder, then pass that path.",
            file=sys.stderr,
        )
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    src_sz = src.stat().st_size
    print(f"[src] {src} ({src_sz / 1e9:.2f} GB)")
    print("[warn] full state dict is held in RAM while writing; prefer a machine with")
    print("[warn] more free RAM than (src + dst) size, e.g. >20 GB for this model.")
    t0 = time.time()

    out: dict[str, torch.Tensor] = {}
    quantized = 0
    kept_2d = 0
    clamped = 0
    max_abs = 0.0
    per_module: dict[str, int] = {}

    with safe_open(str(src), framework="pt") as f:
        keys = list(f.keys())
        total = len(keys)
        print(f"[src] {total} tensors", flush=True)

        for i, k in enumerate(keys):
            t = f.get_tensor(k)
            # All 2D '.weight' tensors (Linear weights and any other 2D params
            # named *.weight in this checkpoint layout).
            is_2d_weight = k.endswith(".weight") and t.dim() == 2

            if is_2d_weight:
                # bf16 -> fp32 is lossless (bf16 subset of fp32), then round to fp8 grid
                x = t.to(torch.float32)
                m = x.abs().max().item() if x.numel() else 0.0
                if m > max_abs:
                    max_abs = m
                if m > FP8_MAX:
                    clamped += int((x.abs() > FP8_MAX).sum())
                    x = x.clamp(-FP8_MAX, FP8_MAX)
                out[k] = x.to(torch.float8_e4m3fn).contiguous()
                quantized += 1
                parts = k.split(".")
                mod = parts[1] if len(parts) > 1 else k
                per_module[mod] = per_module.get(mod, 0) + 1
            else:
                out[k] = t.contiguous()
                if t.dim() == 2:
                    kept_2d += 1

            if (i + 1) % 400 == 0:
                print(f"[{i + 1}/{total}] quantized={quantized}", flush=True)

    print(f"\n[quant] 2D *.weight cast to fp8: {quantized}", flush=True)
    print(f"[quant] per module: {per_module}", flush=True)
    print(f"[quant] max |w| before quant: {max_abs:.4f} (fp8 max {FP8_MAX})", flush=True)
    print(f"[quant] elements clamped to ±FP8_MAX: {clamped}", flush=True)
    if kept_2d:
        print(f"[quant] other 2D tensors kept as-is: {kept_2d}", flush=True)

    metadata = {
        "quant_method": QUANT_METHOD,
        "fp8_dtype": "float8_e4m3fn",
        "fp8_max": str(FP8_MAX),
        "source_file": src.name,
        "note": "naive cast without scale factors; not a production quant method",
    }
    save_file(out, str(tmp), metadata=metadata)
    del out
    tmp.replace(dst)

    dst_sz = dst.stat().st_size
    print(f"\n[dst] {dst} ({dst_sz / 1e9:.2f} GB, {100 * dst_sz / src_sz:.0f}% of src)")

    dtypes: dict[str, int] = {}
    with safe_open(str(dst), framework="pt") as f:
        for k in f.keys():
            d = str(f.get_slice(k).get_dtype())
            dtypes[d] = dtypes.get(d, 0) + 1
        n = len(f.keys())
        try:
            meta = f.metadata()
        except Exception:
            meta = None
    print(f"[verify] {n} tensors, dtypes: {dtypes}")
    if meta:
        print(f"[verify] metadata: {meta}")
    print(f"[done] {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

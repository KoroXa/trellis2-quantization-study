"""Compare weight tensors between a bf16 checkpoint and a (naive) fp8 checkpoint.

Usage:
    python compare_weights.py <bf16.safetensors> <fp8.safetensors> [--csv out.csv]

Prints per-layer metrics for selected DiT layers:
  - mae_over_mean_abs_ref: mean(|x-ref|) / mean(|ref|)
    (not L2 relative error; formula is stated in the README)
  - zero_frac_bf16, zero_frac_fp8, new_zero_frac
    new_zero_frac = fraction of elements that are zero in fp8 but not zero in bf16
  - abs_max_bf16

Does not need ComfyUI; only torch + safetensors.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from safetensors import safe_open

# Exact state-dict keys (Trellis2 DiT structure / img2shape path in Comfy-Org checkpoints).
# Keys include the ComfyUI "model." prefix used in these files.
LAYERS = [
    "model.structure_model.t_embedder.mlp.0.weight",
    "model.structure_model.blocks.0.self_attn.to_qkv.weight",
    "model.structure_model.blocks.0.self_attn.to_out.weight",
    "model.structure_model.adaLN_modulation.1.weight",
    "model.structure_model.input_layer.weight",
    "model.structure_model.out_layer.weight",
    "model.img2shape.input_layer.weight",
    "model.img2shape.out_layer.weight",
]


def mae_over_mean_abs_ref(ref: torch.Tensor, x: torch.Tensor) -> float:
    ref_f = ref.to(torch.float32)
    x_f = x.to(torch.float32)
    denom = ref_f.abs().mean().clamp_min(1e-8)
    return ((x_f - ref_f).abs().mean() / denom).item()


def zero_frac(t: torch.Tensor) -> float:
    return (t == 0).float().mean().item()


def new_zero_frac(ref: torch.Tensor, x: torch.Tensor) -> float:
    ref_z = ref == 0
    x_z = x == 0
    new = x_z & ~ref_z
    return new.float().mean().item()


def find_key(keys, name: str) -> str | None:
    if name in keys:
        return name
    soft = [k for k in keys if k.endswith(name)]
    return soft[0] if soft else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bf16", type=Path, help="bf16 safetensors checkpoint")
    p.add_argument("fp8", type=Path, help="fp8 (or other) safetensors checkpoint")
    p.add_argument("--csv", type=Path, default=None, help="optional CSV output path")
    args = p.parse_args()

    src, dst = args.bf16, args.fp8
    if not src.is_file() or not dst.is_file():
        print(f"missing file:\n  {src}\n  {dst}", file=sys.stderr)
        return 1

    print(f"bf16 : {src.name} ({src.stat().st_size / 1e9:.2f} GB)")
    print(f"other: {dst.name} ({dst.stat().st_size / 1e9:.2f} GB)")
    print()
    print("mae_over_mean_abs_ref = mean(|x-ref|) / mean(|ref|)  [not L2 relative error]")
    print()
    header = (
        f"{'layer':<48} {'mae/mean|ref|':>13} {'zero_bf16':>10} "
        f"{'zero_fp8':>10} {'new_zero':>10} {'max_ref':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    with safe_open(str(src), framework="pt") as f_ref, safe_open(str(dst), framework="pt") as f_x:
        ref_keys = list(f_ref.keys())
        x_keys = list(f_x.keys())
        for layer in LAYERS:
            rk = find_key(ref_keys, layer)
            xk = find_key(x_keys, layer)
            if rk is None or xk is None:
                print(f"{layer:<48} {'MISSING':>13}")
                continue
            ref = f_ref.get_tensor(rk)
            x = f_x.get_tensor(xk)
            if ref.shape != x.shape:
                print(f"{layer:<48} {'SHAPE_MISMATCH':>13} {tuple(ref.shape)} vs {tuple(x.shape)}")
                continue
            mae = mae_over_mean_abs_ref(ref, x)
            zb = zero_frac(ref)
            zf = zero_frac(x)
            zn = new_zero_frac(ref, x)
            m = ref.float().abs().max().item() if ref.numel() else 0.0
            print(f"{layer:<48} {mae:13.4f} {zb:10.4f} {zf:10.4f} {zn:10.4f} {m:8.3f}")
            rows.append(
                {
                    "layer": layer,
                    "mae_over_mean_abs_ref": f"{mae:.4f}",
                    "zero_frac_bf16": f"{zb:.4f}",
                    "zero_frac_fp8": f"{zf:.4f}",
                    "new_zero_frac": f"{zn:.4f}",
                    "abs_max_bf16": f"{m:.3f}",
                }
            )

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "layer",
                    "mae_over_mean_abs_ref",
                    "zero_frac_bf16",
                    "zero_frac_fp8",
                    "new_zero_frac",
                    "abs_max_bf16",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

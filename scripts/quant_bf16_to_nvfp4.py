#!/usr/bin/env python3
"""Research-only: quantize a bf16 DiT checkpoint to ComfyUI-native NVFP4.

Reads the source checkpoint with a manual byte path (works around a
safetensors/torch deserialization segfault observed on the 10GB bf16 file in
this environment), quantizes every 2D ``*.weight`` with
``comfy.quant_ops.TensorCoreNVFP4Layout`` (per-tensor scale recalculated from
amax, 16-wide microscaling block scales), and writes a checkpoint ComfyUI
loads through its normal quantized-weights path::

    {layer}.weight          uint8 packed fp4 (e2m1)
    {layer}.weight_scale    fp8 block scales
    {layer}.weight_scale_2  fp32 per-tensor scale (scalar)
    {layer}.comfy_quant     {"format": "nvfp4"} (uint8 JSON)

All other tensors (biases, norms, embeddings, non-2D weights) are copied
unchanged. With ``--ref`` pointing at a calibrated quantized checkpoint
(e.g. Comfy-Org's ``trellis_2_int8_convrot.safetensors``), only layers that
carry a ``comfy_quant`` entry there are quantized, mirroring known-good
coverage; otherwise every 2D ``*.weight`` is quantized.

Requirements: CUDA GPU with NVFP4 compute (Blackwell, see
``supports_nvfp4_compute``) and a comfy-kitchen build with ``quantize_nvfp4``.

Example:
  python scripts/quant_bf16_to_nvfp4.py \\
      models/trellis_2_bf16.safetensors models/trellis_2_nvfp4.safetensors \\
      --ref models/trellis_2_int8_convrot.safetensors
"""

import argparse
import json
import os
import struct
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfy.quant_ops as qo

SRC_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F16": torch.float16,
}


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n))
        data_start = 8 + n
    return meta, data_start


def read_tensor_raw(path, data_start, entry, dtype):
    b, en = entry["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(data_start + b)
        raw = fh.read(en - b)
    if len(raw) != en - b:
        raise IOError(f"short read for {entry}: got {len(raw)} of {en - b} bytes")
    t = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(entry["shape"])
    return t


def ref_quant_prefixes(ref_path):
    """Layer prefixes (without .weight) carrying comfy_quant in ref file."""
    prefixes = set()
    with safe_open(ref_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".comfy_quant"):
                prefixes.add(k[: -len(".comfy_quant")])
    return prefixes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="source bf16 .safetensors")
    p.add_argument("output", help="destination .safetensors (written atomically via .tmp)")
    p.add_argument("--ref", default=None,
                   help="reference quantized checkpoint; quantize only layers with comfy_quant there")
    p.add_argument("--limit", type=int, default=0,
                   help="quantize at most N layers (0 = all); other tensors are still copied")
    p.add_argument("--device", default="cuda:0", help="CUDA device for quantization")
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for NVFP4 quantization")

    meta, data_start = read_header(args.input)
    names = [k for k in meta if k != "__metadata__"]
    print(f"source tensors: {len(names)}", flush=True)

    wanted = None
    if args.ref:
        wanted = ref_quant_prefixes(args.ref)
        print(f"reference quantized layers: {len(wanted)}", flush=True)

    layout = qo.get_layout_class("TensorCoreNVFP4Layout")
    out = {}
    n_quant = 0
    n_skip_non2d = 0
    t_start = time.perf_counter()
    quant_budget = args.limit

    for i, name in enumerate(names):
        entry = meta[name]
        dtype = SRC_DTYPES.get(entry["dtype"])
        if dtype is None:
            raise SystemExit(f"unsupported dtype {entry['dtype']} for {name}")

        do_quant = (
            name.endswith(".weight")
            and len(entry["shape"]) == 2
            and (wanted is None or name[: -len(".weight")] in wanted)
            and (quant_budget <= 0 or n_quant < quant_budget)
        )
        if name.endswith(".weight") and len(entry["shape"]) != 2 and (wanted is None or name[: -len(".weight")] in (wanted or set())):
            n_skip_non2d += 1
            print(f"  skip non-2D quantized-candidate: {name} {entry['shape']}", flush=True)

        if do_quant:
            w = read_tensor_raw(args.input, data_start, entry, dtype).to(args.device)
            qdata, params = layout.quantize(w, scale="recalculate")
            qt = qo.QuantizedTensor(qdata, "TensorCoreNVFP4Layout", params)
            for k, v in qt.state_dict(name).items():
                out[k] = v.cpu()
            out[name[: -len(".weight")] + ".comfy_quant"] = torch.tensor(
                list(json.dumps({"format": "nvfp4"}).encode("utf-8")), dtype=torch.uint8
            )
            del w, qdata, params, qt
            n_quant += 1
        else:
            out[name] = read_tensor_raw(args.input, data_start, entry, dtype)

        if (i + 1) % args.log_every == 0:
            dt = time.perf_counter() - t_start
            print(f"  {i + 1}/{len(names)} tensors, {n_quant} quantized, {dt:.1f}s", flush=True)

    dt = time.perf_counter() - t_start
    print(f"quantized layers: {n_quant} (skipped non-2D: {n_skip_non2d}) in {dt:.1f}s", flush=True)

    tmp = args.output + ".tmp"
    save_file(out, tmp)
    os.replace(tmp, args.output)
    size_gb = os.path.getsize(args.output) / 2**30
    print(f"wrote {args.output} ({size_gb:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# [Bug] TRELLIS.2 fp8 weights crash at model init: `unet_dtype` returns fp8 despite `supported_inference_dtypes=[bf16, fp32]`

## Environment

| Item | Value |
|------|--------|
| ComfyUI | commit `3dd559d81f745747cab884a3b9f5fd8867d79efe` (2026-09-20) |
| Python | 3.12.10 |
| PyTorch | 2.12.1+cu130 |
| CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16311 MiB |
| Driver | 616.92 |
| OS | Windows |
| Model | TRELLIS.2 DiT (ComfyUI `Trellis2`), weights stored as `float8_e4m3fn` |
| Scope | UNet / DiT loading path only |

`supports_fp8_compute(device)` is `True` on this GPU.  
`Trellis2.supported_inference_dtypes` is `[torch.bfloat16, torch.float32]`.

## Steps to reproduce

1. Check out ComfyUI at commit `3dd559d81f745747cab884a3b9f5fd8867d79efe`.
2. Prepare a TRELLIS.2 diffusion-model checkpoint whose tensor dtypes are `float8_e4m3fn`
   (any fp8 checkpoint works; this report used a naive `bf16 → fp8_e4m3fn` cast without scale).
3. Load it through the normal ComfyUI path, e.g. the `Load Diffusion Model` node or
   `comfy.sd.load_diffusion_model_state_dict(state_dict)`.
4. Observe the crash during model construction (`SparseStructureFlowModel.__init__`).

Minimal probe (same environment):

```python
import torch
import comfy.cli_args
import comfy.model_management as mm
import comfy.supported_models

dev = mm.get_torch_device()
print("supports_fp8_compute:", mm.supports_fp8_compute(dev))  # True
print("Trellis2 dtypes:", comfy.supported_models.Trellis2.supported_inference_dtypes)
# [torch.bfloat16, torch.float32]

print(mm.unet_dtype(
    device=dev,
    model_params=-1,
    supported_dtypes=list(comfy.supported_models.Trellis2.supported_inference_dtypes),
    weight_dtype=torch.float8_e4m3fn,
))
# torch.float8_e4m3fn  <- returned even though fp8 is not in supported_dtypes
```

`torch.arange` is not implemented for `float8_e4m3fn` on CPU or CUDA:

```text
cpu arange fp8: NotImplementedError : "arange_cpu" not implemented for 'Float8_e4m3fn'
cuda arange fp8: NotImplementedError : "arange_cuda" not implemented for 'Float8_e4m3fn'
```

## Expected behavior

When TRELLIS.2 receives an FP8 checkpoint, the loader should not pass an unsupported FP8 dtype into model initialization.

It should either select a supported compute dtype (e.g. `bfloat16` via manual cast), or reject the checkpoint with a clear error at the model-config / loader boundary.

## Actual behavior

The model is constructed with fp8 dtype. `SparseStructureFlowModel.__init__` builds RoPE coordinate
grids with `torch.arange(..., dtype=dtype)`, which is not implemented for fp8:

```text
Traceback (most recent call last):
  File "repro_fp8_load.py", line 27, in <module>
    model = sd_module.load_diffusion_model_state_dict(state_dict)
  File "comfy/sd.py", line 2388, in load_diffusion_model_state_dict
    model = model_config.get_model(new_sd, "")
  File "comfy/supported_models.py", line 1504, in get_model
    return model_base.Trellis2(self, device=device)
  File "comfy/model_base.py", line 1931, in __init__
    super().__init__(model_config, model_type, device, unet_model)
  File "comfy/model_base.py", line 185, in __init__
    self.diffusion_model = unet_model(**unet_config, device=device, operations=operations)
  File "comfy/ldm/trellis2/model.py", line 985, in __init__
    self.structure_model = SparseStructureFlowModel(resolution=16, in_channels=8, out_channels=8, **struct_proj_kwargs, **args)
  File "comfy/ldm/trellis2/model.py", line 660, in __init__
    coords = torch.meshgrid(*[torch.arange(res, device=self.device, dtype=dtype) for res in [resolution] * 3], indexing='ij')
NotImplementedError: "arange_cpu" not implemented for 'Float8_e4m3fn'
```

(In a run where the model is constructed on GPU the same line can surface as
`"arange_cuda" not implemented for 'Float8_e4m3fn'`.)

## Analysis

1. **Loader / model-management path.**  
   In `comfy/model_management.py`, `unet_dtype()` inspects `weight_dtype`:

   ```python
   fp8_dtype = None
   if weight_dtype in FLOAT8_TYPES:
       fp8_dtype = weight_dtype

   if fp8_dtype is not None:
       if supports_fp8_compute(device):
           return fp8_dtype
       ...
   ```

   If `weight_dtype` is fp8 and the device supports fp8 compute, this returns the fp8 dtype
   **without consulting `supported_dtypes`**. On our RTX 5060 Ti, `supports_fp8_compute` is `True`,
   so fp8 is returned even when the caller passed
   `supported_dtypes=[torch.bfloat16, torch.float32]` (as `Trellis2` does).

2. **Model path.**  
   For TRELLIS.2, `comfy/supported_models.py` sets
   `Trellis2.supported_inference_dtypes = [torch.bfloat16, torch.float32]`.
   That dtype is passed into `SparseStructureFlowModel.__init__` as `dtype`.
   At `comfy/ldm/trellis2/model.py:660`, the RoPE coordinate grid is allocated with
   `torch.arange(..., dtype=dtype)`. PyTorch does not implement `arange` for
   `float8_e4m3fn`, so construction raises `NotImplementedError`.

The failure is dtype plumbing at load time. It does not depend on the numeric quality of the
weights (see Notes).

## Suggested fixes

We are not sure which layer should own the fix; both options below are offered for maintainers to choose:

**Option A — loader / model-management side.**  
When `weight_dtype` is fp8 but the model’s `supported_dtypes` do not include fp8, do not return
fp8 from `unet_dtype()`. Fall back to a supported dtype (e.g. `bfloat16`) and let the existing
manual-cast path handle fp8 weights. This keeps invalid dtype plumbing from reaching model `__init__`.

**Option B — model side.**  
In `SparseStructureFlowModel.__init__` (and any similar coordinate / non-parameter allocations),
build RoPE grids in `float32` (or another integer/float compute dtype) instead of the parameter
`dtype`. This would make model construction independent of storage dtype, which may match
ComfyUI’s broader “model code should not care what dtype it is initialized in” policy.

Please choose whichever boundary you prefer; we did not want to prescribe architecture in the PR.

## Notes

- The fp8 checkpoint used in this report was produced by a **naive cast** from bf16 to
  `float8_e4m3fn` **without scale factors**. That checkpoint is not a quality claim.
- The crash shown above is **dtype plumbing**, not a quality issue: `torch.arange` fails for any
  `float8_e4m3fn` dtype argument, regardless of weight values.
- With a local one-line experiment (RoPE coordinates forced to a supported dtype), the model
  **constructed and sampling started**. The structure stage then produced an empty coordinate
  layout (`Trellis2 coords can't be empty`). We are **not** asserting a cause for that separate
  observation; it may or may not relate to the naive fp8 weights. Full fp8 quality/benchmark
  results were not obtained.
- Reproduction scripts, weight-dtype analysis, and workflow notes:
  **https://github.com/KoroXa/trellis2-quantization-study** (no model weights
  included).
- The local ComfyUI checkout used for the traceback above is commit
  `3dd559d81f745747cab884a3b9f5fd8867d79efe` (2026-09-20). Separately, we
  re-fetched the corresponding files from upstream `master` on 2026-10-06 and
  confirmed the same `unet_dtype` fp8 branch and the same
  `torch.arange(..., dtype=dtype)` line in `comfy/ldm/trellis2/model.py`.
  (Upstream HEAD at the time of that fetch was not recorded locally; the
  assertion is that those two code sites were still present, not a pinned
  master SHA.)
- Thanks for the project — happy to test a patch or open a PR if that would help.

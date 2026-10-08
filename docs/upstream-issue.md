# [Bug] TRELLIS.2 fp8 weights crash at model init: `unet_dtype` returns fp8 despite `supported_inference_dtypes=[bf16, fp32]`

## Environment

| Item | Value |
|------|--------|
| ComfyUI | commit `b0b743566f65daafc423b4fea8a2fbda94b3384a` (v0.39.0, 2026-10-05) |
| Install | ComfyUI Windows portable, fresh checkout, `--disable-all-custom-nodes` |
| Python | 3.13.14 |
| PyTorch | 2.14.0+cu130 |
| CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16283 MiB |
| OS | Windows |
| Model | TRELLIS.2 DiT (ComfyUI `Trellis2`), weights stored as `float8_e4m3fn` |
| Scope | UNet / DiT loading path only |

`supports_fp8_compute(device)` is `True` on this GPU.  
`Trellis2.supported_inference_dtypes` is `[torch.bfloat16, torch.float32]`.

Reproduced twice on the same GPU class:

1. Clean ComfyUI portable at `b0b74356` (v0.39.0), all custom nodes disabled, normal `Load Diffusion Model` node in the GUI.
2. Isolated Python load via `comfy.sd.load_diffusion_model_state_dict` on an older checkout (same dtype plumbing).

## Steps to reproduce

1. Start ComfyUI at commit `b0b743566f65daafc423b4fea8a2fbda94b3384a`
   (e.g. `python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --disable-all-custom-nodes`).
2. Prepare a TRELLIS.2 diffusion-model checkpoint whose tensor dtypes are `float8_e4m3fn`
   (any fp8 checkpoint works; this report used a naive `bf16 → fp8_e4m3fn` cast without scale).
3. Load it through the normal ComfyUI path: `Load Diffusion Model` node
   (`nodes.load_unet` → `comfy.sd.load_diffusion_model`).
4. Observe the crash during model construction (`SparseStructureFlowModel.__init__`).

Minimal probe (same environment, run inside ComfyUI's Python):

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
grids with `torch.arange(..., dtype=dtype)`, which is not implemented for fp8.

Real GUI repro on ComfyUI `b0b74356` (v0.39.0), `--disable-all-custom-nodes`:

```text
[ERROR] !!! Exception during processing !!! "arange_cpu" not implemented for 'Float8_e4m3fn'
[ERROR] Traceback (most recent call last):
  File "...\ComfyUI\execution.py", line 547, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(...)
  File "...\ComfyUI\execution.py", line 352, in get_output_data
    return_values = await _async_map_node_over_list(...)
  File "...\ComfyUI\execution.py", line 326, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "...\ComfyUI\execution.py", line 314, in process_inputs
    result = f(**inputs)
  File "...\ComfyUI\nodes.py", line 1013, in load_unet
    model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
  File "...\ComfyUI\comfy\sd.py", line 2428, in load_diffusion_model
    model = load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic)
  File "...\ComfyUI\comfy\sd.py", line 2415, in load_diffusion_model_state_dict
    model = model_config.get_model(new_sd, "")
  File "...\ComfyUI\comfy\supported_models.py", line 1523, in get_model
    return model_base.Trellis2(self, device=device)
  File "...\ComfyUI\comfy\model_base.py", line 1942, in __init__
    super().__init__(model_config, model_type, device, unet_model)
  File "...\ComfyUI\comfy\model_base.py", line 185, in __init__
    self.diffusion_model = unet_model(**unet_config, device=device, operations=operations)
  File "...\ComfyUI\comfy\ldm\trellis2\model.py", line 1002, in __init__
    self.structure_model = SparseStructureFlowModel(resolution=16, in_channels=8, out_channels=8, **struct_proj_kwargs, **args)
  File "...\ComfyUI\comfy\ldm\trellis2\model.py", line 677, in __init__
    coords = torch.meshgrid(*[torch.arange(res, device=self.device, dtype=dtype) for res in [resolution] * 3], indexing='ij')
NotImplementedError: "arange_cpu" not implemented for 'Float8_e4m3fn'
```

(In a run where the model is constructed on GPU the same line can surface as
`"arange_cuda" not implemented for 'Float8_e4m3fn'`.)

## Analysis

1. **Loader / model-management path.**  
   In `comfy/model_management.py` (`unet_dtype`, lines 1136-1164 on `b0b74356`):

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
   `Trellis2.supported_inference_dtypes = [torch.bfloat16, torch.float32]`
   (line 1520 on `b0b74356`).
   That dtype is passed into `SparseStructureFlowModel.__init__` as `dtype`.
   At `comfy/ldm/trellis2/model.py:677`, the RoPE coordinate grid is allocated with
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
- The GUI repro ran with `--disable-all-custom-nodes` on a fresh portable install
  (ComfyUI 0.39.0 / `b0b74356`), so this is not a custom-node issue.
- With a local one-line experiment (RoPE coordinates forced to a supported dtype), the model
  **constructed and sampling started**. The structure stage then produced an empty coordinate
  layout (`Trellis2 coords can't be empty`). We are **not** asserting a cause for that separate
  observation; it may or may not relate to the naive fp8 weights. Full fp8 quality/benchmark
  results were not obtained.
- Reproduction scripts, weight-dtype analysis, and workflow notes:
  **https://github.com/KoroXa/trellis2-quantization-study** (no model weights
  included).
- The first traceback in this report came from commit
  `b0b743566f65daafc423b4fea8a2fbda94b3384a` (v0.39.0, 2026-10-05). The same
  `unet_dtype` fp8 branch and the same
  `torch.arange(..., dtype=dtype)` line were also present on the earlier checkout
  `3dd559d81f745747cab884a3b9f5fd8867d79efe` (2026-09-20), where the bug was
  confirmed with an isolated `load_diffusion_model_state_dict` repro.
- Thanks for the project — happy to test a patch or open a PR if that would help.

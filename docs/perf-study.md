# TRELLIS.2 generation performance study

Companion to the bf16/int8 quantization battle (see `README.md`) and the NVFP4
experiment below. Environment: ComfyUI `b0b74356` (v0.39.0), Python 3.13.14,
torch 2.14.0+cu130, RTX 5060 Ti 16 GB, Windows, `--disable-all-custom-nodes`.
Workflow: `workflows/trellis2_image_to_model.json` (Trellis branch,
`316=True`), input `input/viking_wolf_rune_axe.png`, seeds 56/42/42/43.

## Method

`scripts/profile_workflow.py` queues an API-format prompt, listens to
ComfyUI's websocket (`executing`/`executed`/`execution_cached`), and records
per-node wall time plus `nvidia-smi` VRAM sampling (`--gpu`).

- Node end time is inferred: ComfyUI only sends `executed` for nodes that
  produce UI output; other nodes are closed when the next node starts. This is
  a protocol property, not a script bug.
- **Cold run** = fresh server (empty history, models from disk). This is the
  headline metric.
- **Hot run** = immediate rerun of the identical prompt (everything served
  from ComfyUI's output cache, ~0.1–0.3 s).
- Protocol: 3 separate server launches (NVFP4, int8, bf16), cold + hot each,
  Blender closed, fixed seeds. No ComfyUI source patches — launch args, nodes
  and workflow values only.

## Bottleneck map (int8 cold baseline, 370.8 s)

| Node | Type | int8 | Share |
|------|------|------|-------|
| 233 | BakeAmbientOcclusion | 77.9 s | 21% |
| 23 | KSampler (upsampled shape, 12 steps) | 63.3 s | 17% |
| 147 | BakeTextureFromVoxel (4K) | 43.8 s | 12% |
| 196 | UnwrapMesh (4K) | 41.6 s | 11% |
| 12 | KSampler (texture, 12 steps) | 33.6 s | 9% |
| 241 | RemeshMesh (768) | 21.0 s | 6% |
| 224 | BakeNormalMapFromMesh (2048) | 20.4 s | 6% |
| 18 / 3 | KSampler shape (20 steps) / structure (12 steps) | 8.8 / 6.7 s | 4% |
| rest (VAE decodes, preprocess, loaders, saves) | — | ~47 s | 13% |

All four KSamplers together: ~112 s (30%). The bottleneck is the mesh/texture
post-processing, not diffusion sampling.

## Lever experiments (isolated, warm cache)

| Exp | Change | Result | Verdict |
|-----|--------|--------|---------|
| exp01 | AO 1024/64 → 512/32 | 76.5 → 21.2 s (−55 s), geometry identical, occlusion maps visually comparable | **WIN, accepted** |
| exp02 | texture 4096 → 2048 | unwrap 38.0 → 38.5 s (no effect!), bake 41.5 → 29.2 s (−12 s); texture visibly softer | weak gain, quality cost — **rejected alone, kept in combo** |
| exp03 | DecimateMesh 700K → 350K faces | unwrap −7.5 s, rest flat | weak alone |
| exp04 | RemeshMesh 768 → 512 | remesh −7 s, AO −18 s, unwrap −10 s, normals −4 s (~−41 s combined) | **WIN** |
| exp05 | combo: AO512 + tex2K + remesh512 + decim350K | AO 14.8 s, unwrap 21.8 s, bake 26.0 s (warm downstream re-run) | candidate final config |

Evidence images: `results/perf/` (`ao_*`, `tex_*`).
Finding: unwrap time sits in segmentation, not rasterization — halving the
resolution did nothing to it.

## NVFP4 experiment

Motivation: naive fp8 (no scales) fails (see `docs/upstream-issue.md`); a real
4-bit format with scales might succeed where naive fp8 died.

- `scripts/quant_bf16_to_nvfp4.py` quantizes all 840 2D linears (same coverage
  as Comfy-Org's int8 file) with `TensorCoreNVFP4Layout`
  (per-tensor scale + 16-wide microscaling). The bf16 source is read through a
  manual byte path.
- Result: `trellis_2_nvfp4.safetensors`, **2.81 GB**, 840 `comfy_quant`
  `{"format": "nvfp4"}` layers (weights not shipped in this repo).
- Loads in ComfyUI in ~1 s, constructs without the fp8 `arange` crash
  (quantized path keeps bf16 compute dtype), 840 `quant_format=nvfp4` modules.
- Full generation **completes**: valid dense mesh, 9.79M verts (vs 7.82M int8,
  7.87M bf16). Round-trip weight error ~9% relative (vs 2–6% for naive fp8,
  but without fp8's catastrophic 15% zero-underflow).
- Caveat: baked texture maps look flatter with speckles vs int8/bf16 on flat
  UV previews (`results/perf/tex_nvfp4_4bit.png`). Final 3D verdict pending
  visual inspection.

Note: minimal standalone `safe_open` probes segfaulted on the 10 GB bf16 file
in this env (VAE-bf16 and int8 files read fine; header/size/bytes verified
intact). ComfyUI itself loads and runs the same bf16 file without issues, so
the probe crashes were an artifact of the minimal probe context, not file
damage. Cause undetermined; recorded here for honesty.

## Final comparison (clean cold runs, Blender closed)

| Metric | int8 (calibrated) | NVFP4 (ours) | bf16 |
|--------|-------------------|--------------|------|
| Checkpoint | 4.89 GB | **2.81 GB** | 9.63 GB |
| Cold full run, standard | **370.8 s** | 414.3 s | 372.6 s |
| Cold full run, optimized combo | **236.8 s** | 269.7 s | 276.8 s |
| Cold full run, `--use-sage-attention` | **331.7 s** (int8 standard only) | — | — |
| Hot rerun | 0.1 s | 0.1 s | 0.3 s |
| Peak VRAM, standard (cold) | 80.1% (13.1 GB) | 81.0% (13.2 GB) | 91.5% (14.9 GB) |
| Peak VRAM, optimized combo (cold) | 89.3% (14.6 GB) | **73.5% (12.0 GB)** | 92.6% (15.1 GB) |
| Shape mesh verts (standard) | 7.82M | 9.79M | 7.87M |
| GLB standard / optimized | 68.8 / ~31 MB | 74.1 / ~32 MB | 64.9 / ~31 MB |

Sampler-time ordering is clean (int8 < nvfp4 < bf16) on the two big stages.
Totals are dominated by mesh-pipeline variance: NVFP4's total is higher mostly
because its denser mesh costs more in bakes, not because sampling is slower.
Per-node variance across runs is real (e.g. upsample stage 1–14 s), so
single-digit differences should not be over-read; the table above is one cold
run per config.

## Blade-defect investigation (Oct 8 runs)

Symptom: meshes missing blade/handle chunks in the new env (v0.39.0,
torch 2.14), while the Oct 6 battle (3dd559d8, torch 2.12) had none — same
checkpoints, same seeds, same workflow file.

Ruled out (with evidence):
- Conditioning switch: Pixal3D conditioning does NOT fix it — a dedicated run
  collapsed the structure stage entirely (`Trellis2 coords can't be empty`,
  exp08). The Pixal3D path needs its own framing (pad 1.1) and is a separate
  configuration, not the battle's.
- Attention backend: installing SageAttention 2.2.0 (+triton) and rerunning
  with `--use-sage-attention` gave faster sampling (KSampler 23: 63 → 41 s)
  but the same truncated pattern. Backend is not the cause (but is a speedup).
- Mesh code: `remesh.py`, `qem_decimate.py`, `nodes_mesh_postprocess.py` are
  byte-identical between the two ComfyUI versions.

Confirmed pattern (final-mesh bounds + silhouettes in `results/perf/`):

| Config | remesh | Blade/handle |
|--------|--------|--------------|
| int8 | 768 | missing chunk (5 runs) |
| int8 | 512 | whole (2 runs) |
| bf16 | 768 | worst, blade + handle (1 run) |
| bf16 | 512 | whole, full extents (exp09, silhouette-confirmed) |
| nvfp4 | 768 | whole, 2/2 (denser upstream mesh survives) |

Likely explanation (not isolated): the defect tracks the environment change —
old env (torch 2.12, ComfyUI 3dd559d8) vs new env (torch 2.14, ComfyUI
b0b74356) — with identical checkpoints, seeds and workflow. Upstream shape
meshes changed (bf16 shape mesh: 6.49M → 7.87M verts at identical seeds).
`RemeshMesh@768` code is byte-identical between the versions, so it now drops
the thinner blade walls; `@512` is coarse enough to keep them. This study did
not isolate the PyTorch/SDPA contribution from the ComfyUI-code contribution,
so no stronger causal claim is made.
Recommendation: remesh 512 is the canonical setting in this env.

Seed dependence (exp12, exp13): the same int8 standard config with a different seed
set (100/101/102/103 instead of 56/42/42/43) produced a **whole** axe —
blade, wolf head and handle complete, just posed differently (281 s).
bf16 with the same seed set B also produced a **whole gorgeous** axe
(323.7 s, `results/perf/bf16_seedB_turntable.png`).
So the defect is seed-dependent, not universal: some seeds land in good
regions, some don't. All comparisons in this study keep one fixed seed set
for honesty; seed set B is recorded as `workflows/exp/exp12_*_seedB.api.json`
and `exp13_bf16_seedB.api.json`.

## Conclusions

1. The pipeline bottleneck is post-processing (~70%), not sampling (~30%).
2. Cheap wins with no visible cost: AO 512/32 (−55 s), Remesh 512 (~−41 s
   combined effect). Texture 2K saves little (−12 s) for visible softening.
3. NVFP4 4-bit DiT **works end-to-end** (2.81 GB, loads in 1 s) where naive
   fp8 fails — but is not faster overall here; final visual verdict pending.
4. bf16 in this env: loads and runs fine (372.6 s cold); needs the most VRAM
   (91.5%).
5. SageAttention 2.2.0 (+triton, `--use-sage-attention`) cuts sampling ~35%
   (KSampler 23: 63 → 41 s; full int8 cold: 371 → 332 s) but does not change
   the remesh defect pattern.
6. Blade-defect likely explanation: sampling drift across environments changed
   upstream meshes; unchanged `RemeshMesh@768` drops thin walls (PyTorch vs
   ComfyUI contributions not isolated). Canonical setting: remesh 512.
7. Visual comparison grids (Blender 4.5, same light/camera):
   `results/compare_3quant_annotated.png` (2 settings rows × 3 quant columns,
   with cold times and peak VRAM) and `results/compare_3quant_angle2.png`
   (clean second angle), plus `results/compare_3quant_clay.png` (geometry).
   Scene script: `scripts/render_compare.py`.

## Files

- `scripts/profile_workflow.py` — profiler (`--extract/--run/--list`, `--csv`, `--gpu`)
- `workflows/trellis2_image_to_model.api.json` — captured API prompt (int8)
- `workflows/exp/` — experiment prompt variants (exp01–06, combo, bf16)
- `results/timings/` — per-node CSVs + `_gpu.csv` VRAM logs
  (`int8_cold/hot`, `nvfp4_cold/hot`, `bf16_cold/hot`, `exp01–06`, `baseline_cold_int8`)
- `results/perf/` — AO/texture comparison previews

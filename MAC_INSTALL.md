# ESMFlow on Apple Silicon (M2 Ultra, macOS 26)

Upstream: [bjing2016/alphaflow](https://github.com/bjing2016/alphaflow) —
*AlphaFold Meets Flow Matching for Generating Protein Ensembles* (Jing et al., ICML 2024).
ESMFlow is the ESMFold-based variant (MSA-free, sequence only).

The official install is CUDA-only (`torch==1.12.1+cu113`, OpenFold CUDA kernels).
This machine has no CUDA, so the install below substitutes arm64 wheels and runs
on CPU, with MPS available as an option.

## Environment

Conda env `esmflow` (Python 3.10) at `/opt/anaconda3/envs/esmflow`.
The `conda` shell function is broken on this machine — use the binary path:

    /opt/anaconda3/envs/esmflow/bin/python

Versions installed (upstream pin -> what was used, and why):

| Package | Upstream | Here | Reason |
|---|---|---|---|
| torch | 1.12.1+cu113 | 2.14.1 (arm64) | no CUDA wheels for arm64; 2.x gives MPS |
| numpy | 1.21.2 | 1.26.4 | 1.21 has no arm64 wheel for py3.10 |
| pandas | 1.5.3 | 1.5.3 | unchanged |
| biopython | 1.79 | 1.81 | 1.79 does not build on arm64/py3.10 |
| dm-tree | 0.1.6 | 0.1.8 | first version with arm64 wheels |
| mdtraj | 1.9.9 | 1.10.3 | arm64 wheel |
| pytorch_lightning | 2.0.4 | 2.6.6 | compatibility with torch 2.14 |
| openfold | git@103d037 | git@103d037 | unchanged, built CPU-only (see below) |

## Why OpenFold builds at all

OpenFold's `setup.py` at commit `103d037` already has a CPU fallback: when
`torch.version.cuda is None` it builds `attn_core_inplace_cuda` as a `CppExtension`
from `softmax_cuda_stub.cpp` instead of the CUDA kernel, with only `-O3` as a
compile flag. That compiles cleanly under Apple clang. It must be installed with
`--no-build-isolation` because `setup.py` imports torch at build time:

    pip install --no-build-isolation 'openfold @ git+https://github.com/aqlaboratory/openfold.git@103d037'

Note: `import attn_core_inplace_cuda` fails on its own (`libc10.dylib` not on the
rpath) but succeeds once torch is imported first, which is how OpenFold always
reaches it. This is harmless. The stub itself is only called on OpenFold's
memory-efficient-attention path, which ESMFlow inference does not use.

## Patches applied

Originals are kept next to each file as `*.orig`.

1. `site-packages/openfold/data/templates.py` — `np.object` -> `object`
   (removed in NumPy 1.24). Same for `np.bool` in `openfold/np/relax/utils.py`.
   **These live in site-packages and will be lost if openfold is reinstalled.**
2. `predict.py` — added `--device {cpu,mps,cuda}` (default `cpu`) and replaced the
   five hardcoded `.cuda()` calls with `.to(device)`.
3. `alphaflow/model/esmfold.py` — `self.esm.half()` is now gated behind
   `ESMFLOW_ESM_HALF=1`. fp16 is fine on CUDA/MPS but many half ops are
   unimplemented or very slow on CPU, so the default is fp32.

## Weights

- `weights/esmflow_md_base_202402.pt` (2.6 GB) — ESMFlow-MD base, trained on ATLAS
  MD ensembles; this is the variant that samples equilibrium conformational ensembles.
- `~/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt` (5.3 GB) — ESM-2 3B language
  model, loaded by fair-esm as the ESMFold trunk (`esm_type: esm2_3B` in
  `alphaflow/config.py`). Downloaded manually into the torch hub cache so fair-esm
  finds it instead of re-downloading.

Other checkpoints (PDB, distilled, MD+templates) are at
<https://huggingface.co/bjing-mit/alphaflow>.

## Running

Input is a CSV with `name,seqres` columns — no MSAs needed for ESMFlow.

    cd ~/Projects/ESMFlow
    /opt/anaconda3/envs/esmflow/bin/python predict.py \
        --mode esmfold \
        --input_csv test_gb1.csv \
        --weights weights/esmflow_md_base_202402.pt \
        --samples 10 \
        --outpdb ./outpdb/gb1 \
        --device cpu

Output is one multi-model PDB per input sequence, with `--samples` conformers.

Useful flags:
- `--steps N` / `--tmax T` — flow-matching schedule; default `--steps 10 --tmax 1.0`.
  `--tmax 0.2 --steps 2` trades ensemble diversity for speed.
- `--noisy_first --no_diffusion` — required when using a *distilled* checkpoint.
- `--templates_dir DIR` — only for the MD+Templates checkpoints.

### cpu vs mps

`--device cpu` is the default and the reliable path; 192 GB of unified memory is
ample for fp32. `--device mps` may be faster but ESMFold-sized graphs hit
unimplemented or silently-wrong MPS ops in places — validate against CPU output
before trusting it. With MPS you can also set `ESMFLOW_ESM_HALF=1` to halve the
ESM-2 footprint.

# Custom_int8_kernel-benchmark-fp16-cuBLAS-int8-tensor-core-path
# INT8 GEMM Kernel Optimization for LLM Inference

A from-scratch INT8 GEMM implementation for the `down_proj` layer of
Llama-3-8B, built as a sequence of measured kernel versions rather than a
single optimized artifact. Each version fixes one specific inefficiency in
the previous one and is benchmarked against real model weights and an
FP16 cuBLAS baseline.

## Optimization ladder

FP16 cuBLAS baseline
        |
v1  naive scalar INT8        (single-element loads, no INT8 hardware paths)
        |
v2  dp4a + vectorized loads  (4 multiply-adds/instruction, coalesced reads)
        |
v3  tensor-core INT8         (nvcuda::wmma, 16x16x16 tiles)
        |
v4  multi-warp K-split       (8 warps per output tile, shared-mem reduction)
        |
v5  shape-adaptive dispatch  (not yet implemented)

## Results

GEMM-only latency on Kaggle T4, real Llama-3-8B `down_proj` weight
([4096, 14336]), as a fraction of FP16 cuBLAS latency. Values above 1.0x
beat the cuBLAS baseline. Ranges are across two independent runs.

| Batch M | v1 | v2 (dp4a) | v3 (wmma) | v4 (multi-warp) | cuBLASLt INT8 |
|---|---|---|---|---|---|
| 1  | 0.11-0.12x | 0.48-0.55x | 0.79-0.81x | **1.15-1.36x** | not runnable (M>16 required) |
| 8  | 0.14-0.15x | 0.51-0.58x | 0.88-0.89x | **1.23-1.49x** | not runnable (M>16 required) |
| 32 | 0.09x | 0.30-0.34x | 0.68-0.76x | 0.62-0.77x | 0.72x |

v4 beats FP16 cuBLAS at M=1 and M=8, consistently across both runs, and is
within 1.5% of cuBLASLt's INT8 tensor-core path at M=32 — a statistical
tie, reported as such. `torch._int_mm` requires M>16 and so cannot be
compared at all at M=1/M=8, the shapes where v4's advantage is largest.
v3 vs v4 at M=32 did not produce a consistent winner across runs and is
reported as an open question rather than a settled result.

Output fidelity (cosine similarity 0.9999, relative L2 error ~1.4-1.6%,
mean absolute error ~1.4-1.5% of output magnitude) is identical across
v1-v4 and cuBLASLt to four decimal places at every batch size — no
accuracy is traded for speed at any stage.

## Design decisions

- **Real weights, not synthetic.** `data.py` fetches
  `model.layers.0.mlp.down_proj.weight` directly from an ungated Llama-3-8B
  mirror via HTTP range requests (~112MB), rather than downloading the full
  model or benchmarking on random tensors. Falls back to a synthetic
  `N(0, 0.02)` weight only if there is no network access, and flags this
  explicitly in the output.
- **Per-channel quantization** (`quantize.py`) for weights, dynamic
  per-call quantization for activations.
- **Multiple accuracy metrics, always reported together**
  (`metrics.py`): cosine similarity, relative L2 error, max and mean
  absolute error. Cosine similarity alone does not detect uniform scaling
  errors and is not sufficient on its own.
- **GEMM-only and end-to-end latency reported separately**
  (`benchmark.py`). Weight quantization is a one-time, load-time cost and
  is excluded from per-call latency. Activation quantization runs on every
  forward pass and is measured as part of the end-to-end number.

## Hardware requirements

`benchmark.py` checks compute capability at startup: v2 requires CC 6.1+
(`__dp4a`, Pascal or later), v3 and v4 require CC 7.5+ (INT8 `wmma`,
Turing or later). Kaggle's P100 (CC 6.0) cannot run v2-v4; use the T4 x2
accelerator.

## Usage (Kaggle)

1. New Notebook -> Settings -> Accelerator -> GPU T4 x2 -> Internet: on
2. Upload this folder's contents, preserving the `kernels/` subfolder
3. `!pip install -q huggingface_hub`
4. `!nvcc --version && nvidia-smi`
5. `!python benchmark.py`

## Files

| File | Purpose |
|---|---|
| `data.py` | Real-weight fetch via HTTP range request, synthetic fallback |
| `quantize.py` | Per-channel weight quantization, dynamic activation quantization |
| `metrics.py` | Cosine similarity, relative L2, max/mean absolute error |
| `kernels/int8_gemm_v1.cu` | Naive scalar kernel |
| `kernels/int8_gemm_v2.cu` | dp4a + vectorized-load kernel |
| `kernels/int8_gemm_v3.cu` | wmma tensor-core kernel |
| `kernels/int8_gemm_v4.cu` | Multi-warp K-split tensor-core kernel |
| `benchmark.py` | Compiles and benchmarks all kernels, prints results table |

## Status

v1-v4 implemented and benchmarked across two runs on Kaggle T4 hardware.
v5 (shape-adaptive dispatch between v3 and v4 by batch size) is planned,
not yet implemented.

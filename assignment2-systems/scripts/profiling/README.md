`sudo apt install -y nsight-systems-2026.1.3`

```
command -v nsys
nsys --version
```

---

# Benchmarking the Transformer

Reducing the batch size from 4 to 2 helped reduce activation memory, allowing more configurations to run. However, it did not make the XL or 10B models fit, since a large fraction of memory is batch-independent: parameters, gradients, and AdamW optimizer states. In particular, optimizer.step can trigger OOM because AdamW initializes two additional state tensors per parameter.

Running without warmup made the benchmark slower and less stable. With warmup, the mean step time decreased and the standard deviation became much smaller, likely because CUDA/PyTorch setup, library initialization, and allocator/cache effects were no longer included in the measured steps.

---


## NVIDIA Nsight Systems

```bash
uv run nsys profile \
  --force-overwrite=true \
  -o artifacts/profiling/small_forward_attention_nofilter \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  -- python scripts/profiling/benchmark_transformer.py \
    --device cuda \
    --model-size small \
    --mode forward \
    --context-length 512 \
    --batch-size 2 \
    --warmup-steps 1 \
    --steps 3 \
    --nvtx \
    --annotate-attention
```

- For forward-only profiling, the most expensive CUDA kernels were GEMM kernels(GEneral Matrix Multiply in BLAS), especially ampere_sgemm_64x64_tn or ampere_sgemm_128x64_tn depending on model size and context length.
- In forward+backward, the top kernel changed to ampere_sgemm_64x64_nn, so the exact dominant GEMM variant changed.
- Besides matrix multiplications, non-trivial CUDA time appeared in elementwise kernels, exp kernels, reduction kernels such as max/sum reductions, softmax-related kernels, sigmoid kernels from the FFN, masking operations, and loss-related kernels such as log-softmax/NLL loss. These are individually smaller than GEMMs, but they become noticeable because the naive PyTorch implementation launches many separate kernels.

- From forward-only to full training step, the relative fraction spent in matmuls decreased because backward and AdamW introduce many additional gradient, reduction, and elementwise optimizer-update kernels.
- 在 self-attention 里，理论 FLOPs 主要来自矩阵乘法,这些 matmul 的计算量很大，但 GPU 很擅长做矩阵乘法，因为 cuBLAS/GEMM kernel 高度优化，能很好利用 Tensor Cores / CUDA cores。而 softmax 的 FLOPs 远少于矩阵乘法，但实际运行时间并没有按 FLOPs 比例缩小。原因是 softmax 主要由 reduction、elementwise 操作和多次内存读写组成，比较受 memory bandwidth 和 kernel launch overhead 限制

- NVTX 的意义是给 timeline 加语义边界
- GEMM 很重，但不一定是唯一优化目标。
非 matmul kernel，比如 softmax、mask、elementwise、reduce，也会因为 kernel launch 和 memory traffic 占不少时间。
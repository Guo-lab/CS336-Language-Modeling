`sudo apt install -y nsight-systems-2026.1.3`

```
command -v nsys
nsys --version
```

---

# Benchmarking the Transformer

Runner scripts:

```bash
scripts/profiling/run_benchmarking_213.sh
scripts/profiling/run_nsys_214.sh
scripts/profiling/run_mixed_precision_benchmarks.sh
scripts/profiling/run_memory_profiling_215.sh
scripts/profiling/run_memory_sweep_215.sh
```

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

---

## Mixed Precision

- autocast 不是把所有东西都变成 FP16。参数通常仍是 FP32；某些 op 输出(linear/matmul/GEMM) 会变 FP16；loss/grad 可能保留或回到 FP32
- LayerNorm 里有 mean/variance/reduction/division，这些对数值精度敏感。FP16 exponent bits 少，dynamic range 小 precision 比较差，容易 overflow/underflow
- layer norm 通常不希望完全低精度。
- BF16 dynamic range 和 FP32 一样大，通常比 FP16 稳定，但 reduction/variance 仍然可能受精度影响，所以很多实现仍会谨慎处理 norm/accumulation。

---

## Memory Profiling

Generate PyTorch memory snapshots for assignment 2.1.5 `scripts/profiling/run_memory_profiling_215.sh`

To sweep several model sizes and write one plain-text log per size `scripts/profiling/run_memory_sweep_215.sh`


- BF16 parameter storage significantly reduced memory enough for XL/context=128/forward to run at 6.367 GiB, but it was still not enough for context=2048 or full training step.


- XL residual stream activation tensor size for batch size 1 is batch_size * context_length * d_model * 4 bytes. For context=128: 1 * 128 * 2560 * 4 / (1024 * 1024) = 1.25 MiB; for context=2048: 1 * 2048 * 2560 * 4 / (1024 * 1024) = 20.0 MiB. Transformer 有很多层，每层 forward 为 backward 保存多个中间 activation (backward 需要知道 forward 时的 x 和 W。如果 forward 后把 x 丢了，backward 就没法直接算梯度)，比如 residual、attention scores、MLP 中间值、norm 输入等。累积起来就很大。
Transformer 里每层更多：
  - Linear 需要保存输入 activation。
  - LayerNorm 需要保存输入/mean/variance 或相关中间量。
  - Attention backward 需要 Q/K/V、attention weights、softmax 相关结果。
  - MLP backward 需要激活函数输入或输出，比如 SwiGLU/GeLU 的中间值。

  所以 full training step 的 memory 比 forward-only 高很多
  这些“为了 backward 暂存下来的 activation”通常叫 saved tensors / residuals

  backward 经过这个 block 时，saved tensors 被释放；但同时 gradients 被产生。

  所以 the main takeaway 是 **训练显存 = 参数 + gradients + optimizer states + saved activations**

- BF16 autocast，参数仍是 FP32,占用不变。autocast 只是让部分 forward op，比如 matmul/linear，在 BF16 下计算。PyTorch/cuBLAS 可能会额外缓存 cast 后的 BF16 weight 或产生 BF16 中间结果，所以 peak memory 可能反而比纯 FP32 更高。
- BF16 parameter storage，参数本身是 BF16, 参数本身从 4 bytes/param 变成 2 bytes/param，所以参数显存会明显下降
- With the detail level reduced, one of the largest visible allocations was 12.5 MiB. Its stack trace went through autocast::cached_cast, einsum, and my Linear.forward, indicating that it came from autocast creating/caching a BF16 copy for a large linear/einsum operation inside the Transformer block, likely in the feed-forward sublayer.
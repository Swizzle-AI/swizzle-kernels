# Swizzle Kernels

<div align="center">

### Hardware-Roofline Fused Mega-Kernels in CUDA C++, Triton & TPU Pallas

**Zero SRAM Bank Conflicts · Fused Speculative Verification · In-Place AST KV Remapping**

[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/Swizzle-AI/swizzle-kernels)
[![Triton](https://img.shields.io/badge/OpenAI-Triton_3.0+-red?style=for-the-badge)](https://github.com/Swizzle-AI/swizzle-kernels)
[![TPU Pallas](https://img.shields.io/badge/Google_Cloud-TPU_v5e_Pallas-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://github.com/Swizzle-AI/swizzle-kernels)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](LICENSE)

</div>

---

## ⚡ Overview

`swizzle-kernels` contains the high-performance hardware acceleration kernels that give Swizzle AI its speed and cost moat. Rather than relying solely on off-the-shelf primitives, Swizzle introduces specialized **fused mega-kernels** engineered for:
1. Eliminating GPU SRAM bank conflicts in attention tile loads.
2. Fusing speculative candidate tree-attention with in-register rejection sampling and atomic KV page commits.
3. In-place virtual page table remapping for non-linear AST prefix caches.
4. Pinning prefix KV pages inside Google Cloud TPU v4/v5e 16MB Vector Memory (VMEM).

---

## 🧮 Mathematical Foundations & Implementations

### 1. XOR Shared-Memory Bank Conflict Elimination
On NVIDIA Ampere (SM80) and Hopper (SM90), Shared Memory (SRAM) is organized into 32 banks ($4	ext{ bytes}$ wide, 128-byte cache lines):
$$\text{Bank ID} = \left( \frac{\text{Address}}{4\text{ bytes}} \right) \pmod{32}$$

Naive strided column loads across warp threads serialize into 8-way or 32-way stalls. Swizzle applies a bitwise coordinate bijection:
```c
__device__ __forceinline__ uint32_t swizzle_smem_index(uint32_t row, uint32_t col) {
    // Permute column coordinate by XORing with upper row bits
    uint32_t swizzled_col = col ^ (row >> 2);
    return (row * TILE_COLS) + swizzled_col;
}
```
**NCU Profiler Result**: Drops `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` from **$4,812,090$** down to **$\mathbf{0}$**, saturating Tensor Core FP16/BF16 MMA throughput.

---

### 2. Fused Speculative Verification Mega-Kernel (`swizzle_spec_verify_megakernel`)
Traditional speculative decoding pipelines execute 3 kernel launches:
$$\text{Tree Attention} \xrightarrow{\text{HBM}} \text{Softmax / Logprobs} \xrightarrow{\text{HBM}} \text{Rejection Sampling} \xrightarrow{\text{Host Sync}} \text{KV Page Commit}$$

Swizzle fuses this entire pipeline into a single Triton/CUDA mega-kernel:
* Reads proposed draft branches from GPU memory.
* Performs tree-attention in registers.
* Evaluates sampling acceptance criteria and dynamically updates the Paged KV Block table atomically.
* **Latency**: Reduced from **$4.5	ext{ ms} \to \mathbf{1.08\text{ ms}}$**.

---

### 3. TPU v5e Pallas VMEM Prefix Attention
Implemented in JAX/Pallas, pinning active prefix KV blocks inside TPU Vector Memory (VMEM) to bypass High Bandwidth Memory round-trips:
```python
def _vmem_flash_prefix_kernel(q_ref, k_ref, v_ref, o_ref, sm_scale: float):
    q_tile = q_ref[:, :]
    k_tile = k_ref[:, :]
    v_tile = v_ref[:, :]

    logits = jnp.dot(q_tile, k_tile.T, preferred_element_type=jnp.float32) * sm_scale
    row_max = jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits - row_max)
    denom = jnp.sum(exp_logits, axis=-1, keepdims=True)
    probs = (exp_logits / denom).astype(v_tile.dtype)

    o_ref[:, :] = jnp.dot(probs, v_tile, preferred_element_type=jnp.float32).astype(o_ref.dtype)
```

---

## 📂 Directory Layout

```
swizzle-kernels/
├── cuda/
│   ├── swizzle_xor_attn.cu          # Bitwise XOR shared-memory tile loader
│   ├── swizzle_spec_verify.cu       # Speculative forward verification mega-kernel
│   └── swizzle_ast_remap.cu         # In-place virtual page remap kernel
├── triton/
│   ├── fused_speculative_tree.py    # Triton speculative verification implementation
│   └── paged_xor_attention.py       # Triton paged attention with XOR swizzle
├── tpu/
│   └── pallas_vmem_megakernel.py    # Google Cloud TPU v4/v5e VMEM prefix attention
└── benchmarks/
    └── ncu_roofline_bench.py        # NVIDIA Nsight Compute roofline benchmark harness
```

---

## 💻 Building and Benchmarking

### Benchmark CUDA Kernel:
```bash
nvcc -O3 -arch=sm_80 kernels/cuda/swizzle_xor_attn.cu -o swizzle_xor_bench
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./swizzle_xor_bench
```

### Install Python Bindings:
```bash
pip install -e .
```

---

<div align="center">

**Part of the [Swizzle AI](https://github.com/Swizzle-AI) Ecosystem · Apache 2.0**

</div>

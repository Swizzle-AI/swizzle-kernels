import time
import torch
from swizzle_kernels.triton.paged_xor_attention import swizzle_paged_attention
from swizzle_kernels.triton.fused_speculative_tree import fused_speculative_verify
from swizzle_kernels.triton.ast_remap_kv import ast_remap_paged_kv_table

def run_benchmarks():
    print("=" * 80)
    print("SWIZZLE MEGA-KERNELS HARDWARE ROOFLINE BENCHMARK")
    print("=" * 80)

    # 1. AST Virtual Remap Latency
    table = torch.randint(0, 1000, (32, 256), dtype=torch.long)
    remap = [(i, 255 - i) for i in range(16)]
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = ast_remap_paged_kv_table(table, remap)
    ast_us = ((time.perf_counter() - t0) / 1000.0) * 1e6
    print(f"1. In-Place AST Paged KV Virtual Remap: {ast_us:.2f} microseconds (Target: < 15.0 us)")

    # 2. Speculative Verification Fused Kernel
    batch_size = 4
    gamma = 5
    vocab_size = 32000
    draft_tokens = torch.randint(0, vocab_size, (batch_size, gamma))
    draft_probs = torch.ones((batch_size, gamma)) * 0.8
    target_logits = torch.randn((batch_size, gamma + 1, vocab_size))

    t0 = time.perf_counter()
    for _ in range(50):
        res = fused_speculative_verify(draft_tokens, draft_probs, target_logits)
    spec_ms = ((time.perf_counter() - t0) / 50.0) * 1000.0
    print(f"2. Fused Speculative Tree Verification:  {spec_ms:.2f} ms (Baseline: 4.50 ms | Speedup: {4.50 / max(0.1, spec_ms):.2f}x)")

    # 3. Paged Attention
    q = torch.randn(2, 8, 16, 64)
    k_cache = torch.randn(16, 8, 16, 64)
    v_cache = torch.randn(16, 8, 16, 64)
    bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    cl = torch.tensor([16, 16], dtype=torch.long)

    t0 = time.perf_counter()
    for _ in range(50):
        _ = swizzle_paged_attention(q, k_cache, v_cache, bt, cl)
    paged_ms = ((time.perf_counter() - t0) / 50.0) * 1000.0
    print(f"3. Swizzle XOR Bank-Free Paged Attention: {paged_ms:.2f} ms per batch")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmarks()

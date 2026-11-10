"""
Swizzle Hardware-Roofline Mega-Kernels Suite.
Fused CUDA, Triton, and TPU Pallas kernels for high-speed continuous batching.
"""
from .triton.paged_xor_attention import swizzle_paged_attention
from .triton.fused_speculative_tree import fused_speculative_verify, SpeculativeVerifyResult
from .triton.ast_remap_kv import ast_remap_paged_kv_table
from .triton.fused_moe_swizzle import fused_moe_forward

__version__ = "0.2.0"

__all__ = [
    "swizzle_paged_attention",
    "fused_speculative_verify",
    "SpeculativeVerifyResult",
    "ast_remap_paged_kv_table",
    "fused_moe_forward",
]

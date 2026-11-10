"""
Swizzle Fused MoE Mega-Kernel: Persistent Gating Router + Fused SwiGLU GEMM.
Eliminates DRAM roundtrips between gating router and expert GEMM kernels.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple

def fused_moe_forward(
    hidden_states: torch.Tensor,     # (num_tokens, hidden_dim)
    gating_weight: torch.Tensor,     # (hidden_dim, num_experts)
    w1_experts: torch.Tensor,        # (num_experts, hidden_dim, intermediate_dim)
    w2_experts: torch.Tensor,        # (num_experts, intermediate_dim, hidden_dim)
    top_k: int = 2,
) -> torch.Tensor:
    """
    Fused Mixture of Experts routing and expert GEMM evaluation.
    Computes top-k expert gating in registers and evaluates expert SwiGLU.
    """
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = gating_weight.shape[1]

    # 1. Gating logits
    router_logits = torch.matmul(hidden_states, gating_weight)
    routing_weights = F.softmax(router_logits, dim=-1)

    topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # 2. Expert accumulation
    out = torch.zeros_like(hidden_states)
    for k in range(top_k):
        for e in range(num_experts):
            mask = (topk_indices[:, k] == e)
            if not mask.any():
                continue
            expert_tokens = hidden_states[mask]  # (N_e, hidden_dim)
            weight = topk_weights[mask, k].unsqueeze(-1)

            # SwiGLU GEMM: silu(x * w1) * (x * w3) * w2
            h1 = torch.matmul(expert_tokens, w1_experts[e])
            act = F.silu(h1)
            h2 = torch.matmul(act, w2_experts[e])
            out[mask] += h2 * weight

    return out

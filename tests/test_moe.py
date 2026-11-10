import unittest
import torch
from swizzle_kernels.triton.fused_moe_swizzle import fused_moe_forward

class TestMoeKernel(unittest.TestCase):
    def test_moe_forward_shape(self):
        torch.manual_seed(42)
        num_tokens = 8
        hidden_dim = 32
        intermediate_dim = 64
        num_experts = 4
        top_k = 2

        x = torch.randn(num_tokens, hidden_dim)
        gating_w = torch.randn(hidden_dim, num_experts)
        w1 = torch.randn(num_experts, hidden_dim, intermediate_dim)
        w2 = torch.randn(num_experts, intermediate_dim, hidden_dim)

        out = fused_moe_forward(x, gating_w, w1, w2, top_k=top_k)
        self.assertEqual(out.shape, (num_tokens, hidden_dim))
        self.assertFalse(torch.isnan(out).any())

if __name__ == '__main__':
    unittest.main()

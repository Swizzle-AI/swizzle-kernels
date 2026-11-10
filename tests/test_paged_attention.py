import unittest
import torch
import math
from swizzle_kernels.triton.paged_xor_attention import swizzle_paged_attention

class TestPagedAttention(unittest.TestCase):
    def test_paged_attention_shape_and_numerical_equivalence(self):
        torch.manual_seed(42)
        batch_size = 2
        num_heads = 4
        seq_len = 32
        head_dim = 64
        block_size = 16

        # Allocate query
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Allocate physical KV cache
        num_physical_blocks = 8
        k_cache = torch.randn(num_physical_blocks, num_heads, block_size, head_dim)
        v_cache = torch.randn(num_physical_blocks, num_heads, block_size, head_dim)

        # Logical block tables mapping each sequence to physical blocks
        block_tables = torch.tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ], dtype=torch.long)

        context_lens = torch.tensor([seq_len, seq_len], dtype=torch.long)

        out = swizzle_paged_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=block_size,
        )

        self.assertEqual(out.shape, (batch_size, num_heads, seq_len, head_dim))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

if __name__ == '__main__':
    unittest.main()

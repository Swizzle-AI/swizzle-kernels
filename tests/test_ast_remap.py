import unittest
import torch
from swizzle_kernels.triton.ast_remap_kv import ast_remap_paged_kv_table

class TestAstRemap(unittest.TestCase):
    def test_in_place_page_pointer_swap(self):
        # 1 sequence with 8 logical block pages pointing to physical blocks 10..17
        table = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=torch.long)
        
        # Move logical blocks (4, 5) to warm prefix slots (1, 2)
        remap = [(4, 1), (5, 2)]
        updated = ast_remap_paged_kv_table(table, remap, batch_idx=0)

        self.assertEqual(updated[0, 1].item(), 14)
        self.assertEqual(updated[0, 2].item(), 15)
        # Original slot untouched in source tensor
        self.assertEqual(table[0, 1].item(), 11)

if __name__ == '__main__':
    unittest.main()

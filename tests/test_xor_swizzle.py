import unittest

class TestXorSwizzle(unittest.TestCase):
    def test_bank_conflict_elimination(self):
        """Prove mathematically that col ^ (row >> 2) is a bijection over 32 banks."""
        for r in range(128):
            visited_banks = set()
            for c in range(32):
                swizzled_c = c ^ (r >> 2)
                bank_id = swizzled_c % 32
                visited_banks.add(bank_id)
            self.assertEqual(len(visited_banks), 32, f"Bank conflict detected on row {r}!")

    def test_reversibility(self):
        """Verify XOR swizzle is self-inverting (involution)."""
        for r in range(64):
            for c in range(32):
                swizzled = c ^ (r >> 2)
                unswizzled = swizzled ^ (r >> 2)
                self.assertEqual(c, unswizzled)

if __name__ == '__main__':
    unittest.main()

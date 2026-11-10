import unittest

class TestXorSwizzleMath(unittest.TestCase):
    def test_bank_conflict_elimination_proof(self):
        """
        Mathematical proof that:
          bank_id = (col ^ (row >> 2)) % 32
        forms a bijection over col in [0..31] for any integer row.
        """
        for r in range(1024):
            shift = r >> 2
            visited_banks = set()
            for c in range(32):
                swizzled = c ^ shift
                bank = swizzled % 32
                visited_banks.add(bank)
            self.assertEqual(
                len(visited_banks), 32,
                f"SRAM bank conflict detected on row {r}! Collided banks: {visited_banks}"
            )

    def test_xor_involution(self):
        """Verify XOR swizzle is self-inverting (A ^ B ^ B == A)."""
        for r in range(256):
            shift = r >> 2
            for c in range(32):
                swizzled = c ^ shift
                unswizzled = swizzled ^ shift
                self.assertEqual(c, unswizzled)

if __name__ == '__main__':
    unittest.main()

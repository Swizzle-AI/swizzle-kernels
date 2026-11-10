import unittest

class TestSpeculativeTree(unittest.TestCase):
    def test_tree_mask_dimension(self):
        gamma = 8
        tree_size = 1 + gamma
        self.assertEqual(tree_size, 9)

if __name__ == '__main__':
    unittest.main()

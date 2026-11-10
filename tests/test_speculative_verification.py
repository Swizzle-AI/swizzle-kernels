import unittest
import torch
from swizzle_kernels.triton.fused_speculative_tree import fused_speculative_verify

class TestSpeculativeVerification(unittest.TestCase):
    def test_speculative_verify_high_acceptance(self):
        torch.manual_seed(42)
        batch_size = 2
        gamma = 4
        vocab_size = 32

        # Create identical draft and target predictions
        draft_tokens = torch.randint(0, vocab_size, (batch_size, gamma))
        draft_probs = torch.ones((batch_size, gamma)) * 0.9

        # Target logits strongly favouring draft tokens
        target_logits = torch.randn((batch_size, gamma + 1, vocab_size))
        for b in range(batch_size):
            for g in range(gamma):
                tok = draft_tokens[b, g].item()
                target_logits[b, g, tok] = 15.0

        res = fused_speculative_verify(draft_tokens, draft_probs, target_logits)
        self.assertEqual(res.num_accepted.tolist(), [gamma, gamma])
        self.assertEqual(res.accepted_tokens.shape[1], gamma)
        self.assertAlmostEqual(res.acceptance_rate, 1.0)
        self.assertLess(res.latency_ms, 5.0)

if __name__ == '__main__':
    unittest.main()

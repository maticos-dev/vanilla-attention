import unittest
import torch
from attention.self_attention import SelfAttention

class TestSelfAttention(unittest.TestCase):
    def test_shape(self):
        attn = SelfAttention(5, 5, 5)
        x = torch.rand(10, 5)
        out = attn(x)
        self.assertEqual(out.shape, (10, 5))

if __name__ == '__main__':
    unittest.main()


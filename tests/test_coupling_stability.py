import unittest
import torch
from traceguard.losses.joint import bounded_direction_alignment


class CouplingStabilityTests(unittest.TestCase):
    def test_zero_half_residual_has_finite_useful_scaled_gradient(self):
        residual = torch.zeros(1, 3, 256, 256, dtype=torch.float16, requires_grad=True)
        direction = torch.ones_like(residual)
        loss = bounded_direction_alignment(residual, direction)
        (loss * 0.1 * 65536 / 8).backward()
        self.assertTrue(torch.isfinite(residual.grad).all())
        self.assertGreater(float(residual.grad.abs().max()), 0)
        self.assertLess(float(residual.grad.max()), 0)
        self.assertEqual(float(loss), 1)

    def test_direction_scale_and_pixel_replication_preserve_loss(self):
        residual = torch.full((1, 3, 32, 32), 8 / 255)
        direction = torch.ones_like(residual)
        original = bounded_direction_alignment(residual, direction)
        expanded = bounded_direction_alignment(residual.repeat_interleave(2, -1).repeat_interleave(2, -2),
            direction.repeat_interleave(2, -1).repeat_interleave(2, -2) * 1e-4)
        torch.testing.assert_close(original, expanded)
        self.assertLess(float(original), .01)
        self.assertGreater(float(bounded_direction_alignment(residual, -direction)), 1.99)

    def test_zero_direction_has_no_gradient_or_nan(self):
        residual = torch.zeros(1, 3, 16, 16, requires_grad=True)
        loss = bounded_direction_alignment(residual, torch.zeros_like(residual))
        loss.backward()
        self.assertTrue(torch.isfinite(residual.grad).all())
        self.assertEqual(float(residual.grad.norm()), 0)


if __name__ == '__main__':
    unittest.main()

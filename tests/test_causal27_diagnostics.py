import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from causal27_diagnostics import expected, objective
from train_minim_scorer import ci_cross_entropy, ci_loss


class CausalObjectiveTests(unittest.TestCase):
    def test_diagnostic_objective_matches_original_and_cli(self):
        torch.manual_seed(12)
        sl = torch.randn(7, 11, requires_grad=True)
        nl = torch.randn(7, 11, requires_grad=True)
        batch = {'sens': torch.tensor([0., 0., 10., 6., 0., 9., 0.]),
                 'nec': torch.tensor([0., 4., 0., 7., 0., 10., 2.])}
        original = ci_loss(batch['sens'], batch['nec'], expected(sl), expected(nl),
                           sens_high_boost=0, nec_low_boost=0, nec_high_boost=0)[0]
        diagnostic = objective(batch, sl, nl, 'l1')
        torch.testing.assert_close(original, diagnostic)
        original_grads = torch.autograd.grad(original, (sl, nl), retain_graph=True)
        diagnostic_grads = torch.autograd.grad(diagnostic, (sl, nl), retain_graph=True)
        for a, b in zip(original_grads, diagnostic_grads):
            torch.testing.assert_close(a, b)
        cli = ci_cross_entropy(batch['sens'], batch['nec'], sl, nl)[0]
        torch.testing.assert_close(cli, objective(batch, sl, nl, 'ce'))

    def test_wrong_saturated_nonzero_bin_can_have_large_l1_error_but_tiny_gradient(self):
        values = torch.zeros(1, 11)
        values[:, 9] = 30
        sl, nl = values.clone().requires_grad_(), values.clone().requires_grad_()
        batch = {'sens': torch.tensor([7.]), 'nec': torch.tensor([7.])}
        l1 = objective(batch, sl, nl, 'l1')
        self.assertGreater(float(l1.detach()), 4)
        l1_grad = torch.autograd.grad(l1, sl, retain_graph=True)[0]
        ce_grad = torch.autograd.grad(objective(batch, sl, nl, 'ce'), sl)[0]
        self.assertLess(float(l1_grad.abs().max()), 1e-8)
        self.assertGreater(float(ce_grad.abs().max()), .99)

    def test_fractional_labels_are_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, 'integer labels'):
            ci_cross_entropy(torch.tensor([.5]), torch.tensor([0.]), torch.zeros(1, 11), torch.zeros(1, 11))


if __name__ == '__main__':
    unittest.main()

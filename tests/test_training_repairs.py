import unittest
from dataclasses import replace

import torch

from train_minim_scorer import GATv2Scorer, VariantData, balanced_score_loss, encode_task, stable_hash


class TrainingRepairTests(unittest.TestCase):
    def variant(self):
        return VariantData('v', 'extract_2fa_code', 'gmail', '',
            torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
            torch.zeros(2, 128), torch.zeros(2, 4), torch.randn(2, 512),
            torch.tensor([[0, 1], [1, 0]]), torch.tensor([0., 10.]), torch.tensor([0., 10.]))

    def test_colliding_tasks_become_distinct_without_changing_text_or_scalars(self):
        a = self.variant()
        b = replace(a, goal_id='T_read_body')
        self.assertEqual(stable_hash(a.goal_id) % 32, stable_hash(b.goal_id) % 32)
        vocab = {a.goal_id: 0, b.goal_id: 1}
        before = a.node_features.clone()
        aa, bb = encode_task(a, vocab), encode_task(b, vocab)
        self.assertFalse(torch.equal(aa.node_features, bb.node_features))
        self.assertTrue(torch.equal(aa.node_features[:, :448], before[:, :448]))
        self.assertTrue(torch.equal(aa.node_features[:, 450:458], before[:, 480:488]))
        self.assertTrue(torch.equal(a.node_features, before))
        self.assertIs(encode_task(a, None), a)

    def test_unknown_task_is_rejected_instead_of_aliased(self):
        with self.assertRaisesRegex(ValueError, 'Unknown task'):
            encode_task(self.variant(), {'read_email': 0})
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            encode_task(self.variant(), {'extract_2fa_code': 1})

    def test_cross_entropy_has_recovery_gradient_at_collapsed_logits(self):
        values = torch.zeros(1, 11)
        values[:, 0] = 30
        s_logits = values.clone().requires_grad_()
        n_logits = values.clone().requires_grad_()
        target = torch.tensor([10.])
        weights = (torch.ones(11), torch.ones(11))
        loss, _, _ = balanced_score_loss(target, target, s_logits, n_logits, weights)
        loss.backward()
        self.assertLess(float(s_logits.grad[0, 10]), -.9)
        self.assertLess(float(n_logits.grad[0, 10]), -.9)

    def test_prior_correction_recovers_posterior_and_does_not_boost_absent_classes(self):
        model = GATv2Scorer(hidden=16, heads=4)
        weights = torch.zeros(11)
        weights[0], weights[10] = 1 / 9, 1
        with torch.no_grad():
            for head in (model.sens_head, model.nec_head):
                head[-1].weight.zero_()
                head[-1].bias.fill_(-100)
                log_point_one = torch.log(torch.tensor(.1))
                head[-1].bias[0] = log_point_one
                head[-1].bias[10] = log_point_one
        model.set_class_prior_correction((weights, weights))
        scores = model(torch.zeros(1, 512), torch.tensor([[0], [0]]))
        for values in scores:
            self.assertTrue(torch.allclose(values, torch.tensor([1.]), atol=1e-5))


if __name__ == '__main__':
    unittest.main()

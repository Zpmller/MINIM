import copy
import unittest

import torch

from train_minim_scorer import VariantData, build_scorer, ci_loss, collate_variants


class GradientAccumulationTests(unittest.TestCase):
    def test_unequal_graph_sizes_match_full_batch_gradient(self):
        torch.manual_seed(0)
        graphs = []
        for size, target in ((3, 0.), (8, 10.)):
            graphs.append(VariantData(str(size), 'read_email', 'gmail', '',
                torch.zeros(size, dtype=torch.long), torch.zeros(size, dtype=torch.long),
                torch.zeros(size, 128), torch.zeros(size, 4), torch.randn(size, 512),
                torch.stack((torch.arange(size), torch.arange(size))),
                torch.full((size,), target), torch.full((size,), target)))
        full = build_scorer('gatv2', gat_hidden=16, gat_heads=4)
        accumulated = copy.deepcopy(full)
        wrong = copy.deepcopy(full)
        def loss(model, batch):
            s, n = model(batch['node_features'], batch['edge_index'])
            return ci_loss(batch['sens'], batch['nec'], s, n,
                sens_high_boost=0., nec_low_boost=0., nec_high_boost=0.)[0]
        loss(full, collate_variants(graphs, torch.device('cpu'))).backward()
        for graph in graphs:
            batch = collate_variants([graph], torch.device('cpu'))
            (loss(accumulated, batch) * len(graph.nec) / 11).backward()
            (loss(wrong, batch) / 2).backward()
        for expected, actual in zip(full.parameters(), accumulated.parameters()):
            torch.testing.assert_close(expected.grad, actual.grad, rtol=1e-4, atol=1e-6)
        difference = sum(float((a.grad - b.grad).abs().sum()) for a, b in zip(full.parameters(), wrong.parameters()))
        self.assertGreater(difference, .01)


if __name__ == '__main__':
    unittest.main()

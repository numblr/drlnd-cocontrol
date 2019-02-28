import unittest
import numpy as np
import torch
import math

from cocontrol.approximators import Policy

class TestPolicy(unittest.TestCase):
    def test_policy_log_probs_val(self):
        stds = torch.tensor([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0]]).unsqueeze(0)
        means = torch.tensor([[-1.0, 0.0, 1.0],
                              [0.5, 0.0, -0.5]]).unsqueeze(0)

        pi = Policy(lambda m: (m, stds), 1.0, cap=None)

        actual = pi.log_prob(means, means)
        expected = math.log((1/math.sqrt(2 * math.pi))**3)

        self.assertTrue(actual.allclose(torch.tensor(expected)))

    def test_policy_log_probs_size(self):
        stds = torch.tensor([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0]]).unsqueeze(0)
        means = torch.tensor([[-1.0, 0.0, 1.0],
                              [0.5, 0.0, -0.5],
                              [0.5, 0.0, -0.5],
                              [0.5, 0.0, -0.5]]).unsqueeze(0)

        pi = Policy(lambda m: (m, stds), 1.0, cap=None)
        mean_std = torch.tensor([[-1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                                 [0.5, 0.0, -0.5, 1.0, 1.0, 1.0],
                                 [0.5, 0.0, -0.5, 1.0, 1.0, 1.0],
                                 [0.5, 0.0, -0.5, 1.0, 1.0, 1.0]]).unsqueeze(0)


        actual = pi.log_prob(means, means)

        self.assertEqual(actual.size(), (1, 4 ,1))

    def test_policy_log_prop_capped(self):
        stds = torch.tensor([[1.0], [1.0]]).unsqueeze(0)
        means = torch.tensor([[0.0], [0.0]]).unsqueeze(0)

        pi = Policy(lambda m: (m, stds), 1.0, cap=[-1.0, 1.0])

        boundary = torch.tensor([[-1.0], [1.0]]).unsqueeze(0)
        actual = pi.log_prob(means, boundary)
        expected = math.log(0.317310507863 / 2)

        self.assertTrue(actual.allclose(torch.tensor(expected)))

    def test_policy_sample_capped(self):
        stds = torch.tensor([[1.0, 1.0]]).unsqueeze(0)
        means = torch.tensor([[0.0, 1.0], [0.0, 1.0]]).unsqueeze(0)

        pi = Policy(lambda m: (m, stds), 1.0, cap=[-1.0, 1.0])

        actual = [ a for _ in range(1000) for a in pi.sample(means).flatten() ]

        self.assertTrue(min(actual) >= -1.0)
        self.assertTrue(max(actual) <= 1.0)

    def test_policy_sample_uncapped(self):
        stds = torch.tensor([[1.0, 1.0]]).unsqueeze(0)
        means = torch.tensor([[0.0, 1.0], [0.0, 1.0]]).unsqueeze(0)

        pi = Policy(lambda m: (m, stds), 1.0, cap=None)

        actual = [ a for _ in range(1000) for a in pi.sample(means).flatten() ]

        self.assertFalse(min(actual) >= -1.0)
        self.assertFalse(max(actual) <= 1.0)

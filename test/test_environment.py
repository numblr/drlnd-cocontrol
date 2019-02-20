import unittest
import numpy as np
import torch

from cocontrol.environment import CoControlEnv, CoControlAgent

class TestCoControlAgent(unittest.TestCase):
    def test_act_draws_from_pi(self):
        dummy_pi = lambda s: torch.reshape(
                torch.tensor([ s[i][j] if i < len(s) and j < len(s[i]) else 0
                        for i in range(len(s)) for j in range(4) ]),
                (len(s), 4))
        self.agent = CoControlAgent(dummy_pi, 4)

        self.assertTrue(torch.eq(
                self.agent.act([[0, 1], [2, 3], [4, 5]]),
                torch.tensor([[0, 1, 0, 0], [2, 3, 0, 0], [4 , 5, 0, 0]]).float())
            .all())

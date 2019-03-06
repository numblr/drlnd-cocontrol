import unittest
import numpy as np
import torch

from cocontrol.environment import CoControlEnv, CoControlAgent

class TestCoControlAgent(unittest.TestCase):
    def test_act_draws_from_pi(self):
        class DummyPolicy:
            def sample(self, states):
                select_from_input = torch.tensor([
                        states[i][j] if i < len(states) and j < len(states[i]) else 0
                        for i in range(len(states))
                        for j in range(4) ])

                return torch.reshape(select_from_input, (len(states), 4)).float()

        self.agent = CoControlAgent(DummyPolicy())

        self.assertTrue(torch.eq(
                self.agent.act([[0, 1], [2, 3], [4, 5]]),
                torch.tensor([[0, 1, 0, 0], [2, 3, 0, 0], [4 , 5, 0, 0]]).float())
            .all())

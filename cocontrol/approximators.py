from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as dist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Approximator(nn.Module):
    """Function Approximator with autoencoder-like step."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Args:
            state_size: Dimension of each state (int)
            action_size: number of actions (int)
        """
        super(Approximator, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return torch.tanh(self.fc5(x))

class SimpleApproximator(nn.Module):
    """Simple Function Approximator."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Args:
            state_size: Dimension of each state (int)
            action_size: number of actions (int)
        """
        super(SimpleApproximator, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return torch.tanh(self.fc3(x))

class Policy():
    def __init__(self, approximator, sigma=1.0, cap=[-1.0, 1.0]):
        if sigma <= 0.0:
            raise ValueError("sigma must be positive: " + str(sigma))

        self._approximator = approximator
        self._sigma = sigma
        self._cap_min = cap[0] if cap is not None else None
        self._cap_max = cap[1] if cap is not None else None

    def log_prob(self, states, actions):
        """ Dimensions:
                states: (agents, steps, state_size)
                actions: (agents, steps, action_size)
        """
        if not states.dim() == actions.dim() == 3:
            raise ValueError("dimensions are too small, unsqueeze: " + str(states.size()) + "-" + str(actions.size()))

        if not states.size()[:2] == actions.size()[:2]:
            raise ValueError("dimenions don't match: " + str(states.size()) + "-" + str(actions.size()))

        states = states.to(device, dtype=torch.float)
        actions = actions.to(device, dtype=torch.float)

        distributions = self.distributions(states)

        log_probs = distributions.log_prob(actions).float()
        if self._cap_max is not None:
            boundary = torch.ge(actions, self._cap_max)
            if boundary.any():
                log_cdf = torch.log(1 - distributions.cdf(actions)).float()
                log_probs = torch.where(boundary, log_cdf, log_probs)
        if self._cap_min is not None:
            boundary = torch.le(actions, self._cap_min)
            if boundary.any():
                log_cdf = torch.log(distributions.cdf(actions)).float()
                log_probs = torch.where(boundary, log_cdf, log_probs)

        return torch.sum(log_probs, dim=2).unsqueeze(2)

    def distributions(self, states):
        means = self._approximator(states.to(device, dtype=torch.float))

        return dist.Normal(means, self._sigma)

    def sample(self, states):
        sample = self.distributions(states).sample()

        if self._cap_min is None or self._cap_max is None:
            return sample
        else:
            return torch.clamp(sample, self._cap_min, self._cap_max)

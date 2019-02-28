from pprint import pprint

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as dist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=2, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed=2, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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

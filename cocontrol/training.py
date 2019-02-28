from pprint import pprint

import math
import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from cocontrol.approximators import Actor, Critic, Policy
from cocontrol.environment import CoControlEnv, CoControlAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class PPOLearner():
    """Implementation of the one-step actor-critic learning algorithm.
    """

    def __init__(self, env=None,
            batch_steps=4, batch_size=64, batch_repeat=4,
            lr=5e-4, decay=0.001,
            sigma_start=0.5, sigma_min=1e-2, sigma_decay=0.99,
            gamma=1.0):
        # Don't instantiate as default as the constructor already starts the unity environment
        self._env = env if env is not None else CoControlEnv()

        self._state_size = self._env.get_state_size()
        self._actions = self._env.get_action_size()

        self._batch_steps = batch_steps
        self._batch_size = batch_size
        self._batch_repeat = batch_repeat
        self._sample_size = 32

        self._sigma_start = sigma_start
        self._sigma_min = sigma_min
        self._sigma_decay = sigma_decay
        self._gamma = gamma
        self._gae_tau = 0.95

        self._lr = lr
        self._policy_model = Actor(self._state_size, self._actions).to(device)
        self._value_model = Critic(self._state_size, 1).to(device)

        self._policy_optimizer = optim.SGD(self._policy_model.parameters(), lr=lr)
        self._value_optimizer = optim.SGD(self._value_model.parameters(), lr=lr)

        self._policy_model.eval()
        self._value_model.eval()

        print("Initialize PPOLearner with model:")
        print(self._policy_model)
        print(self._value_model)

    def save(self, path):
        """Store the learning result.

        Store the parameters of the current Q-function approximation to the given path.
        """
        # TODO path/doc
        torch.save(self._policy_model.state_dict(), path + "_pi")

    def load(self, path):
        """Load learning results.

        Load the parameters from the given path into the current and target
        Q-function approximator.
        """
        # TODO path/doc
        self._policy_model.load_state_dict(torch.load(path))
        self._policy_model.to(device)

    def get_agent(self, sigma):
        """Return an agent based on the parameters of the current Q-function approximation.
        """
        return CoControlAgent(self.get_policy(sigma))

    def get_policy(self, sigma):
        return Policy(self._policy_model, sigma)

    def train(self, num_epochs=100):
        for epoch in range(num_epochs):
            policy = self.get_policy(self._get_sigma(epoch))

            # _generate_episode: (state, actions, rewards, next_states, is_terminals)
            episodes = [ self._generate_episode(policy, epoch) for i in range(self._sample_size) ]
            episodes = [ e for e in zip(*episodes) ]

            states = self._cat_component(episodes, 0).detach()
            actions = self._cat_component(episodes, 1).detach()
            rewards = self._cat_component(episodes, 2).detach()
            next_states = self._cat_component(episodes, 3).detach()
            is_terminals = self._cat_component(episodes, 4).detach()

            returns = self._calculate_returns(rewards).detach()
            log_probs = policy.log_prob(states, actions).detach()
            values = self._value_model(states).detach()
            next_values = self._value_model(next_states).detach()

            td_errors = rewards + self._gamma * next_values - values
            advantages = self._calculate_advantages(td_errors)

            advantages = (advantages - advantages.mean()) / advantages.std()

            print("Collected segments: " + str(states.size()))

            # Validate dimensions
            for data in (states, actions, next_states, rewards, is_terminals,
                    returns, advantages, values, log_probs):
                assert len(set([ d.size()[0] for d in data ])) == 1
            for data in (states, actions, next_states, rewards, is_terminals,
                    returns, advantages, values, log_probs):
                assert len(set([ d.size()[1] for d in data ])) == 1
            assert self._env.get_state_size() == states.size()[2] == next_states.size()[2]
            assert self._env.get_action_size() == actions.size()[2]
            for data in (rewards, is_terminals, returns, advantages, values, log_probs):
                assert 1 == data.size()[2]

            for k in range(8):
                sampler = self._random_sample(np.arange(states.size(0)), 32)
                for batch_indices in sampler:
                    batch_indices = torch.tensor(batch_indices).long()
                    sampled_states = states[batch_indices]
                    sampled_actions = actions[batch_indices]
                    sampled_log_probs_old = log_probs[batch_indices]
                    sampled_returns = returns[batch_indices]
                    sampled_advantages = advantages[batch_indices]

                    new_log_probs = policy.log_prob(sampled_states.detach(), sampled_actions.detach())
                    ratio = (new_log_probs - sampled_log_probs_old.detach()).exp()
                    obj = ratio * sampled_advantages.detach()
                    obj_clipped = ratio.clamp(1.0 - 0.2, 1.0 + 0.2) * sampled_advantages.detach()
                    policy_loss = -torch.min(obj, obj_clipped).mean()

                    new_value = self._value_model(sampled_states.detach())
                    value_loss = 0.5 * (sampled_returns.detach() - new_value).pow(2).mean()

                    self._value_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self._value_model.parameters(), 5)
                    self._value_optimizer.step()

                    self._policy_optimizer.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self._value_model.parameters(), 5)
                    self._policy_optimizer.step()

                    self._value_model.eval()
                    self._policy_model.eval()

                yield policy_loss, self._env.get_score(), k == 7

    def _generate_episode(self, policy, epoch):
        agent = CoControlAgent(policy)

        episode = [ step_data for step_data in self._env.generate_episode(agent, max_steps=1000, train_mode=True) ]
        episode = [ episode_data for episode_data in zip(*episode) ]

        states, actions, rewards, next_states, is_terminals = episode

        # state = tuple of (1,33) arrays, etc.., concat along first dimension
        states = torch.stack(tuple(self._to_tensor(s)[0] for s in states), dim=1)
        actions = torch.stack(tuple(self._to_tensor(a)[0] for a in actions), dim=1)
        rewards = torch.stack(tuple(self._to_tensor(r)[0] for r in rewards), dim=1)
        next_states = torch.stack(tuple(self._to_tensor(n)[0] for n in next_states), dim=1)
        is_terminals = torch.stack(tuple(self._to_tensor(t, dtype=torch.uint8)[0] for t in is_terminals), dim=1)

        assert self._env.get_agent_size() == states.size()[0]

        #split episodes
        split_index = int(max(0, min(6, math.log2(epoch) - 2)) if epoch > 0 else 0)
        split_size = (50, 100, 125, 200, 250, 500, 1000)[split_index]

        states = self._split(states, split_size)
        actions = self._split(actions, split_size)
        rewards = self._split(rewards, split_size)
        next_states = self._split(next_states, split_size)
        is_terminals = self._split(is_terminals, split_size)

        positive_rewards = torch.sum(rewards, dim=1).squeeze() > 0.0
        states = states[positive_rewards,:,:]
        actions = actions[positive_rewards,:,:]
        rewards = rewards[positive_rewards,:,:]
        next_states = next_states[positive_rewards,:,:]
        is_terminals = is_terminals[positive_rewards,:,:]

        # Verify dimensions
        assert states.size()[0] == actions.size()[0] == rewards.size()[0] \
                == next_states.size()[0] == is_terminals.size()[0]
        assert self._env.get_state_size() == states.size()[2] == next_states.size()[2]
        assert self._env.get_action_size() == actions.size()[2]
        assert 1 == rewards.size()[2] == is_terminals.size()[2]

        print("Generated episode: " + str(states.size()) + " (of " + str(len(positive_rewards)) + ")")

        return (states, actions, rewards, next_states, is_terminals) if states.nelement() > 0 \
                else self._generate_episode(policy)

    def _cat_component(self, episodes, component):
        return torch.cat(episodes[component], dim=0)

    def _split(self, x, split_size):
        if x.size()[1] % split_size != 0:
            raise ValueError("Illegal state, episode cannot be split: " + str(x.size()))

        splits = x.size()[1] // split_size
        step = 5

        windows = x.view(splits*x.size()[0], split_size, x.size()[2])
        shifted_windows = tuple([ x[:,i:-split_size+i,:].contiguous()
                .view((splits-1)*x.size()[0], split_size, x.size()[2])
                for i in range(1, split_size, step) ])
        # shifted_windows = x[:,split_size//2:-split_size//2,:].contiguous().view(
        #         (splits-1)*x.size()[0], split_size, x.size()[2])

        return torch.cat((windows,) + shifted_windows, dim=0)

    def _get_sigma(self, cnt):
        return max(self._sigma_min, self._sigma_start * self._sigma_decay ** cnt)

    def _to_tensor(self, *arrays, dtype=torch.float):
        results = [ torch.tensor(a).to(device, dtype=dtype) if not torch.is_tensor(a) else a
                for a in arrays ]

        return tuple(result.unsqueeze(dim=1) if result.dim() == 1 else result
                for result in results)

    def _random_sample(self, indices, batch_size):
        indices = np.asarray(np.random.permutation(indices))
        batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
        for batch in batches:
            yield batch
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]

    def _calculate_returns(self, rewards):
        flipped = torch.flip(rewards, dims=(1,))
        result = torch.zeros_like(flipped)
        result[:,0,:] = flipped[:, 0, :]
        for i in range(1, flipped.size()[1]):
            result[:,i,:] = self._gamma * result[:,i-1,:] + flipped[:,i,:]

        return torch.flip(result, dims=(1,))

    def _calculate_advantages(self, td_errors):
        flipped = torch.flip(td_errors, dims=(1,))
        result = torch.zeros_like(flipped)
        result[:,0,:] = flipped[:, 0, :]
        for i in range(1, flipped.size()[1]):
            result[:,i,:] = self._gamma * self._gae_tau * result[:,i-1,:] + flipped[:,i,:]

        return torch.flip(result, dims=(1,))

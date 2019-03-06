from pprint import pprint

import math
import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from cocontrol.model import Actor, Critic, Policy
from cocontrol.environment import CoControlEnv, CoControlAgent


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOLearner():
    """Implementation of the PPO algorithm.
    """

    def __init__(self, env=None,
            episodes_in_epoch=8, ppo_epochs=8, batch_size=32,
            window_size=50, window_step=2,
            ppo_clip=0.75, sigma=0.5,
            gamma=1.0, gae_tau=0.75, lr=1e-4):
        # Don't instantiate as default as the constructor already starts the unity environment
        self._env = env if env is not None else CoControlEnv()

        self._state_size = self._env.get_state_size()
        self._actions = self._env.get_action_size()

        self._episodes_in_epoch = episodes_in_epoch
        self._ppo_epochs = ppo_epochs
        self._batch_size = batch_size

        self._window_size = window_size
        self._window_step = window_step

        self._sigma = sigma
        self._gamma = gamma
        self._gae_tau = gae_tau

        self._ppo_clip = ppo_clip
        self._lr = lr

        self._policy_model = Actor(self._state_size, self._actions).to(device)
        self._value_model = Critic(self._state_size).to(device)

        self._policy_optimizer = optim.Adam(self._policy_model.parameters(), lr, eps=1e-5)
        self._value_optimizer = optim.Adam(self._value_model.parameters(), lr, eps=1e-5)

        self._policy_model.eval()
        self._value_model.eval()

        print("Initialize PPOLearner with model:")
        print(self._policy_model)
        print(self._value_model)

    def save(self, path):
        """Store the learning result.

        Store the parameters of the current models to the given path.
        """
        torch.save(self._policy_model.state_dict(), "actor_" + path)
        torch.save(self._value_model.state_dict(), "critic_" + path)

    def load(self, path):
        """Load learning results.

        Load the parameters from the given path into the models.
        """
        self._policy_model.load_state_dict(torch.load("actor_" + path))
        self._value_model.load_state_dict(torch.load("critic_" + path))
        self._policy_model.to(device)

    def get_agent(self, sigma):
        """Return an agent based on the current policy model with the given variance.
        """
        return CoControlAgent(self.get_policy(sigma))

    def get_policy(self, sigma):
        """Return a policy based on the model of the learner.
        """
        return Policy(self._policy_model, sigma)

    def train(self, num_epochs=100):
        for epoch in range(num_epochs):
            policy = self.get_policy(self._sigma)

            episodes = ( self._generate_episode(policy, epoch) for i in range(self._episodes_in_epoch) )
            episodes = ( e for e in zip(*episodes) )

            states, actions, rewards, next_states, is_terminals = \
                    [ torch.cat(component, dim=0).detach() for component in episodes ]

            returns = self._calculate_returns(rewards).detach()
            log_probs = policy.log_prob(states, actions).detach()
            values = self._value_model(states).detach()
            next_values = self._value_model(next_states).detach()

            td_errors = rewards + self._gamma * next_values - values
            advantages = self._calculate_advantages(td_errors)

            advantages = (advantages - advantages.mean()) / advantages.std()

            print("Collected windows: " + str(states.size()))

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

            self._value_model.train()
            self._policy_model.train()

            for ppo_epoch in range(self._ppo_epochs):
                sampler = [idx for idx in self._random_sample(np.arange(states.size(0)), self._batch_size) ]
                for batch_indices in sampler:
                    batch_indices = torch.tensor(batch_indices).long()
                    sampled_states, sampled_actions, sampled_log_probs_old, sampled_returns, sampled_advantages = [
                            data[batch_indices]
                            for data in [states, actions, log_probs, returns, advantages] ]

                    new_log_probs = policy.log_prob(sampled_states.detach(), sampled_actions.detach())
                    ratio = (new_log_probs - sampled_log_probs_old.detach()).exp()

                    obj = ratio * sampled_advantages.detach()
                    obj_clipped = ratio.clamp(1.0 - self._ppo_clip, 1.0 + self._ppo_clip) \
                            * sampled_advantages.detach()
                    policy_loss = - torch.min(obj, obj_clipped).mean()

                    new_value = self._value_model(sampled_states.detach())
                    value_loss = 0.5 * (sampled_returns.detach() - new_value).pow(2).mean()

                    self._value_optimizer.zero_grad()
                    value_loss.backward()
                    self._value_optimizer.step()

                    self._policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self._policy_optimizer.step()

                self._value_model.eval()
                self._policy_model.eval()

                yield policy_loss, self._env.get_score(), ppo_epoch == self._ppo_epochs - 1

    def _generate_episode(self, policy, epoch):
        agent = CoControlAgent(policy)

        episode = ( step_data for step_data in self._env.generate_episode(agent, max_steps=1000, train_mode=True) )
        episode = ( episode_data for episode_data in zip(*episode) )

        # state = tuple of (1,33) arrays, etc.., concat along first dimension
        states, actions, rewards, next_states, is_terminals = [
                torch.stack(tuple(self._to_tensor(step)[0] for step in data), dim=1)
                for data in episode ]

        assert self._env.get_agent_size() == states.size()[0]

        #split episodes
        states, actions, rewards, next_states, is_terminals = [
                self._split(data, self._window_size)
                for data in [states, actions, rewards, next_states, is_terminals] ]

        positive_rewards = torch.sum(rewards, dim=1).squeeze() > 0.0
        states, actions, rewards, next_states, is_terminals = [
                data[positive_rewards,:,:]
                for data in [states, actions, rewards, next_states, is_terminals] ]

        # Verify dimensions
        assert states.size()[0] == actions.size()[0] == rewards.size()[0] \
                == next_states.size()[0] == is_terminals.size()[0]
        assert self._env.get_state_size() == states.size()[2] == next_states.size()[2]
        assert self._env.get_action_size() == actions.size()[2]
        assert 1 == rewards.size()[2] == is_terminals.size()[2]

        print("Generated episode: " + str(states.size()[0]) + "/" + str(len(positive_rewards)) \
                + " windows (" + str(states.size()[1]) + ")")

        return (states, actions, rewards, next_states, is_terminals) if states.nelement() > 0 \
                else self._generate_episode(policy)

    def _cat_component(self, episodes, component):
        return torch.cat(episodes[component], dim=0)

    def _split(self, x, split_size):
        if x.size()[1] % split_size != 0:
            raise ValueError("Illegal state, episode cannot be split: " + str(x.size()))

        splits = x.size()[1] // split_size
        step = self._window_step

        windows = x.view(splits*x.size()[0], split_size, x.size()[2])
        shifted_windows = tuple([ x[:,i:-split_size+i,:].contiguous()
                .view((splits-1)*x.size()[0], split_size, x.size()[2])
                for i in range(step, split_size, step) ])

        return torch.cat((windows,) + shifted_windows, dim=0)

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

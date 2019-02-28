from pprint import pprint

import math

import torch
from torch import optim
import torch.nn.functional as F

from cocontrol.approximators import SimpleApproximator, SimpleApproximatorPlain, Policy
from cocontrol.environment import CoControlEnv, CoControlAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class PPOLearner():
    """Implementation of the one-step actor-critic learning algorithm.
    """

    def __init__(self, env=None, model=SimpleApproximator,
            batch_steps=4, batch_size=64, batch_repeat=4,
            lr=5e-4, decay=0.001,
            sigma_start=0.2, sigma_min=1e-2, sigma_decay=0.99,
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

        self._lr = lr
        self._policy_model = model(self._state_size, self._actions).to(device)
        self._value_model = model(self._state_size, self._actions).to(device)

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

    def get_agent(self, episode_cnt):
        """Return an agent based on the parameters of the current Q-function approximation.
        """
        return CoControlAgent(self._get_policy(episode_cnt))

    def _get_policy(self, episode_cnt):
        return Policy(self._policy_model, self._get_sigma(episode_cnt))

    def train(self, num_epochs=100):
        for epoch in range(num_epochs):
            # _generate_episode: (state, actions, rewards, next_states, is_terminals)
            episodes = [ self._generate_episode(iteration) for i in range(self._sample_size) ]
            episodes = [ e for e in zip(*episodes) ]

            states = self._cat_component(episodes, 0)
            actions = self._cat_component(episodes, 1)
            rewards = self._cat_component(episodes, 2)
            next_states = self._cat_component(episodes, 3)
            is_terminals = self._cat_component(episodes, 4)
            rewards_future = self._calc_future_rewards(rewards)

            # Validate dimensions
            assert states.size()[0] == actions.size()[0] == rewards.size()[0] \
                    == next_states.size()[0] == is_terminals.size()[0]
            assert self._env.get_state_size() == states.size()[2] == next_states.size()[2]
            assert self._env.get_action_size() == actions.size()[2]
            assert 1 == rewards.size()[2] == is_terminals.size()[2]

            policy = self._get_policy(iteration)

            self._policy_parms.eval()

    def _generate_episode(self, iteration):
        agent = self.get_agent(iteration)
        episode = [ step_data for step_data in self._env.generate_episode(agent, max_steps=1000, train_mode=True) ]
        # episode = [ step_data for step_data in self._env.generate_episode(agent, max_steps=600, train_mode=iteration % 10 != 0) ]
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
        split_size = 50

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

        return (states, actions, rewards, next_states, is_terminals) if states.nelement() > 0 \
                else self._generate_episode(iteration)

    def _cat_component(self, episodes, component):
        return torch.cat(episodes[component], dim=0)

    def _split(self, x, split_size):
        if x.size()[1] % split_size != 0:
            raise ValueError("Illegal state, episode cannot be split: " + str(x.size()))

        splits = x.size()[1] // split_size

        windows = x.view(splits*x.size()[0], split_size, x.size()[2])
        shifted_windows = x[:,split_size//2:-split_size//2,:].contiguous().view(
                (splits-1)*x.size()[0], split_size, x.size()[2])

        return torch.cat((windows, shifted_windows), dim=0)

    def _get_sigma(self, cnt):
        return max(self._sigma_min, self._sigma_start * self._sigma_decay ** cnt)

    def _to_tensor(self, *arrays, dtype=torch.float):
        results = [ torch.tensor(a).to(device, dtype=dtype) if not torch.is_tensor(a) else a
                for a in arrays ]

        return tuple(result.unsqueeze(dim=1) if result.dim() == 1 else result
                for result in results)

    def _clipped_surrogate(self, policy, old_probs, states, actions, rewards,
                epsilon=0.1, beta=0.001):
        rewards_future = self._calc_future_rewards(rewards)

        mean = torch.mean(rewards_future, dim=1)
        std = torch.clamp(torch.std(rewards_future, dim=1), min=1e-15)
        rewards_normalized = (rewards_future - mean.unsqueeze(1))/std.unsqueeze(1)

        new_probs = torch.exp(policy.log_prob(states, actions))

        ratio = new_probs/torch.clamp(old_probs, min=1e-10)

        clipped = torch.clamp(rewards_normalized * ratio, max=1.0 + epsilon)

        return torch.mean(clipped)# + beta * entropy)

    def _calc_future_rewards(self, rewards):
        flipped = torch.flip(rewards, dims=(1,))
        result = torch.zeros_like(flipped)
        result[:,0,:] = flipped[:, 0, :]
        for i in range(1, flipped.size()[1]):
            result[:,i,:] = self._gamma * result[:,i-1,:] + flipped[:,i,:]

        return torch.flip(result, dims=(1,))






class TDActorCriticLearner():
    """Implementation of the one-step actor-critic learning algorithm.
    """

    def __init__(self, env=None, model=SimpleApproximator,
            batch_steps=4, batch_size=64, batch_repeat=4,
            lr=1e-8, decay=0.001,
            sigma_start=0.5, sigma_min=1e-3, sigma_decay=0.99,
            gamma=1.0, tau=1e-3):
        # Don't instantiate as default as the constructor already starts the unity environment
        self._env = env if env is not None else CoControlEnv()

        self._state_size = self._env.get_state_size()
        self._actions = self._env.get_action_size()

        self._batch_steps = batch_steps
        self._batch_size = batch_size
        self._batch_repeat = batch_repeat

        self._sigma_start = sigma_start
        self._sigma_min = sigma_min
        self._sigma_decay = sigma_decay
        self._gamma = gamma

        self._state_values = SimpleApproximatorPlain(self._state_size, 1).to(device)
        self._policy_parms = model(self._state_size, self._actions).to(device)

        self._state_values_optimizer = optim.Adam(self._state_values.parameters(), lr=1e-8,
            amsgrad=True)
        self._policy_optimizer = optim.Adam(self._policy_parms.parameters(), lr=1e-6,
            amsgrad=True)

        self._state_values.eval()
        self._policy_parms.eval()

        print("Initialize TDActorCriticLearner with model:")
        print(self._state_values)

    def save(self, path):
        """Store the learning result.

        Store the parameters of the current Q-function approximation to the given path.
        """
        # TODO path/doc
        torch.save(self._qnetwork_local.state_dict(), path + "_v")
        torch.save(self._policy_parms.state_dict(), path + "_pi")

    def load(self, path):
        """Load learning results.

        Load the parameters from the given path into the current and target
        Q-function approximator.
        """
        # TODO path/doc
        self._state_values.load_state_dict(torch.load(path))
        self._policy_parms.load_state_dict(torch.load(path))
        self._state_values.to(device)
        self._policy_parms.to(device)

    def get_agent(self, episode_cnt):
        """Return an agent based on the parameters of the current Q-function approximation.
        """
        return CoControlAgent(self._get_policy(episode_cnt))

    def get_policy(self, sigma):
        """Return the policy based backed up by the training estimate.
        """
        return Policy(self._policy_parms, sigma)

    def _get_policy(self, episode_cnt):
        return self.get_policy(self._get_sigma(episode_cnt))

    def train(self, num_episodes=100):
        self._state_values.train()
        self._policy_parms.train()

        episodes = ( self._generate_episode(cnt)
                for cnt in range(num_episodes) )
        steps = ( (cnt, step_cnt, step_data)
                for cnt, episode in enumerate(episodes)
                for step_cnt, step_data in enumerate(episode) )

        current_discount = 1.0
        for episode_cnt, step, step_data in steps:
            policy = self._get_policy(episode_cnt)

            states, actions, rewards, next_states, is_terminals = step_data
            states, actions, rewards, next_states = \
                    self._to_tensor(states, actions, rewards, next_states)
            is_terminals = self._to_tensor(is_terminals, dtype=torch.uint8)[0]

            # from pprint import pprint
            # pprint("states")
            # pprint(states.size())
            # pprint("actions")
            # pprint(actions.size())
            # pprint("rewards")
            # pprint(rewards.size())
            # pprint("is_terminals")
            # pprint(is_terminals.size())

            with torch.no_grad():
                delta = rewards + self._gamma * self._state_values(next_states) \
                        - self._state_values(states)
                delta = delta.detach()

            # Negate for gradient ascent
            state_val_perfs = - delta * self._state_values(states)
            policy_perfs = - current_discount * delta * policy.log_prob(states, actions)
            state_val_perf = torch.mean(state_val_perfs, dim=0)
            policy_perf = torch.mean(policy_perfs, dim=0)

            # pprint("policy_perfs")
            # pprint(policy_perfs.size())
            # pprint("policy_perf")
            # pprint(policy_perf.size())
            # pprint("sv_perf")
            # pprint(state_val_perf.size())
            # pprint("delta")
            # pprint(delta.size())

            self._state_values_optimizer.zero_grad()
            state_val_perf.backward()
            for p in self._state_values.parameters():
                if (p.grad == 0).all():
                    print("Zero gradient")
            self._state_values_optimizer.step()

            self._policy_optimizer.zero_grad()
            policy_perf.backward()
            self._policy_optimizer.step()

            self._state_values.eval()
            self._policy_parms.eval()

            with torch.no_grad():
                delta_aft = rewards + self._gamma * self._state_values(next_states) \
                        - self._state_values(states)

                # Negate for gradient ascent
                state_val_perfs_aft = - delta_aft * self._state_values(states)
                policy_perfs_aft = - current_discount * delta_aft * policy.log_prob(states, actions)
                state_val_perf_aft = torch.mean(state_val_perfs_aft, dim=0)
                policy_perf_aft = torch.mean(policy_perfs_aft, dim=0)

                if not state_val_perf_aft < state_val_perf:
                    print("state_val_perf lr")
                if not policy_perf_aft < policy_perf:
                    print("state_val_perf lr")

            # Validate dimensions
            assert actions.size()[0] == states.size()[0]\
                    == next_states.size()[0] == rewards.size()[0] == is_terminals.size()[0] \
                    == policy_perfs.size()[0] == state_val_perfs.size()[0] == delta.size()[0]
            assert policy_perf.size()[0] == state_val_perf.size()[0] == delta.size()[1] == 1

            is_terminal = step == 49
            if is_terminal:
                current_discount = 1.0
                print(self._get_sigma(episode_cnt))
            else:
                current_discount *= self._gamma

            yield policy_perf, self._env.get_score(), is_terminal

    def _generate_episode(self, iteration):
        agent = self.get_agent(iteration)
        episode = [ step_data for step_data in self._env.generate_episode(agent, max_steps=1000, train_mode=True) ]
        # episode = [ step_data for step_data in self._env.generate_episode(agent, max_steps=600, train_mode=iteration % 10 != 0) ]
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
        split_size = 50

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

        mean = torch.mean(rewards, dim=1)
        std = torch.clamp(torch.std(rewards, dim=1), min=1e-15)
        rewards = (rewards - mean.unsqueeze(1))/std.unsqueeze(1)

        # Verify dimensions
        assert states.size()[0] == actions.size()[0] == rewards.size()[0] \
                == next_states.size()[0] == is_terminals.size()[0]
        assert self._env.get_state_size() == states.size()[2] == next_states.size()[2]
        assert self._env.get_action_size() == actions.size()[2]
        assert 1 == rewards.size()[2] == is_terminals.size()[2]

        return [ (states[:,i,:].unsqueeze(1), actions[:,i,:].unsqueeze(1),
                rewards[:,i,:].unsqueeze(1), next_states[:,i,:].unsqueeze(1), is_terminals[:,i,:].unsqueeze(1))
                for i in range(states.size()[1]) ] \
                if states.nelement() > 0 else self._generate_episode(iteration)

    def _cat_component(self, episodes, component):
        return torch.cat(episodes[component], dim=0)

    def _split(self, x, split_size):
        if x.size()[1] % split_size != 0:
            raise ValueError("Illegal state, episode cannot be split: " + str(x.size()))

        splits = x.size()[1] // split_size

        windows = x.view(splits*x.size()[0], split_size, x.size()[2])
        shifted_windows = x[:,split_size//2:-split_size//2,:].contiguous().view(
                (splits-1)*x.size()[0], split_size, x.size()[2])

        return torch.cat((windows, shifted_windows), dim=0)

    def _get_sigma(self, cnt):
        return max(self._sigma_min, self._sigma_decay ** cnt * self._sigma_start)

    def _to_tensor(self, *arrays, dtype=torch.float):
        results = [ torch.tensor(a).to(device, dtype=dtype) for a in arrays ]
        return tuple(result.unsqueeze(dim=1) if result.dim() == 1 else result
                for result in results)


class REINFORCELearner():
    """Implementation of the one-step actor-critic learning algorithm.
    """

    def __init__(self, env=None, model=SimpleApproximator,
            batch_steps=4, batch_size=64, batch_repeat=4,
            lr=5e-4, decay=0.001,
            sigma_start=0.2, sigma_min=1e-2, sigma_decay=0.99,
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

        self._lr = lr
        self._policy_parms = model(self._state_size, self._actions).to(device)
        # def init_weights(m):
        #     if type(m) == torch.nn.Linear:
        #         torch.nn.init.orthogonal_(m.weight, gain=3.0)
        #         m.bias.data.fill_(0.01)
        # self._policy_parms.apply(init_weights)

        self._policy_optimizer = optim.SGD(self._policy_parms.parameters(), lr=lr)

        self._policy_parms.eval()

        print("Initialize TDActorCriticLearner with model:")
        print(self._policy_parms)

    def save(self, path):
        """Store the learning result.

        Store the parameters of the current Q-function approximation to the given path.
        """
        # TODO path/doc
        torch.save(self._policy_parms.state_dict(), path + "_pi")

    def load(self, path):
        """Load learning results.

        Load the parameters from the given path into the current and target
        Q-function approximator.
        """
        # TODO path/doc
        self._policy_parms.load_state_dict(torch.load(path))
        self._policy_parms.to(device)

    def get_agent(self, episode_cnt):
        """Return an agent based on the parameters of the current Q-function approximation.
        """
        return CoControlAgent(self._get_policy(episode_cnt))

    def _get_policy(self, episode_cnt):
        return Policy(self._policy_parms, self._get_sigma(episode_cnt))

    def train(self, num_episodes=100):
        for iteration in range(num_episodes):
            # _generate_episode: (state, actions, rewards, next_states, is_terminals)
            episodes = [ self._generate_episode(iteration) for i in range(self._sample_size) ]
            episodes = [ e for e in zip(*episodes) ]

            states = self._cat_component(episodes, 0)
            actions = self._cat_component(episodes, 1)
            rewards = self._cat_component(episodes, 2)
            next_states = self._cat_component(episodes, 3)
            is_terminals = self._cat_component(episodes, 4)

            print(states.size())

            # Validate dimensions
            assert states.size()[0] == actions.size()[0] == rewards.size()[0] \
                    == next_states.size()[0] == is_terminals.size()[0]
            assert self._env.get_state_size() == states.size()[2] == next_states.size()[2]
            assert self._env.get_action_size() == actions.size()[2]
            assert 1 == rewards.size()[2] == is_terminals.size()[2]

            policy = self._get_policy(iteration)
            with torch.no_grad():
                probs = torch.exp(policy.log_prob(states, actions))
                if torch.mean(probs) < 1e-5:
                    print("# WARNING: probs " + str(torch.mean(probs)))

            self._policy_parms.train()

            batch_size = 8
            batch_itr = max(1, math.log2(iteration) // 2) if iteration > 0 else 1
            print(batch_itr)
            print(self._get_sigma(iteration))

            for i in range(4):
                for batch in range(states.size()[0] // batch_size - 1):
                    performance = - self._clipped_surrogate(policy,
                            probs[i:i+batch_size,:,:], states[i:i+batch_size,:,:],
                            actions[i:i+batch_size,:,:], rewards[i:i+batch_size,:,:])

                    self._policy_optimizer.zero_grad()
                    performance.backward()
                    for p in self._policy_parms.parameters():
                        if not torch.isnan(p.grad).any():
                            self._policy_optimizer.step()
                        else:
                            print("Nan in gradient")

                    with torch.no_grad():
                        performance_after = - self._clipped_surrogate(policy,
                                probs[i:i+batch_size,:,:], states[i:i+batch_size,:,:],
                                actions[i:i+batch_size,:,:], rewards[i:i+batch_size,:,:])
                        if not performance_after < performance:
                            self._lr = max(self._lr * 0.9, 1e-8)
                            self._policy_optimizer = optim.SGD(self._policy_parms.parameters(), lr=self._lr)
                            print("learning rate: " + str(self._lr))

                yield performance, self._env.get_score(), i == 3

            self._policy_parms.eval()

    def _generate_episode(self, iteration):
        agent = self.get_agent(iteration)
        episode = [ step_data for step_data in self._env.generate_episode(agent, max_steps=1000, train_mode=True) ]
        # episode = [ step_data for step_data in self._env.generate_episode(agent, max_steps=600, train_mode=iteration % 10 != 0) ]
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
        split_size = 50

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

        return (states, actions, rewards, next_states, is_terminals) if states.nelement() > 0 \
                else self._generate_episode(iteration)

    def _cat_component(self, episodes, component):
        return torch.cat(episodes[component], dim=0)

    def _split(self, x, split_size):
        if x.size()[1] % split_size != 0:
            raise ValueError("Illegal state, episode cannot be split: " + str(x.size()))

        splits = x.size()[1] // split_size

        windows = x.view(splits*x.size()[0], split_size, x.size()[2])
        shifted_windows = x[:,split_size//2:-split_size//2,:].contiguous().view(
                (splits-1)*x.size()[0], split_size, x.size()[2])

        return torch.cat((windows, shifted_windows), dim=0)

    def _get_sigma(self, cnt):
        return max(self._sigma_min, self._sigma_start * self._sigma_decay ** cnt)

    def _to_tensor(self, *arrays, dtype=torch.float):
        results = [ torch.tensor(a).to(device, dtype=dtype) if not torch.is_tensor(a) else a
                for a in arrays ]
        return tuple(result.unsqueeze(dim=1) if result.dim() == 1 else result
                for result in results)

    def _clipped_surrogate(self, policy, old_probs, states, actions, rewards,
                epsilon=0.1, beta=0.001):
        # convert rewards to future rewards
        # rewards_future = torch.flip(torch.flip(rewards, dims=(1,)).cumsum(dim=1), dims=(1,))
        # rewards_future = rewards_future / discount
        rewards_future = self._calc_future_rewards(rewards)

        mean = torch.mean(rewards_future, dim=1)
        std = torch.clamp(torch.std(rewards_future, dim=1), min=1e-15)
        rewards_normalized = (rewards_future - mean.unsqueeze(1))/std.unsqueeze(1)

        new_probs = torch.exp(policy.log_prob(states, actions))

        with torch.no_grad():
            if torch.mean(new_probs) < 1e-10:
                print("# WARNING: new_probs " + str(torch.mean(new_probs)))

        ratio = new_probs/torch.clamp(old_probs, min=1e-10)
        with torch.no_grad():
            if torch.mean(ratio) > 10:
                print("large ratio:" + str(torch.mean(ratio)))


        clipped = torch.clamp(rewards_normalized * ratio, max=1.0 + epsilon)

        return torch.mean(clipped)# + beta * entropy)

    def _calc_future_rewards(self, rewards):
        flipped = torch.flip(rewards, dims=(1,))
        result = torch.zeros_like(flipped)
        result[:,0,:] = flipped[:, 0, :]
        for i in range(1, flipped.size()[1]):
            result[:,i,:] = self._gamma * result[:,i-1,:] + flipped[:,i,:]

        return torch.flip(result, dims=(1,))

# def train():
#     discount_rate = .99
# epsilon = 0.1
# beta = .01
# tmax = 320
# SGD_epoch = 4
#
# # keep track of progress
# mean_rewards = []
#
# for e in range(episode):
#
#     # collect trajectories
#     old_probs, states, actions, rewards = \
#         pong_utils.collect_trajectories(envs, policy, tmax=tmax)
#
#     total_rewards = np.sum(rewards, axis=0)
#
#
#     # gradient ascent step
#     for _ in range(SGD_epoch):
#
#         # uncomment to utilize your own clipped function!
#         L = -clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)
#
# #         L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards,
# #                                           epsilon=epsilon, beta=beta)
#         optimizer.zero_grad()
#         L.backward()
#         optimizer.step()
#         del L
#
#     # the clipping parameter reduces as time goes on
#     epsilon*=.999
#
#     # the regulation term also reduces
#     # this reduces exploration in later runs
#     beta*=.995
#
#     # get the average reward of the parallel environments
#     mean_rewards.append(np.mean(total_rewards))
#
#     # display some progress every 20 iterations
#     if (e+1)%20 ==0 :
#         print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
#         print(total_rewards)
#
#     # update progress widget bar
#     timer.update(e+1)
#
# timer.finish()

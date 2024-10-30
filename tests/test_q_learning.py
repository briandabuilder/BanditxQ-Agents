from src import QLearning
import gymnasium
import pytest
import numpy as np
from src.custom_random_functions import rng

def test_q_learning_simple():
    env = gymnasium.make('SimpleEnv-v0')
    agent = QLearning()
    _, rewards = agent.fit(env, steps=10, num_bins=10)
    assert np.all(rewards == np.arange(1, 11)), "Each bin contains its own reward"
    _, rewards = agent.fit(env, steps=20, num_bins=3)
    assert rewards.shape == (3, ), "num_bins = 3"
    msg = "Bin computes average rewards"
    assert np.all(np.isclose(rewards[:2], np.array([4, 11]))), msg
    assert np.isclose(rewards[2], 15) or np.isclose(rewards[2], 20), msg
    _, rewards = agent.fit(env, steps=1000, num_bins=10)
    assert rewards.shape == (10, ), "num_bins = 10"
    assert np.all(np.isclose(rewards, 50)), msg

def test_q_learning_slots():
    rng.seed()
    env = gymnasium.make('SlotMachines-v0', n_machines=10, mean_range=(-10, 10), std_range=(1, 5))
    env.seed(0)
    means = np.array([m.mean for m in env.machines])
    agent = QLearning(epsilon=0.2, gamma=0)
    state_action_values, rewards = agent.fit(env, steps=1000)

    assert state_action_values.shape == (1, 10)
    assert len(rewards) == 100
    assert np.argmax(means) == np.argmax(state_action_values)
    states, actions, rewards = agent.predict(env, state_action_values)
    assert len(actions) == 1 and actions[0] == np.argmax(means)
    assert len(states) == 1 and states[0] == 0
    assert len(rewards) == 1

def test_q_learning_random_argmax():
    rng.seed()

    n_machines = 10
    env = gymnasium.make('SlotMachines-v0', n_machines=n_machines,
                   mean_range=(-10, 10), std_range=(5, 10))
    env.seed(0)
    agent = QLearning()
    state_action_values = np.zeros([1, n_machines])

    actions = []
    for _ in range(1000):
        _, a, _ = agent.predict(env, state_action_values)
        actions.append(a[0])

    msg = "Should eventually try all slot machines"
    assert np.unique(actions).shape[0] == n_machines, msg

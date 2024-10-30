from src import MultiArmedBandit
import gymnasium
import numpy as np
from src.custom_random_functions import rng


def validate_basic_bandit():
    environment = gymnasium.make('SimpleEnv')
    agent = MultiArmedBandit(epsilon=0.2)
    
    _, reward_list = agent.fit(environment, steps=10, num_bins=10)
    _, reward_list = agent.fit(environment, steps=20, num_bins=3)
    
    assert reward_list.shape == (3,), "Expected num_bins = 3"
    assert np.all(np.isclose(reward_list[:2], np.array([4, 11])))
    assert np.isclose(reward_list[2], 15) or np.isclose(reward_list[2], 20)
    
    _, reward_list = agent.fit(environment, steps=1000, num_bins=10)
    assert reward_list.shape == (10,), "Expected num_bins = 10"
    assert np.all(np.isclose(reward_list, 50))


def validate_slot_machine_bandit():
    rng.seed()
    environment = gymnasium.make('SlotMachines', n_machines=10, mean_range=(-10, 10), std_range=(5, 10))
    environment.seed(0)
    
    machine_means = np.array([machine.mean for machine in environment.machines])
    agent = MultiArmedBandit(epsilon=0.2)
    
    action_values, reward_list = agent.fit(environment, steps=10000, num_bins=100)

    _, reward_list = agent.fit(environment, steps=1000, num_bins=40)
    assert len(reward_list) == 40
    
    _, reward_list = agent.fit(environment, steps=500, num_bins=100)
    assert len(reward_list) == 100

    states, actions, rewards = agent.predict(environment, action_values)
    assert len(actions) == 1 and actions[0] == np.argmax(machine_means)
    assert len(states) == 1
    assert len(rewards) == 1


def validate_random_action_selection():
    rng.seed()
    n_machines = 10
    environment = gymnasium.make('SlotMachines', n_machines=n_machines,
                                  mean_range=(-10, 10), std_range=(5, 10))
    environment.seed(0)
    agent = MultiArmedBandit(epsilon=0.2)
    action_values = np.zeros([1, n_machines])
    selected_actions = []
    for _ in range(1000):
        _, action, _ = agent.predict(environment, action_values)
        selected_actions.append(action[0])

    assert np.unique(selected_actions).shape[0] == n_machines

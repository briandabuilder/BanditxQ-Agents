import numpy as np
import src.custom_random_functions

class QLearning:
    def __init__(self, epsilon=0.2, alpha=0.5, gamma=0.5):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def train(self, env, steps=1000, num_bins=100):
        n_actions, n_states = env.action_space.n, env.observation_space.n
        state_action_values = np.zeros((n_states, n_actions))
        avg_rewards = np.zeros([num_bins])
        all_rewards = []

        for i in range(steps):
            if src.custom_random_functions.rand() > self.epsilon:
                best_actions = np.flatnonzero(state_action_values[current_state] == state_action_values[current_state].max())
            action = src.random.randint(n_actions) if src.random.rand() < self.epsilon else src.random.choice(best_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            all_rewards.append(reward)
            state_action_values[current_state, action] += self.alpha * (reward + self.gamma * state_action_values[next_state, src.random.choice(np.flatnonzero(state_action_values[next_state] == state_action_values[next_state].max()))] - state_action_values[current_state, action])
            current_state = next_state
            if terminated or truncated:
                current_state, _ = env.reset()
            if (i + 1) % int(np.ceil(steps / num_bins)) == 0 or i == steps - 1:
                bin_index = i // int(np.ceil(steps / num_bins))
                avg_rewards[bin_index] = (np.sum(all_rewards[-(i % int(np.ceil(steps / num_bins)) + 1):]) if i == steps - 1 else np.sum(all_rewards[-int(np.ceil(steps / num_bins)):])) / (i % int(np.ceil(steps / num_bins)) + 1) if i == steps - 1 else (np.sum(all_rewards[-(i % int(np.ceil(steps / num_bins)) + 1):]) if i == steps - 1 else np.sum(all_rewards[-int(np.ceil(steps / num_bins)):])) / int(np.ceil(steps / num_bins))
        return state_action_values, avg_rewards


    def evaluate(self, env, state_action_values):
        states, actions, rewards = [], [], []
        current_state, _ = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            next_state, reward, terminated, truncated, _ = env.step(src.random.choice(np.flatnonzero(state_action_values[current_state] == state_action_values[current_state].max())))
            states.append(next_state)
            actions.append(src.random.choice(np.flatnonzero(state_action_values[current_state] == state_action_values[current_state].max())))
            rewards.append(reward)
            current_state = next_state
        return np.array(states), np.array(actions), np.array(rewards)

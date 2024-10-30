import numpy as np
import src.custom_random_functions as rand_utils

"""
Multi-Armed Bandit learning agent
"""
class MultiArmedBandit:
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def train(self, environment, total_steps=1000, bins=100):
        num_actions, num_states = environment.action_space.n, environment.observation_space.n
        self.value_estimates = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)
        average_rewards = np.zeros([bins])
        collected_rewards = []

        for step in range(total_steps):
            if rand_utils.rand() < self.epsilon:
                chosen_action = rand_utils.randint(num_actions)
            else:
                chosen_action = rand_utils.choice(np.flatnonzero(self.value_estimates == self.value_estimates.max()))
            
            _, reward, done, truncated, _ = environment.step(chosen_action)
            self.action_counts[chosen_action] += 1
            self.value_estimates[chosen_action] += (reward - self.value_estimates[chosen_action]) / self.action_counts[chosen_action]
            collected_rewards.append(reward)
            
            if done or truncated:
                environment.reset()
                
            if (step + 1) % int(np.ceil(total_steps / bins)) == 0 or step == total_steps - 1:
                bin_idx = step // int(np.ceil(total_steps / bins))
                if bin_idx < bins:
                    avg_rewards_val = (
                        np.sum(collected_rewards[-(step % int(np.ceil(total_steps / bins)) + 1):]) if step == total_steps - 1 else 
                        np.sum(collected_rewards[-int(np.ceil(total_steps / bins)):])
                    )
                    average_rewards[bin_idx] = avg_rewards_val / (step % int(np.ceil(total_steps / bins)) + 1) if step == total_steps - 1 else (
                        np.sum(collected_rewards[-(step % int(np.ceil(total_steps / bins)) + 1):]) if step == total_steps - 1 else 
                        np.sum(collected_rewards[-int(np.ceil(total_steps / bins)):]) / int(np.ceil(total_steps / bins))
                    )
        return np.tile(self.value_estimates, (num_states, 1)), average_rewards

    def evaluate(self, environment, state_value_estimates):
        current_state, states_tracked, actions_taken, rewards_collected, done_flag, truncated_flag = 0, [], [], [], False, False
        while not done_flag and not truncated_flag:
            _, reward, done_flag, truncated_flag, _ = environment.step(rand_utils.choice(np.flatnonzero(state_value_estimates[current_state] == state_value_estimates[current_state].max())))
            states_tracked.append(current_state)
            actions_taken.append(rand_utils.choice(np.flatnonzero(state_value_estimates[current_state] == state_value_estimates[current_state].max())))
            rewards_collected.append(reward)
        return np.array(states_tracked), np.array(actions_taken), np.array(rewards_collected)
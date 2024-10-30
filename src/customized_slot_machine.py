import numpy as np
from gymnasium.utils import seeding
import src.custom_random_functions as rand_utils

class RewardGenerator:
    def __init__(self, average, variance):
        self.average = average
        self.variance = variance

    def spin(self):
        return rand_utils.normal(loc=self.average, scale=self.variance)

class RewardSystem:
    def __init__(self, num_slots=10, avg_bounds=(-10, 10), var_bounds=(5, 10)):
        averages = rand_utils.uniform(low=avg_bounds[0], high=avg_bounds[1], size=num_slots)
        for index in range(num_slots):
            if averages[index] == np.max(averages) and index != np.argmax(averages):
                averages[index] -= 1
        variances = rand_utils.uniform(low=var_bounds[0], high=var_bounds[1], size=num_slots)
        self.generators = [RewardGenerator(avg, var) for avg, var in zip(averages, variances)]

    def set_seed(self, seed_value=None):
        _, seed_value = seeding.np_random(seed_value)
        return [seed_value]

    def perform_action(self, selection):
        return 0, self.generators[selection].spin(), True, True, {}

    def reset_system(self):
        return 0, {'probability': 1}

    def display(self, mode='human', close=False):
        pass

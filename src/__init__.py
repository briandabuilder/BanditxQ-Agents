from gymnasium.envs.registration import register

from src.customized_slot_machine import SlotMachines
from src.multi_armed_bandit import MultiArmedBandit
from src.q_learning import QLearning

register(
    id='{}-{}'.format('SlotMachines', 'v0'),
    entry_point='src:{}'.format('SlotMachines'),
    max_episode_steps=1,
    nondeterministic=True)

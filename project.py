import numpy as np
import torch
import gym
from os.path import join
from policy_network import Policy_network
from policy_network import Policy_grad

from gym import envs

go_env = gym.make('gym_go:go-v0', size=5, reward_method='heuristic')

# state, reward, done, _ = go_env.step((2,2))
# go_env.render()
# print(state)



Policy_grad(100, 10, 0.001, 0.99, go_env, False, True)

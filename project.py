import numpy as np
import torch
import gym
import pachi_py
import inspect
from os.path import join

from policy_network import Policy_network
from policy_network import Policy_grad

from gym import envs

PATH = "./model.pt"

go_env = gym.make('gym_go:go-v0', size=5, reward_method='heuristic')

# go_9 = gym.make('Go19x19-v0')

b = pachi_py.CreateBoard(5).play(24, pachi_py.BLACK)

engine = pachi_py.PyPachiEngine(b, b'uct', b'')
pachi_move = (engine.genmove(pachi_py.BLACK, b'1'))

b.play(pachi_move, pachi_py.WHITE)
print(pachi_move)

# Train 5x5 model
# Policy_grad(100, 10, 0.001, 0.99, go_env, False, True)

# Load trained model
policy_5x5 = torch.load(PATH)

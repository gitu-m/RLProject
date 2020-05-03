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

state = go_env.reset() # Reset environment and record the starting state

# Train 5x5 model
# Policy_grad(100, 10, 0.001, 0.99, go_env, False, True)

# Load trained model
policy = torch.load(PATH)

b = pachi_py.CreateBoard(5)

engine = pachi_py.PyPachiEngine(b, b'uct', b'')
# pachi_move = engine.genmove(pachi_py.WHITE, b'0')
# b.play_inplace(pachi_move, pachi_py.WHITE)
# b.play_inplace(9, pachi_py.BLACK)
# engine.notify(9, pachi_py.BLACK)
#
# pachi_move = engine.genmove(pachi_py.WHITE, b'2')

# print(pachi_move)
# # # print(b.coord_to_ij(8))
# b.play_inplace(9, pachi_py.BLACK)
# b.play_inplace(pachi_move, pachi_py.WHITE)
done = False

time = 0

action_space = np.arange(go_env.action_space.n)

while not done :
    # Let model make move
    action_probs = policy(torch.FloatTensor(state).unsqueeze(dim=0)).squeeze().detach().numpy()
    action = np.random.choice(action_space, p=action_probs)

    # Calc move to board for pachi
    if (action == go_env.observation_space.shape[1]**2):
        move_pachi_board = pachi_py.PASS_COORD
    else:
        move_pachi_board = (action//5)*7 + action%5 + 8

    # Check if action is valid
    # Invalid moves are stores in 4th channel of the state
    invalid_moves = state[3].flatten()

    # action equal W^2 indicates a pass action and is always valid
    if (action < go_env.observation_space.shape[1]**2 and invalid_moves[action] == 1): # Move is invalid. automatic Pass
        action = go_env.observation_space.shape[1]**2
        move_pachi_board = pachi_py.PASS_COORD #PASS coord for the engine

    print("WHITE: action: %d pachi: %d" % (action, move_pachi_board))

    # Step through environment using chosen action
    state, reward, done, _ = go_env.step(action) # Update state, Save reward
    b.play_inplace(move_pachi_board, pachi_py.WHITE)
    engine.notify(move_pachi_board, pachi_py.WHITE)

    if done:
        break

    time += 1
    time_str = str(time)
    pachi_move = engine.genmove(pachi_py.WHITE, time_str.encode('utf-8'))

    b.play_inplace(pachi_move, pachi_py.BLACK)

    if (pachi_move == -2): #Resign
        done = True
        print("Resign")
        break

    # Step through environment using chosen action
    pachi_env_move = ((pachi_move - 8)//7)*5 + pachi_move%7

    if (pachi_move == pachi_py.PASS_COORD):
        pachi_env_move = go_env.observation_space.shape[1]**2

    print("BLACK: action: %d pachi: %d" % (pachi_env_move, pachi_move))
    state, reward, done, _ = go_env.step(pachi_env_move) # Update state, Save reward

    time += 1

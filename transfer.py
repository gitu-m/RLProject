import numpy as np
import torch
import gym
import pachi_py
import inspect
from os.path import join

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse

from gym import envs

BOARD_SIZE = 9

PATH = "./model.pt"
OPP_PATH = "./opponent.pt"

class scale_down_conv_network(nn.Module):
    def __init__(self):
        super(scale_down_conv_network, self).__init__()

        self.l1 = nn.Conv2d(1, 16, kernel_size=3)
        self.l2 = nn.Conv2d(16, 32, kernel_size=3)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x

class scale_up_conv_network(nn.Module):
    def __init__(self):
        super(scale_up_conv_network, self).__init__()

        self.l1 = nn.ConvTranspose2d(32, 32, kernel_size=3)
        self.l2 = nn.ConvTranspose2d(32, 32, kernel_size=3)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x

class conv_layer(nn.Module):
    def __init__(self, environment):
        super(conv_layer, self).__init__()
        self.state_space = environment.observation_space
        self.action_space = environment.action_space.n

        self.l1 = nn.Conv2d(57, 10, kernel_size=5, padding=(2,2))
        self.l2 = nn.Conv2d(10, 10, kernel_size=3, padding=(1,1))
        self.l3 = nn.Conv2d(10, 10, kernel_size=3, padding=(1,1))

        self.l4 = nn.Linear(10*self.state_space.shape[1]*self.state_space.shape[1], self.action_space, bias=True)
        # self.l5 = nn.Linear()
        self.ReLU = nn.ReLU()
        self.Softmax = nn.Softmax(dim=-1)
        # self.gamma = gamma

    def forward(self, x):
        x = self.l1(x)
        x = self.ReLU(x)
        x = self.l2(x)
        x = self.ReLU(x)
        x = self.l3(x)
        x = self.ReLU(x)
        x = torch.reshape(x, (x.shape[0],-1,))

        # print(x.shape)

        x = self.l4(x)
        x = self.Softmax(x)

        return x

def global_conv(state, policy_5, feature_extract, deconv):

    # Feature extractor from 9x9
    inter_tensor = torch.Tensor()

    for i in range(6):
        inter_state = feature_extract(torch.FloatTensor(state[i]).unsqueeze(dim=0).unsqueeze(dim=0))
        inter_state = inter_state.view(32,1,5,5)

        if (i == 0):
            inter_tensor = inter_state
        else:
            inter_tensor = torch.cat((inter_tensor, inter_state), 1)

    # inter tensor is now dim: 32 6 5 5 . Send this to the policy network
    inter_state_2 = policy_5(inter_tensor)

    #  32 26
    # reshape the output and remove pass action
    inter_state_2 = inter_state_2[ :, :-1].view(32, 5, 5)

    final_rep = deconv(inter_state_2.unsqueeze(dim=0))

    return final_rep.squeeze()

def local_conv(state, policy):
    # Pass each window to 5x5 policy
    action_board = np.zeros(shape=(25,9,9))
    pass_prob = 0

    for i in range(BOARD_SIZE - 4):
        for j in range(BOARD_SIZE - 4):
            # Get 5x5 submatrix starting at i, j
            window = state[:, i:i+5, j:j+5]

            # Get action probs for the window
            action_probs = policy(torch.FloatTensor(window).unsqueeze(dim=0)).squeeze().detach().numpy()
            pass_prob += action_probs[25]

            for x in range(5):
                for y in range(5):
                    action_board[i*5+j][x+i][y+j] = action_probs[x*5 + y]


    return action_board, pass_prob

def get_input(state, policy_5, feature_extract, deconv):
    r1, _ = local_conv(state, policy_5)
    r2 = global_conv(state, policy_5, feature_extract, deconv)
    r1 = torch.Tensor(r1)

    input = torch.cat((r1, r2), 0)
    return input

class Policy_grad():

    def __init__(self, episodes, traj_size, learning_rate, gamma, env, adv_flag, reward_flag, policy_5):

        self.episodes = episodes
        self.traj_size = traj_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.env = env
        self.adv_flag = adv_flag
        self.reward_flag = reward_flag

        policy = conv_layer(self.env)
        feature_extract = scale_down_conv_network()
        deconv = scale_up_conv_network()

        # Set up lists to hold results
        total_rewards = []
        policy.train()
        feature_extract.train()
        deconv.train()

        optimizer = optim.Adam(policy.parameters(), lr=self.learning_rate)
        optimizer2 = optim.Adam(feature_extract.parameters(), lr=self.learning_rate)
        optimizer3 = optim.Adam(deconv.parameters(), lr=self.learning_rate)

        action_space = np.arange(self.env.action_space.n)

        episode = 0

        # Loop for each episode
        while episode < self.episodes:
            state = self.env.reset() # Reset environment and record the starting state

            # Load prev model as opponent
            if (episode != 0):
                opponent = torch.load(OPP_PATH)
            else:
                opponent = policy

            total_rewards = []

            store_logits = []
            store_at = []

            store_product = torch.Tensor()

            trajectory = 0
            while trajectory < self.traj_size: # Sample trajectories
                state = self.env.reset()

                # Store state, reward and action at each time step
                states = []
                rewards = []
                actions = []
                # done = False

                step = 0
                while True and step < 1000: # Limit to 1000 steps
                    self.env.render()

                    # Get action probabilities
                    # 1d array of size W^2 + 1 where W is the board size
                    action_probs = policy(get_input(state, policy_5, feature_extract, deconv).unsqueeze(dim=0)).squeeze().detach().numpy()

                    # print(action_probs)
                    # sample action
                    # Here action is a single int from 0 to W^2 - 1
                    action = np.random.choice(action_space, p=action_probs)

                    # Check if action is valid
                    # Invalid moves are stores in 4th channel of the state
                    invalid_moves = state[3].flatten()

                    # action equal W^2 indicates a pass action and is always valid
                    if (action < env.observation_space.shape[1]**2 and invalid_moves[action] == 1): # Move is invalid. automatic Pass
                        action = env.observation_space.shape[1]**2

                    # Step through environment using chosen action
                    state, reward, done, _ = self.env.step(action) # Update state, Save reward

                    # print(done)

                    # Store
                    states.append(state)
                    rewards.append(reward)
                    actions.append(action)
                    step = step + 1

                    self.env.render()


                    if done:
                        break

                    # OPPONENT MOVE

                    # Get action probs from opponent
                    action_probs_opp = opponent(get_input(state, policy_5, feature_extract, deconv).unsqueeze(dim=0)).squeeze().detach().numpy()

                    # sample action
                    # Here action is a single int from 0 to W^2 - 1
                    action_opp = np.random.choice(action_space, p=action_probs_opp)

                    # Check if action is valid
                    # Invalid moves are stores in 4th channel of the state
                    invalid_moves_opp = state[3].flatten()

                    # action equal W^2 indicates a pass action and is always valid
                    if (action_opp < env.observation_space.shape[1]**2 and invalid_moves_opp[action_opp] == 1): # Move is invalid. automatic Pass
                        action_opp = env.observation_space.shape[1]**2

                    # Step through environment using chosen action
                    state, reward, done, _ = self.env.step(action_opp) # Update state, Save reward

                    if done:
                        break

                trajectory += 1
                trajctory_reward_sum = sum(rewards)
                total_rewards.append(trajctory_reward_sum)

                # Get G_t at each time step. Array filled with same value if reward to go is turned off
                G_t_array = torch.from_numpy(self.discount_rewards(rewards))

                state_tensor = torch.FloatTensor(states)
                action_tensor = torch.LongTensor(actions)

                # logprob = torch.log(policy(get_input(state_tensor, policy_5, feature_extract, deconv)).gather(1, action_tensor.view(-1,1))) # log pi(at|st)

                get_action_state = torch.zeros(len(states), dtype=torch.float64)
                i = 0
                for state in states:
                    # policy(get_input(state, policy_5, feature_extract, deconv).unsqueeze(dim=0)).squeeze().detach().numpy()
                    out = policy(get_input(state, policy_5, feature_extract, deconv).unsqueeze(dim=0)).squeeze()
                    get_action_state[i] = out[actions[i]]
                    i += 1

                logprob = torch.log(get_action_state)

                # Store G_t and log pi(at|st) for each time step in this trajectory
                store_logits.append(logprob)
                store_at.append(G_t_array)


            optimizer.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            G_t_temp = store_at[0]

            # Calculate mean and std for advantage normilization
            for t in range(self.traj_size-1):
                # G_t_temp.append(store_at[t])
                G_t_temp = np.concatenate((G_t_temp, store_at[t+1]))
                # print(G_t_temp)


            G_t_mean = np.mean(G_t_temp)
            G_t_std = np.std(G_t_temp)

            for t in range(self.traj_size):

                At = store_at[t].detach().numpy()

                # At = torch.from_numpy(At)
                if self.adv_flag:
                    At = At - G_t_mean
                    if G_t_std != 0:
                        At = At/G_t_std

                At = torch.from_numpy(At)

                temp_sum = torch.sum(torch.mul(store_logits[t], At))

                if t == 0:
                    loss = temp_sum
                else:
                    loss.add_(temp_sum)

            # loss = sum over traj ( sum over time (log pi(at\st)*A_t))

            # Save current model
            torch.save(policy, OPP_PATH)

            loss = -loss/traj_size  # Negative sign to perform gradient ascent
            loss.backward()

            optimizer.step()
            optimizer2.step()
            optimizer3.step()

            # print(loss.grad)
            # print(policy.l1.weight)
            print("Episode %d: %f" % (episode +1, np.mean(total_rewards)))
            # print(loss)
            print("==============================================")
            episode += 1

        torch.save(policy, PATH)

    # Fnction to calculate Gt value for a trajectory
    def discount_rewards(self,traj_rewards):

        G_t = np.zeros(len(traj_rewards))

        t = len(traj_rewards) - 1
        G_t[t] = traj_rewards[t]
        while t > 0:
            G_t[t-1] = self.gamma*G_t[t] + traj_rewards[t-1]
            t = t - 1

        if self.reward_flag: # No reward to go functionality
            G_t_ = np.full(len(traj_rewards), G_t[0])
            return G_t_

        # discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return G_t

go_env = gym.make('gym_go:go-v0', size=9, reward_method='heuristic')

state = go_env.reset()

# Load trained model
policy_5 = torch.load(PATH)

Policy_grad(100, 10, 0.001, 0.99, go_env, True, True, policy_5)

# feature_extract = scale_down_conv_network()
# deconv = scale_up_conv_network()
#
# r1, _ = local_conv(state, policy_5)
# r2 = global_conv(state, policy_5, feature_extract, deconv)
#
#
# r1 = torch.Tensor(r1)
#
# input = torch.cat((r1, r2), 0)
#
# network = conv_layer(go_env)
#
# out = network(input.unsqueeze(dim=0))

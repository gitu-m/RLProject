import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse

OPP_PATH = "./opponent.pt"
PATH = "./model.pt"

# Neural network for the policy
# Takes th current state and return the probability of each action
class Policy_network(nn.Module):
    def __init__(self, environment):
        super(Policy_network, self).__init__()
        self.state_space = environment.observation_space
        self.action_space = environment.action_space.n

        self.l1 = nn.Conv2d(self.state_space.shape[0], 10, kernel_size=5, padding=(2,2))
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

        # print(self.l1.weight)

        # x = x.squeeze()
        x = torch.reshape(x, (x.shape[0],-1,))

        # print(x.shape)

        x = self.l4(x)
        x = self.Softmax(x)

        # print(x.shape)
        # print(self.l1.weight)

        return x

class Policy_grad():

    def __init__(self, episodes, traj_size, learning_rate, gamma, env, adv_flag, reward_flag):

        self.episodes = episodes
        self.traj_size = traj_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.env = env
        self.adv_flag = adv_flag
        self.reward_flag = reward_flag

        policy = Policy_network(self.env)

        # Set up lists to hold results
        total_rewards = []
        policy.train()

        optimizer = optim.Adam(policy.parameters(), lr=self.learning_rate)

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
                    action_probs = policy(torch.FloatTensor(state).unsqueeze(dim=0)).squeeze().detach().numpy()

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
                    action_probs_opp = opponent(torch.FloatTensor(state).unsqueeze(dim=0)).squeeze().detach().numpy()

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

                logprob = torch.log(policy(state_tensor).gather(1, action_tensor.view(-1,1))) # log pi(at|st)

                # Store G_t and log pi(at|st) for each time step in this trajectory
                store_logits.append(logprob)
                store_at.append(G_t_array)

            optimizer.zero_grad()

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
                # print(temp_sum)
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

            print(loss)
            # print(loss.grad)
            # print(policy.l1.weight)
            print("Episode %d: %f" % (episode +1, np.mean(total_rewards)))
            # print(loss)
            # print("==============================================")
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

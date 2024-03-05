import numpy as np
import random
from numpy import savetxt
import sys
import matplotlib.pyplot as plt

class Sarsa:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

    def select_action(self, state):
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample() # Explore action space
        return np.argmax(self.q_table[state]) # Exploit learned values

    def train(self, filename, plotFile):
        actions_per_episode = []
        for i in range(1, self.episodes+1):
            (state, _) = self.env.reset()
            reward = 0
            done = False
            actions = 0
            rewards = 0
            action = self.select_action(state)
            while not done:
                next_state, reward, done, truncated, _ = self.env.step(action) 
                old_value = self.q_table[state, action] 
                next_action = self.select_action(next_state)

                new_value = old_value + self.alpha*(reward+ self.gamma*self.q_table[next_state, next_action] - old_value) 
                self.q_table[state, action] = new_value
                state = next_state
                actions=actions+1
                rewards+= reward
                action = next_action

            actions_per_episode.append(rewards)
            if i % 100 == 0:
                sys.stdout.write("Episodes: " + str(i) +'\r')
                sys.stdout.flush()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec

        savetxt(filename, self.q_table, delimiter=',')
        if (plotFile is not None): self.better_plotactions(plotFile, actions_per_episode)
        return self.q_table, actions_per_episode

    def plotactions(self, plotFile, actions_per_episode):
        plt.plot(actions_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('# Actions')
        plt.title('# Actions vs Episodes')
        plt.savefig(plotFile+".jpg")     
        plt.close()

    def better_plotactions(self, plotFile, actions_per_episode):
        plt.scatter(range(len(actions_per_episode)),actions_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('# Rewards')
        plt.title('# Rewards vs Episodes')
        plt.savefig(plotFile+".jpg")     
        plt.close()
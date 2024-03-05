import numpy as np
import gym
import random
from numpy import savetxt
import sys
import matplotlib.pyplot as plt
import seaborn as sns

class QLearning:

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

            while not done:
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action) 
        
                # Adjust Q value for current state
                old_value = self.q_table[state, action] #pegar o valor na q-table para a combinacao action e state
                next_max =  np.max(self.q_table[next_state, :])   #np.max(`do maior valor considerando next_state`)
                new_value = old_value + self.alpha*(reward+ self.gamma*next_max - old_value) #calcula o novo valor
                self.q_table[state, action] = new_value
                # atualiza para o novo estado
                state = next_state
                actions=actions+1
                rewards+= reward

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
        plt.legend()
        plt.savefig(plotFile+".jpg")     
        plt.close()
        self.plot_q_table()

    def better_plotactions(self, plotFile, actions_per_episode):
        plt.scatter(range(len(actions_per_episode)),actions_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('# Rewards')
        plt.title('# Rewards vs Episodes')
        plt.savefig(plotFile+".jpg")     
        plt.close()

    def update(self, state, action, next_state, reward, done):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state, action] = new_value

    def plot_q_table_heatmap(self):
        if len(self.q_table_memo) >= 3:
            fig, ax = plt.subplots(ncols=3)
            for i in range(3):
                ax[i] = sns.heatmap(self.q_table_memo[i][0: 8, :], ax=ax[i], cmap="BuPu")
                ax[i].set_xlabel("Actions")
                ax[i].set_ylabel("States")
            ax[0].set_title("Q-Table Initial")
            ax[1].set_title("Q-Table Middle")
            ax[2].set_title("Q-Table Final")
            plt.tight_layout()
            plt.savefig("./results/qtable-heatmap.jpg", dpi=300)
            plt.close()
        else:
            print("Not enough Q-table snapshots for visualization.")
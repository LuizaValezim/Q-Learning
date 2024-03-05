import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from models.Sarsa import Sarsa
from models.QLearning import QLearning
from numpy import loadtxt
import matplotlib.pyplot as plt

env = gym.make("CliffWalking-v0").env

def mean(r):
    out = []
    a = list(np.arange(0,len(r),10))
    for i in range(1,len(a)):
        out.append(np.mean(r[a[i-1]:a[i]]))
    return out

def specific_plot(qlearning, sarsa):
    r1,r2 = mean(qlearning), mean(sarsa)
    plt.plot(range(len(r1)),r1,'b',label="Q-Learning")
    plt.plot(range(len(r2)),r2,'g', label="Sarsa")
    plt.xlabel('Episodes')
    plt.ylabel('# Rewards')
    plt.title("Hyperparams = (alpha=0.4, gamma=0.99, epsilon=0.7)", y=0.99, fontsize=10)
    plt.suptitle('Rewards vs Episodes | Sarsa vs Q Learning | Cliff Walking',fontsize=12)
    plt.legend(loc="best")
    plt.xlim(-5,305)
    plt.savefig("./results/cliff-walking/cliffwalking-comparison.jpg")     
    plt.close()

print("Training with QLearning")
qlearn = QLearning(env, alpha=0.4, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=10000)
q_table, q_rewards = qlearn.train('data/q-table-cliffwalking.csv', './results/cliff-walking/cliffwalking-qlearning')
#q_table = loadtxt('data/q-table-cliffwalking.csv', delimiter=',')

print("Training with Sarsa")
sarsa = Sarsa(env, alpha=0.4, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=10000)
sarsa_table, sarsa_rewards = sarsa.train('data/sarsa-table-cliffwalking.csv', './results/cliff-walking/cliffwalking-sarsa')
#sarsa_table = loadtxt('data/sarsa-table-cliffwalking.csv', delimiter=',')

specific_plot(q_rewards, sarsa_rewards)

env = gym.make("CliffWalking-v0", render_mode="human").env

(state, _) = env.reset()
rewards_q = 0
actions_q = 0
done = False

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, truncated, info = env.step(action)

    rewards_q = rewards_q + reward
    actions_q = actions_q + 1

(state, _) = env.reset()
rewards_sarsa = 0
actions_sarsa = 0
done = False
old_state = 0

while not done:
    old_state = state
    action = np.argmax(sarsa_table[state])
    state, reward, done, truncated, info = env.step(action)

    if old_state == state:
        print("Something seems wrong. Retraining Sarsa...")
        env = gym.make("CliffWalking-v0").env
        sarsa = Sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=10000)
        sarsa_table,sarsa_rewards = sarsa.train('data/sarsa-table-cliffwalking.csv', './results/cliff-walking/cliffwalking-sarsa')

        print("Reseting env")
        env = gym.make("CliffWalking-v0", render_mode="human").env
        (state, _) = env.reset()
        rewards_sarsa = 0
        actions_sarsa = 0

    rewards_sarsa = rewards_sarsa + reward
    actions_sarsa = actions_sarsa + 1
    

print("\n")
print("Actions taken for Q learning: {}".format(actions_q))
print("Rewards for Q Learning: {}".format(rewards_q))
print("\n")
print("Actions taken for Sarsa: {}".format(actions_sarsa))
print("Rewards for Sarsa: {}".format(rewards_sarsa))
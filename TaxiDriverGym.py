import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from models.Sarsa import Sarsa
from models.QLearning import QLearning
from numpy import loadtxt
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

env = gym.make("Taxi-v3", render_mode='ansi').env

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
    plt.suptitle('Rewards vs Episodes | Sarsa vs Q-Learning | Taxi Driver',fontsize=12)
    plt.legend(loc="best")
    plt.xlim(-5,305)
    plt.savefig("./results/taxi-driver/sarsa-taxi-driver.jpg")     
    plt.close()

print("Training with QLearning")
qlearn = QLearning(env, alpha=0.4, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
q_table, q_rewards = qlearn.train('data/q-table-taxi-driver.csv', './results/taxi-driver/taxidriver-qlearning')
#q_table = loadtxt('data/q-table-taxi-driver.csv', delimiter=',')

print("Training with Sarsa")
sarsa = Sarsa(env, alpha=0.4, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
sarsa_table, sarsa_rewards = sarsa.train('data/sarsa-table-taxi-driver.csv', './results/taxi-driver/taxidriver-sarsa')
#sarsa_table = loadtxt('data/sarsa-table-taxi-driver.csv', delimiter=',')

specific_plot(q_rewards, sarsa_rewards)

(state, _) = env.reset()
epochs_q, penalties_q, reward = 0, 0, 0
done = False
frames_q = []

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, truncated, info = env.step(action)

    if reward == -10:
        penalties_q += 1

    frames_q.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )
    epochs_q += 1

(state, _) = env.reset()
epochs_sarsa, penalties_sarsa, reward = 0, 0, 0
done = False
frames_sarsa = []
 
while not done:
    print("state: "+ str(state))
    action = np.argmax(sarsa_table[state])
    state, reward, done, truncated, info = env.step(action)

    if reward == -10:
        penalties_q += 1

    if epochs_sarsa > 100:
        (state, _) = env.reset()
        epochs_sarsa, penalties_sarsa, reward = 0, 0, 0
        frames_sarsa = []

    frames_sarsa.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )
    epochs_sarsa += 1

clear_output(wait=True)

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames_q)
print_frames(frames_sarsa)

print("\n")
print("Timesteps taken for Q Learning: {}".format(epochs_q))
print("Penalties incurred for Q Learning: {}".format(penalties_q))

print("\n")
print("Timesteps taken for Sarsa: {}".format(epochs_sarsa))
print("Penalties incurred for Sarsa: {}".format(penalties_sarsa))
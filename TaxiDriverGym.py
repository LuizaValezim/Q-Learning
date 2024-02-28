import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from QLearning import QLearning
from numpy import loadtxt
import matplotlib.pyplot as plt


env = gym.make("Taxi-v3", render_mode='ansi').env

# apenas execute as próximas linhas se você deseja treinar o agente novamente

alpha_values = [0.5, 0.3, 0.7]
gamma_values = [0.9, 0.95, 0.05]
epsilon_values = [0.7, 0.8, 0.9]

actions_per_episode = []

for a, g, e in alpha_values, gamma_values:
    print(a,g,e)
    qlearn = QLearning(env, alpha=a, gamma=g, epsilon=e, epsilon_min=0.05, epsilon_dec=0.99, episodes=50000)
    q_table, action_per_episode = qlearn.train('data/q-table-taxi-driver.csv', 'results/actions_taxidriver')
    actions_per_episode.append(action_per_episode)
    
qlearn.plotactions("./results/plotFile", action_per_episode)
#q_table = loadtxt('data/q-table-taxi-driver.csv', delimiter=',')

#
# Depois de treinado, podemos executar novamente o agente.
#

(state, _) = env.reset()
epochs, penalties, reward = 0, 0, 0
done = False
frames = [] # for animation
    
while (not done) and (epochs < 100):
    action = np.argmax(q_table[state])
    state, reward, done, t, info = env.step(action)

    if reward == -10:
        penalties += 1

    frames.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )
    epochs += 1

from IPython.display import clear_output
from time import sleep

clear_output(wait=True)

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        #print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)
print("\n")
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))
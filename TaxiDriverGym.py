import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from QLearning import QLearning
from numpy import loadtxt
import matplotlib.pyplot as plt
import seaborn as sns

env = gym.make("Taxi-v3", render_mode='ansi').env

def mean(action):
    out = []
    a = list(np.arange(0,len(action),10))
    for i in range(1,len(a)):
        out.append(np.mean(action[a[i-1]:a[i]]))
    return out

a, g, e = 0.1, 0.1, 0.7
qlearn = QLearning(env, alpha=a, gamma=g, epsilon=e, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
q_table, action = qlearn.train('data/q-table-taxi-driver.csv', 'results/actions_taxidriver2')
action = mean(action)
plt.plot(range(len(action)), action, label="alpha = " + str(a) + " : gamma= " + str(g) + " : epsilon=" + str(e))

a, g, e = 0.5, 0.5, 0.7
qlearn = QLearning(env, alpha=a, gamma=g, epsilon=e, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
q_table, action = qlearn.train('data/q-table-taxi-driver.csv', 'results/actions_taxidriver2')
action = mean(action)
plt.plot(range(len(action)), action, label="alpha = " + str(a) + " : gamma= " + str(g) + " : epsilon=" + str(e))

a, g, e = 0.9, 0.9, 0.99
qlearn = QLearning(env, alpha=a, gamma=g, epsilon=e, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
q_table, action = qlearn.train('data/q-table-taxi-driver.csv', 'results/actions_taxidriver2')
action = mean(action)
plt.plot(range(len(action)), action, label="alpha = " + str(a) + " : gamma= " + str(g) + " : epsilon=" + str(e))

plt.xlim(0, 100)
plt.xlabel('Episodes')
plt.ylabel('# Rewards')
plt.title('# Rewards vs Episodes')
plt.legend(loc="best")
plt.savefig("./results/actions_taxidriver"+".jpg")     
plt.close()


'''
estados = [0, 2500, 4999]  # Estados inicial, intermediário e final

# Plotagem dos heatmaps para os estados escolhidos
for i, estado in enumerate(estados):
    plt.imshow(q_table, cmap='viridis', interpolation='nearest', aspect='auto')
    plt.title(f'Q-Table - Estado {i+1}')
    plt.xlabel('Ações')
    plt.ylabel('Estados')
    plt.colorbar()
    plt.savefig(f'./results/q_table_estado_{i+1}.jpg')
    plt.close()


# Choose specific states to visualize the evolution of the Q-table
initial_state = 0  # Choose an initial state
intermediate_state = 50  # Choose an intermediate state
final_state = 99  # Choose a final state

# Extract the Q-tables corresponding to the chosen states
q_tables = [q_table1, q_table2, q_table3]
chosen_states = [initial_state, intermediate_state, final_state]

# Plot the heatmaps of the chosen Q-tables
plt.figure(figsize=(15, 5))
for i, (q_table, state) in enumerate(zip(q_tables, chosen_states), 1):
    plt.subplot(1, 3, i)
    sns.heatmap(q_table[state].reshape((1, -1)), cmap='hot', annot=True, cbar=False)
    plt.title(f"State {state}")
plt.tight_layout()
plt.savefig("./results/q_table_evolution_heatmaps.jpg")
plt.close()
'''
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
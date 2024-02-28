import numpy as np
import random
from numpy import savetxt
import sys
import matplotlib.pyplot as plt

#
# Esta classe implementa o algoritmo Q-Learning.
# Você pode usar esta implementação para criar agentes para atuar em alguns ambientes do projeto Gymansyium.
#

class QLearning:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        '''
        __init__(): Recebe todos os hiperparâmetros do algoritmo Q-Learning e inicializa a Q-table com base no número 
        de ações e estados informados pelo parâmetro env. Alguns destes hiperparâmetros não serão utilizados neste momento, 
        mas vamos mantê-los.
        '''
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

    def select_action(self, state):
        ''' 
        select_action(): Dado um estado, este método seleciona uma ação que deve ser executada.
        '''
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample() # Explora o espaço de ações
        return np.argmax(self.q_table[state]) # Faz uso da tabela Q

    def train(self, filename, plotFile):
        '''
        train(): método responsável por executar as simulações e popular a Q-table. Este método retorna uma q-table, 
        mas também atualiza um arquivo CSV com os dados da q-table e uma imagem que é um plot da quantidade de 
        ações executadas em cada episódio.
        '''
        actions_per_episode = []
        for i in range(1, self.episodes+1):
            (state, _) = self.env.reset()
            rewards = 0
            done = False
            actions = 0

            while not done:
                action = self.select_action(state)  # escolher uma ação para o estado atual
                next_state, reward, done, _ = self.env.step(action)[:4]

                # Atualizar a Q-table usando a equação do algoritmo Q-Learning
                self.q_table[state, action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])

                state = next_state  # atualizar o estado atual para o próximo estado

                rewards += reward  # acumular as recompensas recebidas durante o episódio


            actions_per_episode.append(actions)
            if i % 100 == 0:
                sys.stdout.write("Episodes: " + str(i) +'\r')
                sys.stdout.flush()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec

        savetxt(filename, self.q_table, delimiter=',')
        if (plotFile is not None): self.plotactions(plotFile, actions_per_episode)
        return self.q_table

    def plotactions(self, plotFile, actions_per_episode):
        '''
        plotactions(): É um método que cria uma imagem com o plot da quantidade de ações executadas em cada episódio.
        '''
        plt.plot(actions_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('# Actions')
        plt.title('# Actions vs Episodes')
        plt.savefig(plotFile+".jpg")     
        plt.close()
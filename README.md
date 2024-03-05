# Q-Learning vs Sarsa

## Questions

**1. Qual algoritmo tem os melhores resultados para o ambiente do taxi-driver? A curva de aprendizado dos dois algoritmos é a mesma? O comportamento final do agente, depois de treinado, é ótimo?**

Para o treinamento do Taxi Driver o Sarsa conseguiu realizá-lo em menos ações que o Q-Learning na maioria das vezes. O que percebi é que após rodar algumas vezes o valor ia variando. Vezes que o Q-Learning performava melhor, vezes pior. 

![taxidriver-comparison](https://github.com/insper-classroom/05-q-learning-sarsa-LuizaValezim/assets/62949391/66e529ba-7913-4f60-8529-67bfe6860804)

Mesmo assim, podemos concluir que ambos obtiveram um resultado satisfatório e ótimo.

---

**2. Qual algoritmo tem os melhores resultados para o ambiente do Cliff Walking? A curva de aprendizado dos dois algoritmos é a mesma? O comportamento final do agente, depois de treinado, é ótimo? Qual agente tem um comportamento mais conservador e qual tem um comportamento mais otimista?**

Com base nos resultados que obtive, pude ver que ambos os algoritmos tiveram um comportamento ótimo ao final, porém, com os hiperparâmetros setados iguais, o algoritmo de Q-Learning demorou um pouco para se estabilizar em comparação com o do Sarsa. 

![cliffwalking-comparison](https://github.com/insper-classroom/05-q-learning-sarsa-LuizaValezim/assets/62949391/5808a07e-381e-4650-9d64-b8c42dedbb6a)

Para o treinamento do Cliff Walking o algoritmo Q-Learning mostrou-se mais eficiente para chegar em seu objetivo em menos ações e maior potuação em comparação com o do Sarsa.Isso mesmo após algumas vezes rodando o treinamento.

Em termos de comportamento, o Sarsa tende a ser mais conservador, pois leva em consideração a política atual ao escolher a ação seguinte, o que pode resultar em um comportamento mais cauteloso para evitar riscos. Por outro lado, o Q-Learning pode ser mais otimista, pois atualiza os valores Q sem considerar a política atual, o que pode levar a uma exploração mais agressiva do ambiente em busca de recompensas maiores.

---

**3. Suponha uma seleção de ação gulosa (greedy), qual seria a diferença entre os algoritmos Q-Learning e Sarsa? Os agentes treinados teriam o mesmo comportamento? As curvas de aprendizado também?**

Na seleção de ação gulosa, Q-Learning e Sarsa diferem na forma como atualizam os valores Q e escolhem a próxima ação. 

Enquanto o Q-Learning atualiza os valores Q considerando a ação que maximiza o valor Q para o próximo estado, independentemente da ação que será realmente tomada, o Sarsa atualiza os valores Q considerando a próxima ação que será tomada de acordo com a política atual do agente. 

Como resultado, os agentes treinados com Q-Learning tendem a ser mais agressivos na exploração, enquanto os treinados com Sarsa tendem a ser mais conservadores. Essas diferenças podem levar a curvas de aprendizado distintas, com o Sarsa possivelmente apresentando uma curva mais suave e estável, enquanto o Q-Learning pode ser mais volátil durante a fase de exploração inicial.

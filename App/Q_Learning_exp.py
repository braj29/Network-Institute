# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 19:27:14 2020

@author: Hang Yu
"""
#import wandb
#import schedualing as Env
#import Stacking as Env
import Q_Learning as QL
import Policy_Shaping as PS
import matplotlib.pyplot as plt
import pickle
import numpy as np
import gym
import griddly
from griddly import gd

q_learning_table_path = 'q_learning_oracle_3.pkl' 
env = gym.make('GDY-Labyrinth-v0', player_observer_type = gd.ObserverType.VECTOR, global_observer_type = gd.ObserverType.SPRITE_2D)
#env = Env.Stacking()
#env = Env.Fixing()

state = env.reset()
episodes = 100000
times = 1
total_reward = [0 for i in range(episodes)]
cnt = 0
#wandb.init(project='QL-Baseline')

#wandb.log({'total_reward': total_reward})

 

for t in range(times):
    print(t)
    Qagent = QL.QLAgent(env.action_space, env.observation_space, epsilon = 0.2, mini_epsilon = 0.01, decay = 0.999)
    #Pagent = PS.PSAgent(env.action_space)
    for epsd in range(episodes):
        #total_reward.append(0)
        #print(epsd)

        if epsd % 1000 == 0:
            print(epsd)
        start_f = cnt
        while(1):
            cnt += 1
            #env.render(observer = "global")
            action = Qagent.choose_action(state)

            next_state, reward, is_done, info = env.step(action)
            Qagent.learning(action,reward,state,next_state)
            state = next_state
            total_reward[epsd] += reward

            if is_done or cnt - start_f > 1000:
                #print(epsd, state)
                state = env.reset()
                break
        #wandb.log({'reward': total_reward[epsd], 'episode': epsd})
with open(q_learning_table_path, 'wb') as pkl_file:
            pickle.dump(Qagent, pkl_file)


x=[i+1 for i in range(episodes)]
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot(x,np.array(total_reward)/times,color='green',label='test')
plt.savefig("Q_Learning_Oracle_3")

res= np.array(total_reward)/times
f = open('None_feedback', 'w')  
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close()
# import numpy as np
# value_map = np.zeros((16, 14))
# for x in range(1, 17):
#     for y in range(1,15):
#         q_values_of_state = QL.QLAgent.qtable[(x, y)]
#         maxValue = max(q_values_of_state.values())
#         value_map[x, y] = maxValue
#
# wandb.log({'Value Function': wandb.plots.HeatMap(list(range(1, 17)),
#                                          list(range(1,15)), value_map)})
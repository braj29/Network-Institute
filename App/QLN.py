# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 03:25:58 2020

@author: Hang Yu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 19:27:14 2020

@author: Hang Yu
"""

#import schedualing as Env
#import Stacking as Env
import Q_Learning as QL
import Policy_Shaping as PS
#import Estimation as Est

import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
import gym
import griddly
from griddly import gd

q_learning_table_path = 'q_learning_oracle_2.pkl' 
env = gym.make('GDY-Labyrinth-v0', player_observer_type = gd.ObserverType.VECTOR, global_observer_type = gd.ObserverType.SPRITE_2D)
#env = Env.Stacking()
# episodes = 1000
# times = 5
#env = Env.Fixing()
epsilon_max = 1
epsilon_min = 0.1
eps_decay = 3000

weight_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
                -1. * frame_idx / eps_decay)

def test(model = 1, episodes= 1000, times  = 1):
    
    state = env.reset()
    total_reward = [0 for i in range(episodes)]
    cnt = 0
    for t in range(1):
        #print(t)
        Qagent = QL.QLAgent(env.action_space, env.observation_space, epsilon = 0.2, mini_epsilon = 0.01, decay = 0.999)
        #Pagent = PS.PSAgent(env.action_space)
        #Eagent = Est.PSAgent(env.action_space)
        
        sof = 100 * model
        demo_time = 1 * model
        
        # for i in range(demo_time):
        #     state = env.reset()
        #     while(1 and sof > 0):
        #         sof -= 1
        #         action = env.perfect_action(state)
        #         next_state, reward, is_done = env.step(action)
        #         Qagent.learning(action,reward,state,next_state)
        #         Pagent.learning(action, 1, state, next_state)
        #         state = next_state
        #         if is_done:
        #             break
        
        for epsd in range(episodes):
            #total_reward.append(0)
            #print(epsd)
            state = env.reset()
            start_f = cnt
            while(1):
                cnt += 1
                feedback = 0
                weight = weight_by_frame(cnt)
                prob = Qagent.action_prob(state) #+ np.asarray(Pagent.action_prob(state)) * weight
                
                
                # if cnt % 6 ==0 and sof > 0:
                #     sof -= 1
                #     action = env.perfect_action(state)
                #     feedback = 1
                #advice
                
                #action = np.random.choice(np.flatnonzero(prob == prob.max()))
                action = np.random.choice([i for i in range(env.action_space.n)],p = prob/sum(prob))
    
                        
    
                next_state, reward, is_done, info = env.step(action)
                total_reward[epsd] += reward 
                
                
                
                # reward += Eagent.estimation(next_state) - Eagent.estimation(state)
                # if cnt % 10 == 0 and sof > 0:
                #     sof -= 1
                #     Eagent.learning(0, env.estimation(state),state,next_state)
                #     Eagent.learning(0, env.estimation(next_state),next_state,next_state)
                # #Estimation
            
                            
                # if cnt % 5 ==0 and sof > 0:
                #     sof -= 1
                #     if  action == env.perfect_action(state):
                #         feedback = 1
                #     else:
                #         feedback = -1
                # #evaluation 
                
                Qagent.learning(action,reward,state,next_state)
    
                # if feedback !=0 and sof > 0:
                #     #feedback_num -= 1
                #     #print(action, feedback,state,next_state)
                #     Pagent.learning(action, feedback,state,next_state)
                # # F learning
                
                
                state = next_state
                
                if is_done or cnt - start_f > 100:
                    #print(epsd, state)
                    state = env.reset()
                    break
        with open(q_learning_table_path, 'wb') as pkl_file:
            pickle.dump(Qagent, pkl_file)
    return total_reward
        
        
length = 100000
times = 10
M=[0 for i in range(length)]
S=[0 for i in range(length)]
N=[0 for i in range(length)]
R=[0 for i in range(length)]

for i in range(times):
    print('\n',i,"-th Trial")
    M = np.sum([test(1, length, times),M], axis=0)
    #S = np.sum([test(5, length, times),S], axis=0)
    #N = np.sum([test(10, length, times),N], axis=0)
    #R = np.sum([test(100, length, times),R], axis=0)
    





x=[i+1 for i in range(length)]
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot(x,M/times,color='green',label='M')
# plt.plot(x,S/times,color='red',label='S')
# plt.plot(x,N/times,color='yellow',label='N')
# plt.plot(x,R/times,color='pink',label='R')

plt.legend()
plt.savefig("Q_Learning_3")


res=M/times
f = open('Q_Learning.txt', 'w')  
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close() 


# res=S/times
# f = open('rf_5.txt', 'w')    
# for r in res:  
#     f.write(str(r))  
#     f.write('\n')  
# f.close() 

# res=N/times
# f = open('rf_10.txt', 'w')    
# for r in res:  
#     f.write(str(r))  
#     f.write('\n')  
# f.close() 

# res=R/times
# f = open('rf_100.txt', 'w')    
# for r in res:  
#     f.write(str(r))  
#     f.write('\n')  
# f.close() 
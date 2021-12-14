# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:26:12 2020

@author: Hang Yu
"""
'''
things need to be done:
    1. learning experience while there is a feedback
    2. an auto coder that reduces img 
    3. loss func


'''
import DQN
import DFN
import DEN
import gym, random, pickle, os.path, math, glob, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import pdb

from atari_wrappers import make_atari, wrap_deepmind,LazyFrames
from IPython.display import clear_output
#from tensorboardX import SummaryWriter

def decide(f,q):
    prob=f*q/sum(f*q)
    return np.random.choice([i for i in range(len(prob))], p = prob)
def softmax(prob):
    #p = np.random.rand()
    #print(prob)
    return np.random.choice([i for i in range(len(prob))], p = prob)
def argmax(prob):
    return prob.argmax()
def shape(prob):
    prob[0] += prob[1]
    prob[1] =prob[0]
    prob[2] += prob[4]
    prob[4] =prob[2]
    prob[3] += prob[5]
    prob[5] =prob[3]
    return prob
def policy_shaping_prob(p):
    b =0.9
    p = pow(b,p)/(pow(b,p)+pow(1-b,p))
    return p

# if __name__ == '__main__':
    
# if __name__ == '__main__':
    
# Training DQN in PongNoFrameskip-v4 
env = make_atari('PongNoFrameskip-v4')
env = wrap_deepmind(env, scale = False, frame_stack=True)

gamma = 0.99
epsilon_max = 1
epsilon_min = 0.01
eps_decay = 30000
#frames = 1000000
episodes=500
USE_CUDA = False

learning_rate = 0.001
max_buff = 30000
update_tar_interval = 1000
batch_size = 32
print_interval = 1000
log_interval = 1000
learning_start = 1000
feedback_learning_start = 5000
win_reward = 18     # Pong-v4
win_break = True
#num_of_feedback=10000000
feedback_update_interval= 1000



action_space = env.action_space
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]

state_channel = env.observation_space.shape[2]
#feedback_agent = DFN.DFNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = 1)
#feedback_agent = DFNB.DFNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = 10*learning_rate)

oracle=DQN.DQNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = learning_rate)
oracle.DQN.load_state_dict(torch.load('PongGame.pth', map_location=torch.device('cpu')))

def DPS(episodes,model = 0):
    
    q_agent=DQN.DQNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = learning_rate)
    qe_agent=DQN.DQNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = learning_rate)
    f_agent = DFN.DQNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = learning_rate)
    e_agent = DEN.DQNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = learning_rate)
    episode_rewards = []
    all_rewards = []
    sum_rewards=[]
    losses = []
    episode_num = 0
    is_win = False
    sof = 10000 * model
    epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
                -1. * frame_idx / eps_decay)
    # plt.plot([epsilon_by_frame(i) for i in range(10000)])
    start_f=0
    end_f=0
    prt=False
    

            
            
    frame = env.reset()
    for epsd in range(episodes):
        start_f=end_f
        cnt=0
        all_cnt=0
        while(True):
            #env.render()
            end_f+=1
            
            epsilon = epsilon_by_frame(end_f)
            state_tensor = f_agent.observe(frame) 
            prob = q_agent.act(state_tensor, epsilon) #+ f_agent.act(state_tensor, epsilon) + qe_agent.act(state_tensor, epsilon)
            prob = prob/sum(prob)
            if np.isnan(prob).any():
                prob = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
            action = np.random.choice([0,1,2,3,4,5],p = prob)
            #action = np.random.choice(np.flatnonzero(prob == prob.max()))
            #choose action
            
            feedback = 0
     
                
        
  

            next_frame, reward, done, _ = env.step(action)
            episode_rewards.append( reward )
            next_state_tensor = q_agent.observe(next_frame) 
            loss = 0
            fb_loss=0
            #est = e_agent.estimation(next_state_tensor) - e_agent.estimation(state_tensor)
            q_agent.memory_buffer.push(frame, action, reward, next_frame, done)
            if cnt % 5 == 0 and sof > 0:
                sof -= 1
                if action == np.random.choice([0,1,2,3,4,5], p = oracle.act(state_tensor , 0)):
                    feedback += 0.1
                else:
                    feedback  -= 0.1
                
                q_agent.memory_buffer.push(frame, action, feedback + reward, next_frame, done)   
            #evaluation
            if reward != 0:
                q_agent.memory_buffer.push(frame, action,  reward , next_frame, done)   
                
            #deepq_agent.memory_buffer.push(frame, action, reward, next_frame, done)
            
             

            
            
            if q_agent.memory_buffer.size() >= learning_start:
                loss = q_agent.learn_from_experience(batch_size)
                losses.append(loss)
            if end_f % log_interval == 0:
                q_agent.DQN_target.load_state_dict(q_agent.DQN.state_dict())
            # Q learning
            # if e_agent.memory_buffer.size() >= learning_start:
            #     loss = e_agent.learn_from_experience(batch_size)
            #     losses.append(loss)
            # if end_f % log_interval == 0:
            #     e_agent.DQN_target.load_state_dict(e_agent.DQN.state_dict())
            # # Q learning
            
            
 
            
            frame = next_frame
            if done:
                if epsd  % 1 ==0 :
                    print("frames: %.5d, reward: %.5f,  loss: %.4f, acc: %.4f, epsilon: %.5f, episode: %.4d" % (end_f-start_f, 
                                                                                                    np.sum(episode_rewards),
                                                                                                    loss, epsd, epsilon,
                                                                                                    episode_num))
                all_rewards.append(episode_rewards)
                sum_rewards.append(np.sum(episode_rewards))
                
                episode_num += 1
                avg_reward = float(np.mean(sum_rewards[-10:]))/10

                #
                #print('\rEpisode {}\tframe: {:d}\tScore: {:.2f}\tavg_Score: {:.2f}\tepsilon: {:.4f}'.format( epsd,end_f-start_f, np.sum(episode_rewards),avg_reward,epsilon), end="")

                episode_rewards = []
                frame = env.reset()
                #torch.save(agent,'Oracle.pkl')
                break
    env.close()
    return sum_rewards 


length = 200
times = 1
M=[0 for i in range(length)]
S=[0 for i in range(length)]
N=[0 for i in range(length)]
R=[0 for i in range(length)]

for i in range(times):
    print('\n',i,"-th Trial")
    M = np.sum([DPS(length,model = 1),M], axis=0)
    S = np.sum([DPS(length,model = 2),S], axis=0)
    N = np.sum([DPS(length,model = 5),N], axis=0)
    R = np.sum([DPS(length,model = 10),R], axis=0)


x=[i+1 for i in range(length)]
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot(x,M/times,color='green',label='M')
plt.plot(x,S/times,color='red',label='S')
plt.plot(x,N/times,color='yellow',label='N')
plt.plot(x,R/times,color='pink',label='R')

plt.legend()


res=M/times
f = open('DQN+TAMER_1.txt', 'w')  
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close() 


res=S/times
f = open('DQN+TAMER_5.txt', 'w')   
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close() 

res=N/times
f = open('DQN+TAMER_10.txt', 'w')   
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close() 

res=R/times
f = open('DQN+TAMER_100.txt', 'w')   
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close() 
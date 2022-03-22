# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:47:24 2020

@author: Hang Yu
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import math

def get_state(state):
    
    if len(state) != 2:
        s = np.array(np.where(state[0] == 1)).T.flatten()
        if s != []:
            x = s[0]
            y = s[1]
            return (str(x),str(y))
        
        else:
            return (str(1), str(1)) 
         
    else:
        return state


class QLAgent:
    def __init__(self, action_space, state_space, alpha = 0.5, gamma=0.8, temp = 1, epsilon = 0.95, mini_epsilon = 0.01, decay = 0.999):
        self.action_space = 5# action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.decay = decay
        #print(self.action_space)
        #self.qtable=pd.DataFrame(columns=[ i for i in range(self.action_space.n)], dtype=object)
        d, w, h = 4, 16, 14
        total_states = w * h
        #shp = state_space.shape
        xs = list(range(1, w+1))
        ys = list(range(1, h+1))
        coordinates = [(str(x), str(y)) for x in xs for y in ys]
        index = pd.MultiIndex.from_tuples(coordinates, names=["X", "Y"])
        self.qtable = pd.DataFrame(index = index, columns = range(1, self.action_space), dtype= object)
        #print(self.qtable)
        self.qtable = self.qtable.fillna(1)
        #self.qtable = np.zeros(total_states, action_space.n)
        # for x in range(environment.height): # Loop through all possible grid spaces, create sub-dictionary for each
        #     for y in range(environment.width):
        #         self.q_table[(x,y)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0}
        
    # def trans(self, state):
    #     s = ""
    #     for i in range(len(state)):
    #         s+=str(state[i])
    #     return s
    # def check_add(self,state):
    #     if self.trans(state) not in self.qtable.index:
    #         self.qtable.loc[state]=pd.Series(np.zeros(self.action_space),index=[ i for i in range(self.action_space)])
            
    def learning(self, action, rwd, state, next_state):
        #self.check_add(state)
        #self.check_add(next_state)
        state = get_state(state)
        next_state = get_state(next_state)
        q_sa = self.qtable.loc[state, action]
        max_next_q_sa = self.qtable.loc[next_state, :].max()
        new_q_sa = q_sa + self.alpha * (rwd + self.gamma * max_next_q_sa - q_sa)
        self.qtable.loc[state, action] = new_q_sa
        #print(self.qtable.loc[state,:])
    # def learning(self, action, feedback, state, next_state):
    #     self.check_add(state)
    #     self.check_add(next_state)
    #     #print(math.exp(self.feedback.loc[self.trans(state),action]))
    #     self.qtable.loc[self.trans(state),action] +=  feedback
                                                     
    # def learning(self, action, rwd, state, next_state):
    #     self.check_add(state)
    #     self.check_add(next_state)
    #     q_sa= self.qtable.loc[self.trans(state),action]
    #     max_next_q_sa=self.qtable.loc[self.trans(next_state),:].max()
    #     new_q_sa= q_sa + self.alpha *( rwd + self.gamma * max_next_q_sa - q_sa) 
    #     self.qtable.loc[self.trans(state),action]= new_q_sa
    
    
            
    def action_prob(self, state):
        #self.check_add(state)
        state = get_state(state)
        p = np.random.uniform(0,1)
        #self.epsilon = 0.95
        #p=100
        if p <= self.epsilon:
            return np.array([1/self.action_space for i in range(1, self.action_space)])
        else:
            prob = F.softmax(torch.tensor(self.qtable.loc[(state), :].to_list()).float(),dim = 0).detach().numpy()
            return prob
    def choose_action(self, state):
        #self.check_add(state)
        #print(state)
        state = get_state(state)
        p = np.random.uniform(0,1)
        
        if self.epsilon >= self.mini_epsilon:
            self.epsilon *= self.decay
        if p <= self.epsilon:
            return np.random.choice([i for i in range(self.action_space)])
        else:
            #print(self.qtable)
            prob = F.softmax(torch.tensor(self.qtable.loc[(state), :].to_list()).float(), dim = 0).detach().numpy()
            #action = np.argmax(self.qtable.loc[(state), :])

            return np.random.choice(np.flatnonzero(prob == prob.max()))



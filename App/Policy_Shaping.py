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
import wandb
# m = np.zeros((16,14))
# #m[0][13] = 1
# w = 16
# h = 14
# #print(num_list)
# n = 1
# for i in range(w):
#     for j in range(h):
#         m[i][j] = n
#         n += 1

# def get_state (state):
#     if state[0].shape == (16,14):
#         s = np.array(np.where(state[0] == 1)).T.flatten()
#         #print(s)
#         if s != []:
#             x = s[0]
#             y = s[1]
#             cell = m[x][y]
#         else:
#             cell = 1
#         return int(cell)
#     else:
#         return int(state)


def get_state(state):
    
    if len(state) != 2:
        s = np.array(np.where(state[0] == 1)).T.flatten()
        if s != []:
            x = s[0]
            y = s[1]
            if (x,y) != (0,0):
                
                return (str(x),str(y))
            else:
                return (str(1), str(1))
        
        else:
            return (str(1), str(1)) 
         
    else:
        return state


class PSAgent:
    def __init__(self, action_space, state_space, alpha = 0.5, gamma=0.8, temp = 1, epsilon = 1):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma= gamma
        self.temp = temp
        self.epsilon = epsilon
        #self.feedback=pd.DataFrame(columns=[ i for i in range(self.action_space.n)], dtype=object)
        d, w, h = state_space
        total_states = w * h
        #shp = state_space.shape
        xs = list(range(0,w+1))
        ys = list(range(0, h+1))
        self.coordinates = [(str(x),str(y)) for x in xs for y in ys]
        index = pd.MultiIndex.from_tuples(self.coordinates, names=["X", "Y"])
        self.feedback = pd.DataFrame(index = index, columns= range(1, self.action_space), dtype= object)
        #print(self.qtable)
        self.feedback = self.feedback.fillna(1)
        #wandb.init(name = "test5")
        #print(self.feedback.index)
        #pass
        #self.feedback = np.ones((self.num_actions, self.iht_size))
    
    def trans(self, state):
        s = ""
        for i in range(len(state)):
            s+=str(state[i])
        return s
    def check_add(self,state):
        if self.trans(state) not in self.feedback.index:
            self.feedback.loc[self.trans(state)]=pd.Series(np.zeros(self.action_space),index=[ i for i in range(self.action_space)])
            
    def learning(self, action, feedback, state, next_state):
        #self.check_add(state)
        #self.check_add(next_state)
        #print(math.exp(self.feedback.loc[self.trans(state),action]))
        # self.feedback.loc[self.trans(state),action] = np.tanh(np.arctanh(self.feedback.loc[self.trans(state),action]) 
        #                       + feedback )
        state = get_state(state)
        self.feedback.loc[state,action] += feedback
        #print(self.feedback.loc[state, :])
        #matrx = self.feedback.sum(axis = 1)
        #matrx = matrx.to_numpy()

        #matrx = matrx.reshape((14,16))
        #print(matrx)
        #print(matrx.shape)
        #w,h = matrx.shape
        #wandb.log({'heatmap': wandb.plots.HeatMap(list(range(1,17)), list(range(1,15)), matrx, show_text = False)})

    def action_prob(self, state):
        #self.check_add(state)
        prob = []
        
        state = get_state(state)
        if all(self.feedback.loc[state,:].to_numpy() == 0):
            return np.array([1/self.action_space for i in range(1, self.action_space)])
        for i in range(1, self.action_space):
            if self.feedback.loc[state, i] < -50:
                self.feedback.loc[state, i] = -50
            prob.append(math.pow(0.95,self.feedback.loc[state,i])/
                        (math.pow(0.95,self.feedback.loc[state,i]) +
                         math.pow(0.05,self.feedback.loc[state,i])) )
        return prob
    def choose_action(self, state):
        #self.check_add(state)
        prob = []
        state = get_state(state)
        if all(self.feedback.loc[state].to_numpy() == 0):
            return np.random.choice([i for i in range(self.action_space)])
        for i in range(self.action_space):
            prob.append(math.pow(0.95,self.feedback.loc[state,i])/
                        (math.pow(0.95,self.feedback.loc[state,i]) + 
                         math.pow(0.05,self.feedback.loc[state,i])) )
        prob = np.array(prob)
        return np.random.choice(np.flatnonzero(prob == prob.max()))

if __name__ == "__main__":
    pass
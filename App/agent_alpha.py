'''
This demo of a TAMER algorithm implmented with HIPPO Gym has been adapted
from code provided by Calarina Muslimani of the Intelligent Robot Learning Laboratory
To use this code with the default setup simply rename this file to agent.py
'''

import gym
import time
import numpy as np
import itertools
import Q_Learning as QL
#from gym import wrappers

#import highway_env
#from pettingzoo.mpe import simple_v2

import griddly
from griddly import gd
#from Policy_Shaping import PS

#global state

#from gym.utils import play
#import wandb



#This is the code for tile coding features
basehash = hash

'''
This is a demo file to be replaced by the researcher as required.
This file is imported by trial.py and trial.py will call:
start()
step()
render()
reset()
close()
These functions are mandatory. This file contains minimum working versions
of these functions, adapt as required for individual research goals.
'''
import Policy_Shaping as PS
from Policy_Shaping_copy import get_state



class PSA:
    
    def __init__(self, env):
        self.first_state = None
        self.Agent = PS.PSAgent(env.action_space, env.observation_space)
        self.time_step = 0

    def update_feedback(self, reward):
        # feedback = 0
        if reward == "good":
            return 1
        elif reward == "bad":
            return -1
        elif reward == "None":
            return 0

        #print(reward, feedback)

epsilon_max = 1
epsilon_min = 0.1
eps_decay = 3000
import math
weight_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
                -1. * frame_idx / eps_decay)
        
d, w, h = 4, 16, 14
total_states = w * h
#shp = state_space.shape
# xs = list(range(1,w+1))
# ys = list(range(1, h+1))
# #coordinates = [(str(x),str(y)) for x in xs for y in ys]
# coordinates = list(range(1,225))
# actionList = list(range(5))

class Agent():
    '''
    Use this class as a convenient place to store agent state.
    '''
     
    def start(self, game:str):
        '''
        Starts an OpenAI gym environment.
        Caller:
            - Trial.start()
        Inputs:
            -   game (Type: str corresponding to allowable gym environments)
        Returs:
            - env (Type: OpenAI gym Environment as returned by gym.make())
            Mandatory
        '''
        #game = 'GDY-Labyrinth-v0'
        self.demo = False
        self.env = gym.make(game, player_observer_type = gd.ObserverType.VECTOR)
        #self.env.reset()
        
        
        self.PS = True
        if self.PS:
            np.random.seed(0)
            self.PolSh = PSA(self.env)
            self.Qagent = QL.QLAgent(self.env.action_space, self.env.observation_space, epsilon=0.2, mini_epsilon=0.01,
                                     decay=0.999)

        return

    def step(self, human_action, human_feedback):
        '''
        Takes a game step.
        Caller:
            - Trial.take_step()
        Inputs:
            - env (Type: OpenAI gym Environment)
            - action (Type: int corresponding to action in env.action_space)
        Returns:
            - envState (Type: dict containing all information to be recorded for future use)
              change contents of dict as desired, but return must be type dict.
        '''
        #print(human_feedback)
        #time.sleep(0.1)
        if self.demo == False:
            if self.PolSh.time_step == 0:
                self.state =self.PolSh.first_state
                time.sleep(1.5)
                
            self.PolSh.time_step += 1
            #Q_prob = self.Qagent.action_prob(state)
            #P_prob = self.PolSh.Agent.action_prob(self.state)
            self.cnt += 1
            weight = weight_by_frame(self.cnt)
            #print(weight)
            
            prob = self.Qagent.action_prob(self.state) + weight * np.asarray(self.PolSh.Agent.action_prob(self.state))
            #print(prob, sum(prob)) 
            
            prob = np.asarray(prob)
            sum = np.sum(prob)
            prob = prob/sum
            #prob = prob/5
            print(prob)
            #print(prob==prob.max())
            #action = np.random.choice(np.flatnonzero(prob == prob.max()))
            #prob_d = 
            a = [1,2,3,4]
            action = np.random.choice(a,p = prob)
            #stochastic probability distribution
            feedback = self.PolSh.update_feedback(human_feedback)

            next_state, reward, done, info = self.env.step(action)
            self.next_state = next_state
            #state - next_state =
            # if reward == 0:
            #     reward = -1
            #manhattan
            # goal_state = ("14", "2")
            # current_state = get_state(self.state)
            # manhtn_dist = abs(int(goal_state[0]) - int(current_state[0])) + abs(int(goal_state[1]) - int(current_state[1]))

            reward_f = feedback
            if action != 0:
                self.PolSh.Agent.learning(action, feedback, self.state, self.next_state)
                self.Qagent.learning(action, reward, self.state, self.next_state)
            self.state = next_state

        elif self.demo == True:
            feedback_demo = 1
            next_state, reward, done, info = self.env.step(human_action)
            self.next_state = next_state
            if human_action != 0:
               self.PolSh.Agent.learning(human_action, feedback_demo, self.state, self.next_state)
               self.Qagent.learning(human_action, reward, self.state, self.next_state)
            action = human_action
            self.state = next_state
            if done == True:
                self.demo = False

        envState = {'observation': next_state, 'reward': reward, 'done': done, 'info': info, 'agentAction': action}
        return envState

    def render(self):
        '''
        Gets render from gym.
        Caller:
            - Trial.get_render()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns:
            - return from env.render('rgb_array') (Type: npArray)
              must return the unchanged rgb_array
        '''
        return self.env.render('rgb_array', observer = "global")

    def reset(self):
        '''
        Resets the environment to start new episode.
        Caller:
            - Trial.reset()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns:
            No Return
        '''
        if self.PS:
            self.cnt = 0
            self.PolSh.time_step = 0
            self.PolSh.first_state = self.env.reset()
            #state = self.PolSh.first_state
        else:
            self.env.reset()

    def close(self):
        '''
        Closes the environment at the end of the trial.
        Caller:
            - Trial.close()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns:
            No Return
        '''
        self.env.close()
        

if __name__ == '__main__':
    pass
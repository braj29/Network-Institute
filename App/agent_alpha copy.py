'''
This demo of a TAMER algorithm implmented with HIPPO Gym has been adapted
from code provided by Calarina Muslimani of the Intelligent Robot Learning Laboratory
To use this code with the default setup simply rename this file to agent.py
'''

import gym
import time
import numpy as np
import itertools

#from gym import wrappers

#import highway_env
#from pettingzoo.mpe import simple_v2

import griddly
from griddly import gd
import math
#from Policy_Shaping import PS

#global state

#from gym.utils import play

#This is the code for tile coding features

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
import Policy_Shaping_copy as PS
#from Policy_Shaping_copy import get_state
m = np.zeros((16,14))
#m[0][13] = 1
w = 16
h = 14
#print(num_list)
n = 1
for i in range(w):
    for j in range(h):
        m[i][j] = n
        n += 1

def get_state (state):
    if state[0].shape == (16,14):
        s = np.array(np.where(state[0] == 1)).T.flatten()
        #print(s)
        if s != []:
            x = s[0]
            y = s[1]
            cell = m[x][y]
        else:
            cell = 1
        return int(cell)
    else:
        return int(state)


class PolicyShaping:
    """
        Initialization of Tamer Agent. All values are set to None so they can
        be initialized in the agent_init method.
        """

    def __init__(self):

        self.last_action = None
        # self.previous_tiles = None
        self.first_state = None
        self.current_action = None
        # self.current_tiles= None

        # self.num_tilings =  8
        # self.num_tiles =  8
        self.iht_size = 224
        self.epsilon = 0.001
        # self.x = .08
        self.alpha = 0.001  # self.x/self.num_tilings  #this is step size
        self.num_actions = 5
        self.actions = list(range(self.num_actions))
        self.time_step = 0
        self.experiences = list()
        self.max_n_experiences = 1000
        self.window_size = 1

        # We initialize self.w to three times the iht_size. Recall this is because
        # we need to have one set of weights for each action.
        self.feedback = np.ones((self.iht_size, self.num_actions))

        # We initialize self.mctc to the mountaincar verions of the
        # tile coder that we created

        # self.mctc = MountainCarTileCoder(iht_size=self.iht_size,
        #                                  num_tilings=self.num_tilings,
        #                                  num_tiles=self.num_tiles)
        #
    def learning(self, action, feedback, state, next_state):
        #self.check_add(state)
        #self.check_add(next_state)
        #print(math.exp(self.feedback.loc[self.trans(state),action]))
        # self.feedback.loc[self.trans(state),action] = np.tanh(np.arctanh(self.feedback.loc[self.trans(state),action]) 
        #                       + feedback )
        
        state = get_state(state)
        self.feedback[state][action] += feedback
    
    def action_prob(self, state):
        #self.check_add(state)
        prob = []
        state = get_state(state)
        if all(self.feedback[state] == 0):
            return np.array([1/self.action_space for i in range(self.action_space)])
        for i in range(self.action_space):
            if self.feedback[state][i] < -50:
                self.feedback[state][i] = -50
            prob.append(math.pow(0.60,self.feedback[state][i])/
                        (math.pow(0.60,self.feedback[state][i]) +
                         math.pow(0.40,self.feedback[state][i])) )
        return prob
    
    def update_feedback(self, reward):
        # feedback = 0
        if reward == "good":
            return 1
        elif reward == "bad":
            return -1
        elif reward == "None":
            return 0
    # def argmax(self, q_values):
    #     """argmax with random tie-breaking
    #         Args:
    #         q_values (Numpy array): the array of action values
    #         Returns:
    #         action (int): an action with the highest value
    #         """
    #     top = float("-inf")
    #     ties = []

    #     for i in range(len(q_values)):
    #         if q_values[i] > top:
    #             top = q_values[i]
    #             ties = []

    #         if q_values[i] == top:
    #             ties.append(i)

    #     return np.random.choice(ties)

    # def select_greedy_action(self, tiles):
    #     """
    #         Selects an action using greedy
    #         Args:
    #         tiles - np.array, an array of active tiles
    #         Returns:
    #         (chosen_action, action_value) - (int, float), tuple of the chosen action
    #         and it's value
    #         """
    #     action_values = []
    #     chosen_action = None

    #     for a in range(self.num_actions):
    #         action_values.append(np.sum(self.w[tiles][a]))
    #     # First loop through the weights of each action and populate action_values
    #     # with the action value for each action and tiles instance

    #     if np.random.random() < self.epsilon:
    #         chosen_action = np.random.choice(self.actions)
    #     else:
    #         chosen_action = self.argmax(action_values)

    #     return chosen_action

    # def action_selection(self, state):
    #     #position, velocity = state
    #     # active_tiles=self.mctc.get_tiles(position, velocity)
    #     state = get_state(state)
    #     current_action = self.select_greedy_action(state)
    #     self.current_action = current_action
    #     # self.current_tiles = np.copy(active_tiles)

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
            the environment starts.
            Args:
            state (Numpy array): the state observation from the
            environment's evn_start function.
            Returns:
            The first action the agent takes.
            """
        print("State is", state)
        # position, velocity = state
        state = get_state(state)
        # active_tiles=self.mctc.get_tiles(position, velocity)

        self.current_action = np.random.choice(self.actions)
        # self.current_tiles= np.copy(active_tiles)

        self.experiences.append((self.current_action, state, time.time()))
        return self.current_action

    # def update_reward_function(self, reward):

    #     if reward == 'good':
    #         r = 10
    #     elif reward == 'reallygood':
    #         r = 100
    #     elif reward == 'bad':
    #         r = -10

    #     elif reward == 'None':
    #         #r = -1
    #         return

    #     current_time = time.time()
    #     while len(self.experiences) > 0:
    #         experience = self.experiences[0]

    #         # diff= current_time-experience[2]

    #         # if (diff < .2 or diff > 2):

    #         if experience[2] < current_time - self.window_size:  #
    #             self.experiences.pop(0)

    #         else:
    #             break

    #     # update weights using Algorithm 1 in paper
    #     n_experiences = len(self.experiences)

    #     if n_experiences == 0:
    #         return
    #     weight_per_experience = 1.0 / n_experiences
    #     cred_features = np.zeros((self.num_actions, self.iht_size))

    #     for experience in self.experiences:
    #         exp_features = np.zeros((self.num_actions, self.iht_size))
    #         exp_features[experience[0]][experience[1]] = 1

    #         exp_features *= weight_per_experience
    #         cred_features = np.add(cred_features, exp_features)

    #     error = r - self.w * cred_features
    #     self.w += (.01 * error * cred_features)
        # print(self.w)

# Original HIPPO Gym Agent

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
        self.PS = True
        self.demo = False

        if self.PS:
            np.random.seed(0)
            self.PS = PolicyShaping()
        self.env = gym.make(game, player_observer_type = gd.ObserverType.VECTOR)
        return

    def step(self, action, reward):
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
        #time.sleep(0.5)
        if self.PS ==True and self.demo == False:
            if self.PS.time_step == 0:
                self.PS.agent_start(self.PS.first_state)
                time.sleep(1.5)

            self.PS.time_step += 1
            #self.PS.update_reward_function(reward)
            self.PS.last_action = self.PS.current_action
#            self.PS.previous_tiles = self.PS.current_tiles

            if reward != 'None':
                updated = True
            else:
                updated = False

            observation, reward, done, info = self.env.step(self.PS.current_action)
            action = self.PS.current_action
            if action != 0:
                self.PS.learning(action, reward, self.state, self.next_state)
            
        elif self.demo ==True:
            feedback = "reallygood"
            observation, reward, done, info = self.env.step(action)
            if action != 0:
                self.PS.update_reward_function(feedback)
                #self.PS.experiences.append((self.PS.current_action, observation, time.time()))
                self.PS.learning(action, reward, observation, self.next_state)
            
            if done:
                self.demo = False

        envState = {'observation': observation, 'reward': reward, 'done': done, 'info': info, 'agentAction': action}

        if self.PS:
            self.PS.action_selection(observation)
            #self.PS.experiences.append((self.PS.current_action, observation, time.time()))
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
            self.PS.time_step=0
            self.PS.first_state = self.env.reset()
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

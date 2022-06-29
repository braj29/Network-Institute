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

from time import perf_counter
from time_util import ElapsedTimeThread

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
import math
from Test5 import state_pos

class LevelGenerator:

    def __init__(self, config):
        self._config = config

    def generate(self):
        raise NotImplementedError()


# rapper.build_gym_from_yaml('SokobanTutorial', 'sokoban.yaml', level=0)
class ClustersLevelGenerator(LevelGenerator):
    # BLUE_BLOCK = 'a'
    # BLUE_BOX = '1'
    # RED_BLOCK = 'b'
    # RED_BOX = '2'
    # GREEN_BLOCK = 'c'
    # GREEN_BOX = '3'
    EXIT = 'x'
    AGENT = 'A'

    WALL = 'w'
    SPIKES = 'h'

    def __init__(self, config):
        super().__init__(config)
        self._width = config.get('width', 16)
        self._height = config.get('height', 14)
        self._m_spike = config.get('m_spike', 5)
        self._m_exit = config.get('m_exit', 5)

    def _place_walls(self, map):

        # top/bottom wall

        wall_y = np.array([0, self._height - 1])
        map[:, wall_y] = ClustersLevelGenerator.WALL
        map[3, 9] = ClustersLevelGenerator.WALL

        # left/right wall
        wall_x = np.array([0, self._width - 1])
        map[wall_x, :] = ClustersLevelGenerator.WALL

        return map

    def generate(self, pos):
        map = np.chararray((self._width, self._height), itemsize=2)
        map[:] = '.'
        # print(map)
        # Generate walls
        map = self._place_walls(map)
        #map.tofile('test.txt')
        # print(map)

        # all possible locations
        possible_locations = []
        for w in range(1, self._width - 1):
            for h in range(1, self._height - 1):
                possible_locations.append([w, h])

        def addChar(text, char, place):
            return text[:place - 1] + char + text[place + 1:]

        level_string = """w w w w w w w w w w w w w w w w
        w w . . . . . w w w . . . . x w
        w w . w w w . w w w . w w w w w
        w w . w . w . . . . . . . w t w
        w w . w . w w w w . w w w w . w
        w . . . . . . w w w w . . . . w
        w . w w w w . w w w w . w w w w
        w . . . . w . . . . . . . . . w
        w w w w w w . w w w w . w w . w
        w . . . . . . . . . . . . . . w
        w . w w w w . w w w . w w w . w
        w . w . w w . w w w . w w w w w
        w . w . . . . . t . . . . . . w
        w w w w w w w w w w w w w w w w"""

        places = "."

        occurrences = level_string.count(places)

        indices = [i for i, a in enumerate(level_string) if a == places]

        #print(indices)

        # Place Agent
        #agent_location_idx = np.random.choice(indices)
        level_string = addChar(level_string, "A", pos)
        # print(level_string)
        # agent_location = possible_locations[agent_location_idx]
        # print(agent_location_idx)
        # map[agent_location[0], agent_location[1]] = ClustersLevelGenerator.AGENT

        # level_string = ''
        # for h in range(0, self._height):
        #     for w in range(0, self._width):
        #         level_string += map[w, h].decode().ljust(4)
        #     level_string += '\n'

        # print(type(level_string))
        return level_string




#print(reward, feedback)
def update_feedback(reward):
    # feedback = 0
    if reward == "good":
        return 0.2
    elif reward == "bad":
        return -0.2
    elif reward == "None":
        return 0


epsilon_max = 1
epsilon_min = 0.1
eps_decay = 3000

weight_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
                -1. * frame_idx / eps_decay)

def get_state(state):
    if len(state) != 2:
        s = np.array(np.where(state[0] == 1)).T.flatten()
        if s != []:
            x = s[0]
            y = s[1]
            if (x, y) != (0, 0):

                return (str(x), str(y))
            else:
                return (str(1), str(1))

        else:
            return (str(1), str(1))

    else:
        return state


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

        self.config = {
            'width': 16,
            'height': 14
        }

        self.phase = 1
        self.total_reward = 100
        self.demo_steps = 100
        self.feedback_steps = 100
        self.demo = False
        self.env = gym.make("GDY-Labyrinth-v0", player_observer_type=gd.ObserverType.VECTOR, level = 0, max_steps = 1000)
        #state_position = state_pos()
        #print(state_position)
        #self.env.reset(level_string=level_generator.generate(state_position))
        self.env.reset()
        self.action_space = self.env.action_space.n
        # print(action_space)
        self.observation_space = self.env.observation_space.shape
        # print(observation_space)

        self.PS = True
        if self.PS:
            np.random.seed(0)
            self.PolSh = PS.PSAgent(self.action_space, self.observation_space)
            self.Qagent = QL.QLAgent(self.action_space, self.observation_space, epsilon=0.2, mini_epsilon=0.01,
                                     decay=0.999)

        self.elaps_time = ElapsedTimeThread()
        #self.elaps_time.start()

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

        if self.demo == False and self.phase not in [1,2]:
            if self.time_step == 0:
                self.state =self.first_state
                time.sleep(1.5)
                
            self.time_step += 1
            self.last_state = np.copy(self.state)
            #print((self.last_state))
            #Q_prob = self.Qagent.action_prob(state)
            #P_prob = self.PolSh.Agent.action_prob(self.state)
            self.cnt += 1
            weight = weight_by_frame(self.cnt)
            #print(weight)

            prob = self.Qagent.action_prob(self.last_state) + weight * np.asarray(self.PolSh.action_prob(self.last_state))
            #print(prob, sum(prob)) 
            
            prob = np.asarray(prob)
            #sum = np.sum(prob)
            #prob = prob/sum
            #prob = prob/5
            #print(prob)

            #print(prob==prob.max())
            action = np.random.choice(np.flatnonzero(prob == prob.max())) + 1
            #action_space = list(range(1, self.action_space))
            #action = np.random.choice(action_space, p=prob / sum(prob))

            #print(action)
            #prob_d =
            #a = [1,2,3,4]
            #action = np.random.choice(a,p = prob)
            #stochastic probability distribution
            feedback = update_feedback(human_feedback)

            next_state, reward, done, info = self.env.step(action)
            if self.phase > 5:
                if action != 0 and self.feedback_steps > 0:
                    self.PolSh.learning(action, feedback, self.last_state, next_state)
                    self.Qagent.learning(action, reward, self.last_state, next_state)
                    if feedback != 0:
                        self.feedback_steps -= 1

            self.state = next_state
            self.total_reward += reward
            if done and self.phase < 6:
                self.phase+=1
                self.reset()



        elif self.demo == True or self.phase in [1,2]:
            if self.demo_steps == 0:
                self.demo = False
            feedback_demo = 1
            if self.phase ==1:
                self.last_state = self.first_state
            else:
                self.last_state = np.copy(self.state)
            next_state, reward, done, info = self.env.step(human_action)
            #PSA.next_state = next_state
            if self.phase > 5:
                if human_action != 0 and self.demo_steps > 0:
                   self.PolSh.learning(human_action, feedback_demo, self.last_state, next_state)
                   self.Qagent.learning(human_action, reward, self.last_state, next_state)
                   #print(get_state(self.last_state), get_state(self.state))
                   self.demo_steps -= 1
            action = human_action
            #PSA.state = next_state
            self.state = next_state

            self.total_reward += reward
            if done == True:
                if self.phase < 6:
                    self.phase += 1
                    self.reset()

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
            self.time_step = 0
            if self.phase in [3,6,7]:
                level_generator = ClustersLevelGenerator(self.config)
                state_position = state_pos()
                self.first_state = self.env.reset(level_string=level_generator.generate(state_position))
            # print(state_position)
            #self.env.reset(level_string=level_generator.generate(state_position))
            elif self.phase in [1,2,4,5]:
                self.first_state = self.env.reset()
            else:
                level_generator = ClustersLevelGenerator(self.config)
                state_position = state_pos()
                self.first_state = self.env.reset(level_string=level_generator.generate(state_position))

            #state = self.PolSh.first_state
            #self.time_start = perf_counter()

        else:
            #level_generator = ClustersLevelGenerator(self.config)
            #self.env.reset(level_string=level_generator.generate())
            self.elaps_time.stop()
            self.elaps_time.join
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
        #self.elaps_time.stop()
        #self.elaps_time.join()
        

if __name__ == '__main__':
    pass
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym
import random
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
env = gym.make("CartPole-v0")
env.reset()


training_runs = 10000,
goal_steps = 500


def test():
    for testruns in range(0,5):
        score = 0
        env.reset()
        for i in range(0,200):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            score = score + reward
        print(score)
test()

            
class Cartpole():
    def __init__ (self, runs = 1000, win_score = 195, gamma = 1, epsilon = 1, epsilon_min = 0.01, epsilon_decay = 0.995, alpha = 0.01, alpha_decay = 0.01, batch_size = 64):
        self.memory = []
        self.env = gym.make("CartPole-v0")        
        self.runs = runs
        self.win_score = win_score
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        
        #model
        self.model = Sequential()
        self.model.add(Dense(24, input_dim = 4, activation = 'tanh'))
        self.model.add(Dense(48, activation = 'tanh'))
        self.model.add(Dense(2, activation = 'linear'))
        self.model.compile(loss = 'mse', optimizer = Adam(lr = self.alpha, decay = self.alpha_decay))
        
    def perform_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))
    
    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))
            
    def preprocess_state(self, state):
        return np.reshape(state, [1,4])
        
    def save_memory(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        
    def replay(self):
        x,y = [], []
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        for state, action, reward, next_state, done in batch:
            y_ind = self.model.predict(state)
            if done == True:
                 y_ind[0][action] = reward
            else:    
                y_ind[0][action] = reward + self.gamma*np.max(self.model.predict(next_state)[0])
            x.append(state[0])
            y.append(y_ind[0])
        
        self.model.fit(np.array(x), np.array(y), batch_size = len(x), verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_decay
        
    def run_game(self):
        for i in range(1,self.runs):
            state = self.preprocess_state(self.env.reset())
            done = False
            while not done:
                action = self.perform_action(state, self.get_epsilon(i))
                next_state, reward, done, info = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.save_memory(state, action, reward, next_state, done)
                state = next_state
                
            self.replay()
            print(i)
            
    def test(self):
        for _ in range(5):
            score = 0
            state = self.preprocess_state(self.env.reset())
            for _ in range(200):
                
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                state = next_state
                score = score+reward
            print(score)
cartpole = Cartpole()
cartpole.run_game()
cartpole.test()
#cartpole.run_game()
                        
    
        
        
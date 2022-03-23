import numpy as np
import torch as T
import torch.nn as nn
import gym
from networks import Q_network
from utilities import argmax
from copy import deepcopy
from buffer import ReplayBuffer
import math 
class SEQ():
    
    
    def __init__(self, state_dim, action_dim, layers, epsilon_start, epsilon_end, epsilon_decay, discount, lr, buffer_size, batch_size, writer):
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        self.layers = layers 
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.discount = discount
        self.mse = nn.MSELoss()
        self.checkpoint_file='tmp/seq/'
        self.writer=writer
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_steps_done = 0
        
        # Buffer 
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.state_dim, self.action_dim)
        # Q Network
        self.q_network = Q_network(self.layers, self.lr, nn.ReLU)
        
        self.q_network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        
        )
        self.optimizer = T.optim.Adam(self.q_network.parameters(), lr=0.01)

        self.target_q_network = deepcopy(self.q_network)
        
        
    def save_model(self, name):
        T.save(self.q_network.state_dict(), self.checkpoint_file + name)
        
    def choose_action(self, state, i, target_network=False):
        epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.epsilon_steps_done / self.epsilon_decay)
        self.epsilon_steps_done += 1
        if target_network:
            q_values = self.target_q_network(state).detach().numpy()
        else:
            q_values = self.q_network(state).detach().numpy()
        if np.random.random() > epsilon_threshold:
            
            action_index = argmax(q_values)
            return action_index, q_values[:, action_index]
        else:
            action_index = np.random.choice([0,1])
            return action_index, q_values[:, action_index]
    
    def learn(self, batch, i):
        states, actions, rewards, states_, terminals = batch
        # print(sum(rewards))
        states = T.tensor(states)
        actions = T.tensor(actions).unsqueeze(-1)
        rewards = T.tensor(rewards)
        states_ = T.tensor(states_)
        terminals = T.tensor(terminals)
        
        q_values = self.q_network(states).gather(1, actions.type(T.int64))
        
        
        q_values_ = self.target_q_network(states_).detach()
        
        backup = rewards + self.discount * T.amax(q_values_, dim=1) * (1 - terminals)
        
        self.writer.add_scalar('Loss/backup', backup.sum().item(), i)
        
        loss = self.mse(backup.detach(), q_values.squeeze())
        self.writer.add_scalar('Loss/q_net_loss', loss.item(), i)
        # print(loss)
        
        # self.q_network.optimizer.zero_grad()
        # loss.backward()
        # self.q_network.optimizer.step()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        
        
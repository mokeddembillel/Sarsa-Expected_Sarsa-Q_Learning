import numpy as np 

class ReplayBuffer():
    
    def __init__(self, buffer_max_size, batch_size, state_dim, action_dim):
        
        self.buffer_max_size = buffer_max_size
        self.batch_size = batch_size
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        
        self.counter = 0
        
        self.states = np.zeros((self.buffer_max_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_max_size), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_max_size), dtype=np.float32)
        self.states_ = np.zeros((self.buffer_max_size, self.state_dim), dtype=np.float32)
        self.terminals = np.zeros((self.buffer_max_size), dtype=np.float32)
    
        
    def store(self, state, action, reward, state_, terminal):
        index = self.counter % self.buffer_max_size
        self.states[index] = state
        self.actions[index] = action 
        self.rewards[index] = reward
        self.states_[index] = state_
        self.terminals[index] = terminal
        self.counter += 1
        
    def sample(self):
        buffer_size = min(self.buffer_max_size, self.counter)
        batch = np.random.choice(buffer_size, self.batch_size)
        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        states_ = self.states_[batch]
        terminals = self.terminals[batch]
        
        return states, actions, rewards, states_, terminals
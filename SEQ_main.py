import numpy as np
import torch as T
import gym 
from SEQ_torch import SEQ 
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from utilities import argmax
from networks import Q_network

          
if __name__ == '__main__':
    writer = SummaryWriter()
    # Init env 
    env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    state_bounds = (env.observation_space.low, env.observation_space.high)
    env.action_space.sample()
  
    
    num_episodes = 1000
    batch_size = 128
    # Init agent 
    agent = SEQ(state_dim=state_dim, 
                action_dim=action_dim, 
                layers=[state_dim, 32, 32, action_dim], 
                epsilon_start = 0.9,
                epsilon_end = 0.05,
                epsilon_decay = 500,
                discount=0.9, 
                lr=0.1, 
                buffer_size=int(1000), 
                batch_size=batch_size,
                writer=writer)
    
    # Main Loop
    
    best_average_reward = float('-inf')
    episode_reward = []
    episode_steps = []
    
    for i in range(num_episodes):
        """In each episode, the agent runs on the environment 
        until it falls or reaches the maximum number of steps in an episode"""
        state= env.reset()
        terminal = False
        rewards_sum = 0
        steps_sum = 0
        
        while not terminal:
            action, _ = agent.choose_action(T.tensor(state).unsqueeze(0), i)
            state_, reward, terminal, _  = env.step(action) 
            a = env.step(action) 
            state_, reward, terminal, _ = a
            rewards_sum += reward
            steps_sum += 1
            agent.buffer.store(state, action, reward, state_, terminal)
            
            if agent.buffer.counter > batch_size:
                batch = agent.buffer.sample()
                agent.learn(batch, i)
            state = state_ 
            
            env.render()
            
        writer.add_scalar('Reward/train', rewards_sum, i)
        episode_reward.append(rewards_sum)
        episode_steps.append(steps_sum)
        
        
        best_reward_tmp = np.mean(episode_reward[-5:])
        if best_reward_tmp > best_average_reward:
            best_average_reward = best_reward_tmp 
            agent.save_model('Q_learning')
            print('Saving a model with a best average reward: ', best_average_reward)
        
        if i % 10 == 0:
            agent.target_q_network = deepcopy(agent.q_network)
        
        print('Episode: ', i, ' -- Expected reward: ', rewards_sum, ' -- number of steps: ', steps_sum)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
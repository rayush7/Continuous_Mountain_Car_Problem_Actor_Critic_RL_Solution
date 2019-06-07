#from ActorNetwork import Actor
#from CriticNetwork import Critic
#from ReplayBuffer import ReplayBuffer
from mountaincar import MountainCar
#from OU import OU_Noise

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os

# To delete
import random
import copy

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


"""
Contains the definition of the agent that will run in an
environment.
"""

class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """
        mc = MountainCar()

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change, but your initial
        location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        return np.random.normal(-10, 10)

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass


    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 40)
        #self.l2 = nn.Linear(256, action_dim)
        self.l2 = nn.Linear(40, 30)
        self.l3 = nn.Linear(30, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        #x = self.max_action * torch.tanh(self.l2(x))
        
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 40)
        #self.l2 = nn.Linear(256, 1)

        self.l2 = nn.Linear(40, 30)
        self.l3 = nn.Linear(30, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        #x = self.l2(x)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    
class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.3):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state
    
class ReplayBuffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=10000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        if batch_size>len(self.storage):
            batch_size = len(self.storage)
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
        
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        u = np.stack(u, axis=0)
        r = np.concatenate(r, axis=0)
        d = np.array(d)


        return x, y, u, r.reshape(-1, 1), d.reshape(-1, 1)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DDPG:
    def __init__(self):
        self.mc = MountainCar()
        
        # Params for env
        self.state_dim  = 2
        self.action_dim = 1
        self.max_action = self.mc.F
        
        # Params for training
        self.batch_size = 32
        self.gamma      = 0.99
        self.tau        = 0.001
        self.actor_lr   = 0.0001
        self.critic_lr  = 0.0004
        self.noise      = OU_Noise(self.action_dim, 0)
        self.explore    = 100000
        self.epsilon    = 0.3
        self.counter    = 0

        # Init 
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.actor_lr)

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.critic_lr)
        self.replay_buffer = ReplayBuffer()
        self.num_training = 0

        self.state_curr = None
        self.state_prev = None

        # Load
        #if not os.path.isdir('./Model/'):
        #    os.mkdir('./Model')
        #if os.path.isfile('./Model_small/actor.pth') and (os.path.isfile('./Model_small/critic.pth')):
        #    self.load('./Model_small/')

    def act(self, state):
        #return np.random.normal(-10, 10)

            state = np.array(state).reshape(1, -1)
            self.state_prev = self.state_curr # we should recored the state
            self.state_curr = state

            state  = torch.FloatTensor(state).to(device)
            action = self.actor(state).cpu().data.numpy().flatten()
            noise  = self.noise.sample()*max(self.epsilon, 0)
            self.epsilon -= 1.0/self.explore
        
            action = action+noise
            #action_clip = np.clip(action, a_min=-self.max_action, a_max=self.max_action)
            return action
        
    def reset(self, x_range):
        self.noise.reset()


    def reward(self, observation, action, reward):
        
        if self.state_prev is not None:
            self.replay_buffer.push((self.state_prev, self.state_curr, action, reward, np.float(False)))
        self.counter += 1
        #if self.counter%20==0:
        self.update()
        
    def update(self):
        # Sample replay buffer
        if len(self.replay_buffer.storage)==0:
            return None

        for ep in range(1):
            x, y, u, r, d = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        

    def save(self, directory):
        torch.save(self.actor.state_dict(), os.path.join(directory, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(directory, 'critic.pth'))
        #print("====================================")
        #print("Model has been saved...")
        #print("====================================")

    def load(self, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(directory, 'critic.pth')))
        #print("====================================")
        #print("model has been loaded...")
        #print("====================================")


#-----------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(2, 256)
        
        self.action_layer = nn.Linear(256, 2)
        self.value_layer = nn.Linear(256, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.actions = []
        
        

    def forward(self, observation):
        # Convert tuple into tensor
        observation_as_list = []
        observation_as_list.append(observation[0])
        observation_as_list.append(observation[1])
        observation_as_list = np.asarray(observation_as_list)
        observation_as_list = observation_as_list.reshape(1,2)
        observation = observation_as_list
        
        state = torch.from_numpy(observation).float()
        forward_state = F.tanh(self.affine(state))
        #forward_state = (self.affine(state))

        state_value = self.value_layer(forward_state)
        action_parameters = (self.action_layer(forward_state)) # A voir quelle activation function mettre
        action_distribution = Normal(action_parameters[0][0], action_parameters[0][1])
        
        action = action_distribution.sample() # Torch.tensor; action
        action_log_prob = action_distribution.log_prob(action)
#        while math.isnan(action_log_prob.item()) == True:
##            forward_state = F.tanh(self.affine(state))
#             #action_parameters = (self.action_layer(forward_state)) # A voir quelle activation function mettre
#             action_parameters[0][0] += 0.01
#             action_parameters[0][1] += 0.01
#             action_distribution = Normal(action_parameters[0][0], action_parameters[0][1])
##            action = action_distribution.sample()
#             print("ca bloque", action)
#             action_log_prob = action_distribution.log_prob(action)
#            print(action_distribution)
        self.logprobs.append(action_log_prob)
        self.state_values.append(state_value)
        #print(action.item())
        return action.item() # Float element
        
    
    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in list(zip(self.logprobs, self.state_values, rewards)):
            
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            
            value_loss = F.smooth_l1_loss(value, reward)
            #print(action_loss)
            loss += (action_loss + value_loss)  
            
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]




class ActorCriticAgent():
    def __init__(self):
        """Init a new agent.
        """
        #self.theta = np.zeros((3, 2))
        #self.state = RandomAgent.reset(self,[-20,20])
        
        self.count_episodes = -1
        self.max_position = -0.4
        self.epsilon = 0.5
        self.gamma = 0.99
        self.running_rewards = 0
        self.policy = ActorCritic()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0001, betas=(0.9, 0.999))
        self.check_new_episode = 1
        self.count_iter = 0
        self.successful = 0        
        
    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change, but your initial
        location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        self.epsilon = (self.epsilon * 0.94)
        self.count_episodes += 1
        self.victory = False
        #if self.count_episodes == 99:
        #    plt.plot([i for i in range(99)], self.policy.all_loss, 'ro')
        #    plt.show()
            
        
        #return (np.random.uniform(x_range[0],x_range[1]), np.random.uniform(-20,20))

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        
#        observation_as_list = []
#        observation_as_list.append(observation[0])
#        observation_as_list.append(observation[1])
#        observation_as_list = np.asarray(observation_as_list)
#        observation_as_list = observation_as_list.reshape(1,2)
#        observation = observation_as_list
        
        
        if np.random.rand(1) < self.epsilon:
            return np.random.uniform(-1,1)
        
        else:
            x = observation[0]
            vx = int((observation[1]+20)//5)
            action = (x,vx)
            action = self.policy(observation)
            return action
        
        

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        
        if reward > 0:
            print(observation)
            self.successful += 1
            print(self.successful)
            self.policy.rewards.append(reward)
            index = len(self.policy.rewards)
            for i in range(index-12,index):
                self.policy.rewards[i] += 30
            self.optimizer.zero_grad()
            self.loss = self.policy.calculateLoss(self.gamma)
            #self.policy.all_loss.append(self.loss)
            self.loss.backward()
            self.optimizer.step() 
            self.policy.clearMemory()
            self.count_iter = 0
            
        else:
        
            self.count_iter +=1
            self.policy.rewards.append(reward)
            if self.count_iter == 400:
                # We want first to update the critic agent:
                for i in range(400):
                    self.policy.rewards[i] -= 10  
                self.optimizer.zero_grad()
                self.loss = self.policy.calculateLoss(self.gamma)
                #self.policy.all_loss.append(self.loss)
                self.loss.backward()
                self.optimizer.step() 
                self.policy.clearMemory()
                self.count_iter = 0


#-----------------------------------------------------------


Agent = ActorCriticAgent

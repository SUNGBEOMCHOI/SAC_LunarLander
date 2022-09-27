import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.distributions import Categorical

class ReplayBuffer:
    def __init__(self, buf_size, device):
        """
        Make replay buffer of deque structure which stores data samples
        
        Args:
            buf_size: deque max size
            device: Device for train, It can be 'cpu' or 'cuda'

        Returns:
            replay_buffer: replay buffer
        """
        self.memory = deque(maxlen=buf_size)
        self.device = device

    def get_random_sample(self, batch_size):
        """
        Return random samples of batch size

        Args:
            batch_size: Number you want to extract
        
        Returns:
            state_tensor: Torch tensor of state samples of shape [batch_size, state_dim]
            action_tensor: Torch tensor of action samples of shape [batch_size]
            reward_tensor: Torch tensor of reward samples of shape of shape [batch_size, 1]
            next_state_tensor: Torch tensor of next state samples of shape [batch_size, state_dim]
            done_tensor: Torch tensor of done booleans of shape [batch_size]
        """
        random_samples = random.sample(self.memory, k=batch_size)
        for sample in random_samples:
            state = np.expand_dims(sample.state, axis=0)
            action = sample.action
            reward = np.expand_dims(sample.reward, axis=0)
            next_state = np.expand_dims(sample.next_state, axis=0)
            done = sample.done
            try:                
                state_tensor = np.concatenate((state_tensor, state), axis=0)
                action_tensor = np.concatenate((action_tensor, action), axis=0)
                reward_tensor = np.concatenate((reward_tensor, reward), axis=0)
                next_state_tensor = np.concatenate((next_state_tensor, next_state), axis=0)
                done_tensor = np.concatenate((done_tensor, done), axis=0)
            except:
                state_tensor = state
                action_tensor = action
                reward_tensor = reward
                next_state_tensor = next_state
                done_tensor = done
        state_tensor = torch.from_numpy(state_tensor).to(self.device)
        action_tensor = torch.from_numpy(action_tensor).type(torch.int64).to(self.device)
        reward_tensor = torch.from_numpy(reward_tensor).type(torch.float32).to(self.device)
        next_state_tensor = torch.from_numpy(next_state_tensor).to(self.device)
        done_tensor = torch.from_numpy(done_tensor).to(self.device)
        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor

    def append(self, x):
        """
        Append x into replay memory

        Args:
            x: Input
        """
        self.memory.append(x)

def loss_func():
    """
    Return value, Q, policy criterion

    Returns:
        loss_list: List contains value_criterion, q_criterion, policy_criterion
    """
    value_criterion = nn.MSELoss()
    q_criterion = nn.MSELoss()
    # policy_criterion = nn.MSELoss()
    policy_criterion = nn.KLDivLoss(reduction='batchmean')
    loss_list = [value_criterion, q_criterion, policy_criterion]
    return loss_list

def optim_func(model, learning_rate):
    """
    Return value, Q, policy optimizer

    Returns:
        optim_list: List contains value_optim, q_optim, policy_optim
    """
    value_optim = optim.Adam(model.value_net.parameters(), lr=learning_rate)
    q_optim = optim.Adam(model.q_net.parameters(), lr=learning_rate)
    policy_optim = optim.Adam(model.policy_net.parameters(), lr=learning_rate)
    optim_list = [value_optim, q_optim, policy_optim]
    return optim_list

def scheduler_func(optim_list):
    """
    Return value, Q, policy learning rate scheduler

    Args:
        optim_list: List of value, q, policy optimizer

    Returns:
        scheduler_list: List contains value, q, policy learning rate scheduler
    """
    scheduler_list = []
    for optimizer in optim_list:
        scheduler_list.append(MultiStepLR(optimizer, milestones=[30000,80000], gamma=0.1))
    return scheduler_list

def running_mean(x, N):
    """
    Calculate moving average of x

    Args:
        x: Numpy array you want to calculate moving average
        N: Window size

    Returns:
        Moving average result
    """
    cumsum = np.cumsum(np.insert(x, 0, [x[0]]*N))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_progress(history, save=True):
    """
    plot train progress, x-axis: episode, y-axis: total rewards
    
    Args:
        history: Dictionary which contains score of episodes
    """    
    score_array = np.array(history['score'])
    loss_array = np.array(history['loss'])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    moving_average_score = running_mean(score_array, N=200)
    plt.plot(np.arange(len(moving_average_score)), moving_average_score)
    plt.xlabel('Time steps')
    plt.ylabel('Score')
    plt.title('Train Progress[Score]')

    plt.subplot(1, 2, 2)
    moving_average_loss = running_mean(loss_array, N=200)
    plt.plot(np.arange(len(moving_average_loss)), moving_average_loss)
    plt.xlabel('Time steps')
    plt.ylabel('Loss')
    plt.title('Train Progress[Loss]')
    if save:
        plt.savefig(f'./train_progress/step_{len(loss_array)}.png')
    else:
        plt.show()
    plt.close()

def update_params(replay_buffer, model, criterion_list, optim_list, discount_factor, batch_size):
    """
    Update model networks' parameters with replay buffer

    Args:
        replay_buffer: Replay memory that contains date samples
        model: SAC model
        criterion_list: List contains value_criterion, q_criterion, policy_criterion
        optim_list : criterion_list: List contains value_optimizer, q_optimizer, policy_optimizer
        discount_factor
        batch_size

    Returns:
        total_loss: sum of value loss, q loss, and policy loss
    """
    state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor =\
        replay_buffer.get_random_sample(batch_size)
    
    value_criterion, q_criterion, policy_criterion = criterion_list
    value_optim, q_optim, policy_optim = optim_list

    # update value network
    value_optim.zero_grad()
    value_prediction = model.value_net(state_tensor) # [B, 1]
    target_action_prob = model.policy_net(state_tensor) # [B, 4]
    m = Categorical(target_action_prob)
    target_action = m.sample().unsqueeze(dim=1)
    value_target = torch.gather(model.q_net(state_tensor), dim=-1, index=target_action) -\
        torch.log(torch.gather(model.policy_net(state_tensor), dim=-1, index=target_action)) # [B, 1]
    value_loss = value_criterion(value_prediction, value_target.detach())
    value_loss.backward()
    value_optim.step()

    # update q network
    q_optim.zero_grad()
    q_output = model.q_net(state_tensor) # [B, 4]
    selected_action = action_tensor.unsqueeze(dim=1).detach() # [B, 1]
    q_prediction = torch.gather(q_output, dim=-1, index=selected_action)# [B, 1]
    target_q = torch.zeros_like(reward_tensor) # [B, 1]
    target_q[done_tensor] = reward_tensor[done_tensor] # case done
    # case not done
    target_q[~done_tensor] = reward_tensor[~done_tensor] +\
        discount_factor*model.target_value_net(next_state_tensor[~done_tensor])
    q_loss = q_criterion(q_prediction, target_q.detach())
    q_loss.backward()
    q_optim.step()

    # update policy network
    policy_optim.zero_grad()
    policy_prediction = model.policy_net(state_tensor) # [B, 4]
    target_policy = F.softmax(model.q_net(state_tensor), dim=-1) # [B, 4]
    policy_loss = policy_criterion(policy_prediction, target_policy.detach())
    policy_loss.backward()
    policy_optim.step()

    total_loss = (value_loss + q_loss + policy_loss).item()
    return total_loss
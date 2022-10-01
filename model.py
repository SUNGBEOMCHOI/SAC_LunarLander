import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Encoder(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        color, width, height = self.state_dim
        self.model = nn.Sequential(
            nn.Conv2d(color, 8, 8, 4),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Return state values corresponding input states

        Args:
            x: input state of size

        Returns:
            Torch tensor of state values of size [batch_size, 1]
        """
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x

    def get_output_size(self):
        """
        Return output size of encoder network

        Returns:
            Output size of encoder network
        """
        color, width, height = self.state_dim
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size-1) - 1) // stride + 1
                
        conv1_w = conv2d_size_out(width, 8, 4)
        conv2_w = conv2d_size_out(conv1_w, 4, 2)
        conv3_w = conv2d_size_out(conv2_w, 3, 1)

        conv1_h = conv2d_size_out(height, 8, 4)
        conv2_h = conv2d_size_out(conv1_h, 4, 2)
        conv3_h = conv2d_size_out(conv2_h, 3, 1)

        output_size = conv3_w * conv3_h * 32
        return output_size
        


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Return state values corresponding input states

        Args:
            x: input state of size

        Returns:
            Torch tensor of state values of size [batch_size, 1]
        """
        return self.model(x)

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        """
        Return Q values corresponding input states

        Args:
            x: input state of size

        Returns:
            Torch tensor of state values of size [batch_size, action_dim]
        """
        return self.model(x)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Return Q values corresponding input states

        Args:
            x: input state of size

        Returns:
            Torch tensor of state values of size [batch_size, action_dim]
        """
        action_probs = self.model(x)
        return action_probs + (action_probs <= 1e-8).float() * 1e-8
        

class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=128):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.encoder = Encoder(state_dim).to(device)
        self.target_encoder = Encoder(state_dim).to(device)
        linear_input_dim = self.encoder.get_output_size()
        self.value_net = ValueNet(linear_input_dim, hidden_dim).to(device)
        self.target_value_net = ValueNet(linear_input_dim, hidden_dim).to(device)
        self.q_net = QNet(linear_input_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNet(linear_input_dim, action_dim, hidden_dim).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        
    def forward(self, x):
        raise NotImplementedError

    def get_action(self, x, validation=False):
        """
        Return action corresponding to input state

        x: input state

        Returns
            action
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        x = x.to(self.device)
        x = self.target_encoder(x)
        if validation:
            action = torch.argmax(self.policy_net(x))
        else:
            m = Categorical(self.policy_net(x))
            action = m.sample()
        return action
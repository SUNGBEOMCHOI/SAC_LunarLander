import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        if isinstance(self.state_dim, int):
            self.model = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            color, width, height = state_dim
            self.conv1 = nn.Conv2d(color, 8, 8, 4)
            self.bn1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8, 16, 4, 2)
            self.bn2 = nn.BatchNorm2d(16)
            self.conv3 = nn.Conv2d(16, 32, 3, 1)
            self.bn3 = nn.BatchNorm2d(32)

            def conv2d_size_out(size, kernel_size, stride):
                return (size - (kernel_size-1) - 1) // stride + 1
                
            conv1_w = conv2d_size_out(width, 8, 4)
            conv2_w = conv2d_size_out(conv1_w, 4, 2)
            conv3_w = conv2d_size_out(conv2_w, 3, 1)

            conv1_h = conv2d_size_out(height, 8, 4)
            conv2_h = conv2d_size_out(conv1_h, 4, 2)
            conv3_h = conv2d_size_out(conv2_h, 3, 1)

            linear_input_size = conv3_w * conv3_h * 32

            self.fc1 = nn.Linear(linear_input_size, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Return state values corresponding input states

        Args:
            x: input state of size

        Returns:
            Torch tensor of state values of size [batch_size, 1]
        """
        if isinstance(self.state_dim, int):
            return self.model(x)
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.fc1(x.view(x.size(0), -1)))
            x = self.fc2(x)
            return x

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        if isinstance(self.state_dim, int):
            self.model = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            color, width, height = state_dim
            self.conv1 = nn.Conv2d(color, 8, 8, 4)
            self.bn1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8, 16, 4, 2)
            self.bn2 = nn.BatchNorm2d(16)
            self.conv3 = nn.Conv2d(16, 32, 3, 1)
            self.bn3 = nn.BatchNorm2d(32)

            def conv2d_size_out(size, kernel_size, stride):
                return (size - (kernel_size-1) - 1) // stride + 1
                
            conv1_w = conv2d_size_out(width, 8, 4)
            conv2_w = conv2d_size_out(conv1_w, 4, 2)
            conv3_w = conv2d_size_out(conv2_w, 3, 1)

            conv1_h = conv2d_size_out(height, 8, 4)
            conv2_h = conv2d_size_out(conv1_h, 4, 2)
            conv3_h = conv2d_size_out(conv2_h, 3, 1)

            linear_input_size = conv3_w * conv3_h * 32

            self.fc1 = nn.Linear(linear_input_size, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Return q values corresponding input states for all actions

        Args:
            x: input state of size

        Returns:
            Torch tensor of q values of size [batch_size, action_space]
        """
        if isinstance(self.state_dim, int):
            return self.model(x)
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.fc1(x.view(x.size(0), -1)))
            x = self.fc2(x)
            return x

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        if isinstance(self.state_dim, int):
            self.model = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            )
        else:
            color, width, height = state_dim
            self.conv1 = nn.Conv2d(color, 8, 8, 4)
            self.bn1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8, 16, 4, 2)
            self.bn2 = nn.BatchNorm2d(16)
            self.conv3 = nn.Conv2d(16, 32, 3, 1)
            self.bn3 = nn.BatchNorm2d(32)

            def conv2d_size_out(size, kernel_size, stride):
                return (size - (kernel_size-1) - 1) // stride + 1
                
            conv1_w = conv2d_size_out(width, 8, 4)
            conv2_w = conv2d_size_out(conv1_w, 4, 2)
            conv3_w = conv2d_size_out(conv2_w, 3, 1)

            conv1_h = conv2d_size_out(height, 8, 4)
            conv2_h = conv2d_size_out(conv1_h, 4, 2)
            conv3_h = conv2d_size_out(conv2_h, 3, 1)

            linear_input_size = conv3_w * conv3_h * 32

            self.fc1 = nn.Linear(linear_input_size, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Return actions' probability corresponding input states

        Args:
            x: input state of size [batch_size, state_dim]

        Returns:
            Torch tensor of actions of size [batch_size, action_space]
        """
        if isinstance(self.state_dim, int):
            action_probs = self.model(x)
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.fc1(x.view(x.size(0), -1)))
            action_probs = self.softmax(self.fc2(x))
        return action_probs + (action_probs <= 0.0).float() * 1e-8
        

class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=128):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.value_net = ValueNet(state_dim, hidden_dim).to(device)
        self.target_value_net = ValueNet(state_dim, hidden_dim).to(device)
        self.q_net = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
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
        if not isinstance(self.state_dim, int): # for image input
            x = x.unsqueeze(0)
        if validation:
            action = torch.argmax(self.policy_net(x))
        else:
            m = Categorical(self.policy_net(x))
            action = m.sample()
        return action
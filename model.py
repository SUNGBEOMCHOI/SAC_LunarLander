import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

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
            x: input state of size [batch_size, state_dim]

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
        Return q values corresponding input states for all actions

        Args:
            x: input state of size [batch_size, state_dim]

        Returns:
            Torch tensor of q values of size [batch_size, action_space]
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
        Return actions' probability corresponding input states

        Args:
            x: input state of size [batch_size, state_dim]

        Returns:
            Torch tensor of actions of size [batch_size, action_space]
        """
        action_probs = self.model(x)
        return action_probs + (action_probs <= 0.0).float() * 1e-8

class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=128):
        super().__init__()
        self.device = device
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
        x = x.to(self.device)
        if validation:
            action = torch.argmax(self.policy_net(x))
        else:
            m = Categorical(self.policy_net(x))
            action = m.sample()
        return action
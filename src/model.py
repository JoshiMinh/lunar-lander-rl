import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128, dueling=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            dueling (bool): Whether to use Dueling DQN architecture
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = dueling
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        if self.dueling:
            # Value stream
            self.value_stream = nn.Linear(fc2_units, 1)
            # Advantage stream
            self.advantage_stream = nn.Linear(fc2_units, action_size)
        else:
            self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            # Combine value and advantage: Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            return self.fc3(x)

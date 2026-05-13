import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN network with separate value and advantage streams."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.value_stream = nn.Linear(fc2_units, 1)
        self.advantage_stream = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

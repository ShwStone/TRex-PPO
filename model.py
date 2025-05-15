import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary

class DinoActor(nn.Module):
    def __init__(self, input_shape, hidden_dim, num_actions):
        super(DinoActor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[1], out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape[1:])
        o = self.conv1(o)
        o = self.bn1(o)
        
        o = self.conv2(o)
        o = self.bn2(o)

        o = self.conv3(o)
        o = self.bn3(o) 
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.relu(self.bn3(self.conv3(x))) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

class DinoCritic(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super(DinoCritic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[1], out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape[1:])
        o = self.conv1(o)
        o = self.bn1(o)
        
        o = self.conv2(o)
        o = self.bn2(o)

        o = self.conv3(o)
        o = self.bn3(o) 
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.relu(self.bn3(self.conv3(x))) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        values = self.fc2(x)
        return values
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

if __name__ == '__main__':
    input_shape = (1, 4, 175, 500)
    summary(DinoActor(input_shape, 128, 3))
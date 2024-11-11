from torch import nn
from torch import cat, Tensor

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim_width, n_actions):
        super().__init__()
        self.obs_dim = obs_dim
        
        self.layers = nn.Sequential(*[
            nn.Linear(1*3*3+4, hidden_dim_width), nn.ReLU(), # Calculate the input dimension : Here it is 20
            nn.Linear(hidden_dim_width, hidden_dim_width), nn.ReLU(),
            nn.Linear(hidden_dim_width, n_actions),
        ])

    def forward(self, obs):
        # cnnOut = self.cnnlayer(obs.unsqueeze(-3))
        # cnnOut = cnnOut.view(-1, 64*6*6)
        # cnnOut = cat((cnnOut, Tensor(goals), Tensor(states)), dim=1)
        return self.layers(obs)

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

class CriticNetwork(nn.Module):
    def __init__(self, all_obs_dims, all_acts_dims, hidden_dim_width):
        super().__init__()
        input_size = sum(all_obs_dims) + sum(all_acts_dims)

        self.layers = nn.Sequential(*[
            nn.Linear(11565, hidden_dim_width),
            nn.ReLU(),
            nn.Linear(hidden_dim_width, hidden_dim_width),
            nn.ReLU(),
            nn.Linear(hidden_dim_width, 1),
        ])

    def forward(self, obs_and_acts):
        return self.layers(obs_and_acts) # TODO hmmm

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

class CNNHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnnlayer = nn.Sequential(*[
            nn.Conv2d(1, 4, 3, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, 3, stride=3),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        ])
    
    def forward(self, obs):
        out = self.cnnlayer(obs.unsqueeze(-3))
        out = out.view(-1, 1*3*3)
        return out
    




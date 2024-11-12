from torch import nn

class CNNHead2(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_depth = 1
        self.cnnlayer = nn.Sequential(*[
            nn.Conv2d(1, 4, 5, stride=3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5, stride=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 5, stride=3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, 3)
        ])
    
    def forward(self, obs):
        out = self.cnnlayer(obs.unsqueeze(-3))
        out = out.view(-1, 1*9)
        return out
    


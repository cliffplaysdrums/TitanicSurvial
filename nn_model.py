import torch.nn as nn


class TitanicPredictor(nn.Module):
    def __init__(self, num_features, num_targets):
        super().__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=num_features)
        self.out = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x):
        sig = nn.Sigmoid()
        x = sig(self.fc1(x))
        x = sig(self.out(x))
        return x

import torch.nn as nn
import torch

class IntensityNet(nn.Module):
    def __init__(self, demo_dim, history_dim):
        super().__init__()

        self.demo_encoder = nn.Sequential(
            nn.Linear(demo_dim, 8),
            nn.GELU(),
            nn.Dropout(0.1)
        ).cuda()

        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1)
        ).cuda()

        self.intensity_net = nn.Sequential(
            nn.Linear(8+128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Softplus()
        ).cuda()

    def forward(self, demographic_features, history_repr):
        demo = self.demo_encoder(demographic_features)
        hist = self.history_encoder(history_repr)
        combined = torch.cat([demo, hist], dim=1)
        return self.intensity_net(combined)
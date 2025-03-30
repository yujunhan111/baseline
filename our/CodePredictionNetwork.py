import torch.nn as nn
import torch
import torch.nn.functional as F
# # 做code预测的神经网络
class CodePredictionNetwork(nn.Module):
    def __init__(self, demo_dim, hist_dim, num_codes):
        super().__init__()

        self.demographic_projection = nn.Sequential(
            nn.Linear(demo_dim, 16),
            nn.GELU(),
            nn.Dropout(0.1)
        ).cuda()

        self.history_projection = nn.Sequential(
            nn.Linear(hist_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        ).cuda()

        self.classifier = nn.Sequential(
            nn.Linear(16+256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_codes)
        ).cuda()

    def forward(self, demographic_features, history_repr):
        demo = self.demographic_projection(demographic_features)
        hist = self.history_projection(history_repr)
        combined = torch.cat([demo, hist], dim=1)
        return self.classifier(combined)
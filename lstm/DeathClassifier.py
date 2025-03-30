import torch.nn as nn
import torch
class DeathSpecificClassifier(nn.Module):
    def __init__(self, demo_dim, hist_dim):
        super().__init__()

        self.demo_encoder = nn.Sequential(
            nn.Linear(demo_dim, 8),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.hist_encoder = nn.Sequential(
            nn.Linear(hist_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(8 + 128, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, demo_features, hist_repr):
        demo_encoded = self.demo_encoder(demo_features)
        hist_encoded = self.hist_encoder(hist_repr)
        combined = torch.cat([demo_encoded, hist_encoded], dim=1)
        return self.classifier(combined)


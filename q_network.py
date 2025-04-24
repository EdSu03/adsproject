import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, num_zones, num_time_bins, embed_dim=16):
        super().__init__()
        self.zone_embed = nn.Embedding(num_zones, embed_dim)
        self.time_embed = nn.Embedding(num_time_bins, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),  # 适度扩大 hidden layer
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s_zone, t_bin, a_zone):
        s_embed = self.zone_embed(s_zone)
        t_embed = self.time_embed(t_bin)
        a_embed = self.zone_embed(a_zone)
        x = torch.cat([s_embed, t_embed, a_embed], dim=1)
        return self.net(x).squeeze(1)
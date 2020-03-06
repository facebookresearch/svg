# https://github.com/denisyarats/pytorch_sac_ae/blob/master/encoder.py

import torch
from torch import nn

class EncoderCNN(nn.Module):
    def __init__(self, frame_stack, latent_obs_dim):
        super().__init__()

        # TODO: Move some params to config once settled.
        in_num_filters = 3*frame_stack
        num_filters = 32
        self.num_layers = 4

        self.convs = nn.ModuleList()
        num_filters = 3*frame_stack
        for i in range(self.num_layers):
            num_out_filters = num_filters if i < self.num_layers - 1 else 1
            self.convs.append(
                nn.Conv2d(num_filters, num_out_filters, 3, dilation=2, padding=2))
            num_filters = num_out_filters
        out_dim = 64*64
        self.fc = nn.Linear(out_dim, latent_obs_dim)

    def forward(self, x):
        assert x.dim() == 4
        n_batch = x.size(0)
        x = (x / 255.) - 0.5

        h = x
        for i in range(self.num_layers):
            h = torch.relu(self.convs[i](h))

        h = h.view(n_batch, -1)
        h = torch.tanh(self.fc(h))
        return h

# https://github.com/denisyarats/pytorch_sac_ae/blob/master/decoder.py

import torch
from torch import nn

class DecoderCNN(nn.Module):
    def __init__(self, frame_stack, latent_obs_dim):
        super().__init__()

        # (frame_stack*3)x64x64

        # TODO: Move some params to config once settled.
        num_layers = 4
        self.num_layers = num_layers
        self.num_filters = 32
        self.out_dim = 25

        self.fc = nn.Linear(
            latent_obs_dim, self.num_filters * self.out_dim * self.out_dim
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(self.num_filters, self.num_filters, 3, stride=1)
            )

        self.deconvs.append(
            nn.ConvTranspose2d(
                self.num_filters, frame_stack*3, 3, stride=2, output_padding=1
            )
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))

        obs = self.deconvs[-1](deconv)
        return obs

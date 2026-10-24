"""Real-motion encoder variant for the structured foot-contact VAE."""
import torch
from torch import nn

from model import StructuredFootVAE

FLOOR_HEIGHT = -0.0109


class RealVAE(StructuredFootVAE):
    def __init__(self, hidden=64, layers=2, pooling='mean', standardize=False,
                 stats=None, fixed_floor_height=FLOOR_HEIGHT):
        super().__init__(
            context_dim=4, fixed_floor_height=fixed_floor_height, hidden=hidden
        )
        self.pooling = pooling
        self.standardize = standardize

        encoder_layers = []
        channels = 9
        for layer in range(layers):
            dilation = 2 ** layer
            encoder_layers.extend([
                nn.Conv1d(channels, hidden, 5, padding=2 * dilation, dilation=dilation),
                nn.GELU(),
            ])
            channels = hidden
        self.geometry_posterior_network.encoder = nn.Sequential(*encoder_layers)
        head_width = hidden * (2 if pooling == 'meanmax' else 1)
        self.geometry_posterior_network.cont_head = nn.Linear(head_width, 27)
        nn.init.zeros_(self.geometry_posterior_network.cont_head.weight)
        nn.init.zeros_(self.geometry_posterior_network.cont_head.bias)

        mean, sd = stats if stats is not None else (torch.zeros(9), torch.ones(9))
        self.register_buffer('input_mean', mean.float())
        self.register_buffer('input_sd', sd.float())

    def encode(self, u, ankle, toe, lengths):
        inputs = self.encoder_inputs(u, ankle, toe)
        if self.standardize:
            inputs = (inputs - self.input_mean) / self.input_sd

        valid = torch.arange(u.shape[1], device=u.device)[None] < lengths[:, None]
        valid = valid[..., None]
        features = self.geometry_posterior_network.encoder(inputs.transpose(1, 2))
        features = features.transpose(1, 2)
        pooled = (features * valid).sum(1) / lengths[:, None]
        if self.pooling == 'meanmax':
            max_pooled = features.masked_fill(~valid, -1e4).amax(1)
            pooled = torch.cat([pooled, max_pooled], dim=-1)
        return self.geometry_posterior_network.cont_head(pooled)

import itertools

import torch
import torch.nn as nn


## Adapated from conditional-flow-matching to allow for variable depth
## github.com/atong01/conditional-flow-matching/blob/main/torchcfm/models/models.py
class MLP(nn.Module):
    def __init__(self, dim, out_dim=None, w=64, depth=2, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        self.act = nn.SELU()
        if out_dim is None:
            out_dim = dim
        _layers = [nn.Linear(dim + (1 if time_varying else 0), w),
                   self.act]
        for _ in range(depth):
            _layers.append(nn.Linear(w, w))
            _layers.append(self.act)
        _layers.append(nn.Linear(w, out_dim))
        self.net = nn.Sequential(*_layers)

    def forward(self, xt):
        return self.net(xt)


## Adapted from MIOFlow
## https://github.com/KrishnaswamyLab/MIOFlow/blob/main/MIOFlow/models.py
class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder_layers,
        decoder_layers=None,
        activation='Tanh'
    ):
        super().__init__()
        if decoder_layers is None:
            decoder_layers = [*encoder_layers[::-1]]

        encoder_shapes = list(zip(encoder_layers, encoder_layers[1:]))
        decoder_shapes = list(zip(decoder_layers, decoder_layers[1:]))

        encoder_linear = list(map(lambda a: nn.Linear(*a), encoder_shapes))
        decoder_linear = list(map(lambda a: nn.Linear(*a), decoder_shapes))

        encoder_riffle = list(
            itertools.chain(
                *zip(encoder_linear, itertools.repeat(getattr(nn, activation)()))
            )
        )[:-1]
        # encoder = nn.Sequential(*encoder_riffle).to(device)
        encoder = nn.Sequential(*encoder_riffle)

        decoder_riffle = list(
            itertools.chain(
                *zip(decoder_linear, itertools.repeat(getattr(nn, activation)()))
            )
        )[:-1]

        # decoder = nn.Sequential(*decoder_riffle).to(device)
        decoder = nn.Sequential(*decoder_riffle)

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decoder(self.encoder(x))

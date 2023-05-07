#!/usr/bin/env python3

import torch
from torch import nn
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, NUM_TYPES_OF_PROGRESSION_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES
)
from kanachan.training.positional_embedding import PositionalEmbedding


class Encoder(nn.Module):
    def __init__(
            self, num_dimensions: int, num_heads: int, num_layers:int,
            dropout: float=0.1, sparse: bool=False) -> None:
        super(Encoder, self).__init__()

        self.__sparse_embedding = nn.Embedding(
            NUM_TYPES_OF_SPARSE_FEATURES + 1, num_dimensions,
            padding_idx=NUM_TYPES_OF_SPARSE_FEATURES, sparse=sparse)

        self.__positional_embedding = PositionalEmbedding(
            NUM_TYPES_OF_PROGRESSION_FEATURES + 1, num_dimensions,
            padding_idx=NUM_TYPES_OF_PROGRESSION_FEATURES,
            max_length=MAX_LENGTH_OF_PROGRESSION_FEATURES, sparse=sparse)

        encoder_layer = nn.TransformerEncoderLayer(
            num_dimensions, num_heads, dropout=dropout, batch_first=True)
        self.__encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        sparse, numeric, positional = x
        sparse = self.__sparse_embedding(sparse)
        positional = self.__positional_embedding(positional)
        embedding = torch.cat((sparse, numeric, positional), dim=1)
        encode = self.__encoder(embedding)
        return encode

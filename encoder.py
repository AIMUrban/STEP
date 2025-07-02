import torch.nn as nn
from torch.nn import init


class TransEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(TransEncoder, self).__init__()
        input_dim = embedding_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dim_feedforward=input_dim*2,
                                                   nhead=2,
                                                   dropout=0.3)

        encoder_norm = nn.LayerNorm(input_dim)

        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=2,
                                             norm=encoder_norm)

    def forward(self, embedded_out, src_mask):
        out = self.encoder(embedded_out, mask=src_mask)

        return out




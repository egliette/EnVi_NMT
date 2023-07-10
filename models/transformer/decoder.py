import torch
import torch.nn as nn

from models.transformer.self_attention import MultiHeadAttentionLayer
from models.transformer.positionwise_feedforward import PositionwiseFeedforwardLayer


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_src, tgt_mask, src_mask):

        #tgt = [batch size, tgt len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #tgt_mask = [batch size, 1, tgt len, tgt len]
        #src_mask = [batch size, 1, 1, src len]

        #self attention
        _tgt, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)

        #dropout, residual connection and layer norm
        tgt = self.self_attn_layer_norm(tgt + self.dropout(_tgt))

        #tgt = [batch size, tgt len, hid dim]

        #encoder attention
        _tgt, attention = self.encoder_attention(tgt, enc_src, enc_src, src_mask)

        #dropout, residual connection and layer norm
        tgt = self.enc_attn_layer_norm(tgt + self.dropout(_tgt))

        #tgt = [batch size, tgt len, hid dim]

        #positionwise feedforward
        _tgt = self.positionwise_feedforward(tgt)

        #dropout, residual and layer norm
        tgt = self.ff_layer_norm(tgt + self.dropout(_tgt))

        #tgt = [batch size, tgt len, hid dim]
        #attention = [batch size, n heads, tgt len, src len]

        return tgt, attention
    

class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, tgt, enc_src, tgt_mask, src_mask):

        #tgt = [batch size, tgt len]
        #enc_src = [batch size, src len, hid dim]
        #tgt_mask = [batch size, 1, tgt len, tgt len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]

        pos = torch.arange(0, tgt_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, tgt len]

        tgt = self.dropout((self.tok_embedding(tgt) * self.scale) + self.pos_embedding(pos))

        #tgt = [batch size, tgt len, hid dim]

        for layer in self.layers:
            tgt, attention = layer(tgt, enc_src, tgt_mask, src_mask)

        #tgt = [batch size, tgt len, hid dim]
        #attention = [batch size, n heads, tgt len, src len]

        output = self.fc_out(tgt)

        #output = [batch size, tgt len, output dim]

        return output, attention
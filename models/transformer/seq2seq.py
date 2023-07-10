import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 tgt_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_tgt_mask(self, tgt):

        #tgt = [batch size, tgt len]

        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        #tgt_pad_mask = [batch size, 1, 1, tgt len]

        tgt_len = tgt.shape[1]

        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device = self.device)).bool()

        #tgt_sub_mask = [tgt len, tgt len]

        tgt_mask = tgt_pad_mask & tgt_sub_mask

        #tgt_mask = [batch size, 1, tgt len, tgt len]

        return tgt_mask

    def forward(self, src, tgt):

        #src = [batch size, src len]
        #tgt = [batch size, tgt len]

        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        #src_mask = [batch size, 1, 1, src len]
        #tgt_mask = [batch size, 1, tgt len, tgt len]

        enc_src = self.encoder(src, src_mask)

        #enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(tgt, enc_src, tgt_mask, src_mask)

        #output = [batch size, tgt len, output dim]
        #attention = [batch size, n heads, tgt len, src len]

        return output, attention
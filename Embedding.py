# PADformer source; data and trained artifacts are not included.

import torch
import torch.nn as nn

class MovingAvg(nn.Module):

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x_padded = torch.cat([front, x, end], dim=-1)
        return self.avg(x_padded)

class FrequencyTimeEmbed(nn.Module):

    def __init__(self, input_len, num_nodes, d_model=144):
        super(FrequencyTimeEmbed, self).__init__()
        self.half_dim = d_model // 2
        self.node_emb = nn.Parameter(torch.zeros(1, num_nodes, input_len))
        nn.init.xavier_uniform_(self.node_emb)
        self.emb_proj = nn.Linear(input_len, self.half_dim)
        self.raw_proj = nn.Linear(input_len, self.half_dim)
        self.freq_len = input_len // 2 + 1
        self.freq_input_dim = self.freq_len * 2
        self.freq_proj = nn.Linear(self.freq_input_dim, self.half_dim)
        self.decomp = MovingAvg(kernel_size=25, stride=1)
        self.trend_proj = nn.Linear(input_len, self.half_dim)

    def forward(self, x, mode='freq_trend'):
        batch_size = x.shape[0]
        node_emb_expanded = self.node_emb.expand(batch_size, -1, -1)
        emb_feat = self.emb_proj(node_emb_expanded)
        if mode == 'raw':
            x_raw_data = self.raw_proj(x)
            x_raw = torch.cat([x_raw_data, emb_feat], dim=-1)
            return (x_raw, x_raw)
        x_freq_complex = torch.fft.rfft(x, dim=-1)
        x_freq_raw = torch.cat([x_freq_complex.real, x_freq_complex.imag], dim=-1)
        x_freq_data = self.freq_proj(x_freq_raw)
        x_freq = torch.cat([x_freq_data, emb_feat], dim=-1)
        x_trend_raw = self.decomp(x)
        x_trend_data = self.trend_proj(x_trend_raw)
        x_trend = torch.cat([x_trend_data, emb_feat], dim=-1)
        return (x_freq, x_trend)

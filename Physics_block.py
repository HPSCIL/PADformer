# PADformer source; data and trained artifacts are not included.

import math
import torch
from torch import nn
import torch.nn.functional as F

class Physics_TVA_block(nn.Module):

    def __init__(self, Input_len, output_len, num_id, num_layer, dropout, num_head, num_samp, num_nodes, var_dim, lat_lon=None, num_terrain_channels=3, constrain_physics_sign=True):
        super(Physics_TVA_block, self).__init__()
        self.num_lay = num_layer
        self.Time_att = Time_att(Input_len, dropout, num_head)
        self.space_att = Adaptive_Physics_Space_Att(num_nodes, var_dim, dropout, num_head, lat_lon, num_terrain_channels, constrain_physics_sign)
        self.cross_att = cross_att(Input_len, dropout, num_head)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(Input_len, output_len)

    def forward(self, x, terrain_3ch=None, wind_batch=None):
        if x.dim() == 4:
            B, N, T, C = x.shape
            x = x.reshape(B, N, T * C)
        for _ in range(self.num_lay):
            x_time = self.Time_att(x)
            x_space = self.space_att(x, terrain_3ch, wind_batch)
            x = self.cross_att(x_time, x_space)
        return self.projection(x)

class Adaptive_Physics_Space_Att(nn.Module):

    def __init__(self, num_nodes, var_dim, dropout, num_head, lat_lon=None, num_terrain_channels=3, constrain_physics_sign=True):
        super(Adaptive_Physics_Space_Att, self).__init__()
        self.num_nodes = num_nodes
        self.var_dim = var_dim
        self.num_head = num_head
        self.num_terrain_channels = num_terrain_channels
        self.constrain_physics_sign = constrain_physics_sign
        self.query = nn.Linear(var_dim, var_dim)
        self.key = nn.Linear(var_dim, var_dim)
        self.value = nn.Linear(var_dim, var_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.terrain_weights = nn.Parameter(torch.full((num_terrain_channels,), 1.0 / num_terrain_channels))
        self.terrain_bias_scale = nn.Parameter(torch.tensor(0.0))
        wind_gate_init = -3.0 if constrain_physics_sign else 0.0
        self.wind_gate = nn.Parameter(torch.tensor(wind_gate_init))
        self.decay_power = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.proj_scale = nn.Parameter(torch.tensor(math.log(math.exp(1.0) - 1.0)))
        if lat_lon is not None:
            if not isinstance(lat_lon, torch.Tensor):
                lat_lon = torch.tensor(lat_lon, dtype=torch.float32)
            self._register_geo_buffers(lat_lon)
            self.has_physics = True
        else:
            self.has_physics = False

    def _register_geo_buffers(self, lat_lon):
        lat = lat_lon[:, 0].view(1, -1)
        lon = lat_lon[:, 1].view(1, -1)
        lat_i = lat.unsqueeze(2)
        lat_j = lat.unsqueeze(1)
        lon_i = lon.unsqueeze(2)
        lon_j = lon.unsqueeze(1)
        mean_lat_rad = (lat_i + lat_j) * 0.5 * math.pi / 180.0
        dx = (lon_i - lon_j) * torch.cos(mean_lat_rad) * 111.0
        dy = (lat_i - lat_j) * 111.0
        dist = torch.sqrt(dx ** 2 + dy ** 2) + 1e-06
        self.register_buffer('geo_dist', dist)
        self.register_buffer('geo_dir_u', dx / dist)
        self.register_buffer('geo_dir_v', dy / dist)

    def forward(self, x, terrain_3ch=None, wind_batch=None):
        if x.dim() == 4:
            B, N, T, C = x.shape
            x = x.reshape(B, N, T * C)
        B, N_all, T = x.shape
        x_reshaped = x.permute(0, 2, 1).reshape(B, T, self.num_nodes, self.var_dim)
        q = self.query(x_reshaped)
        k = self.key(x_reshaped)
        v = self.value(x_reshaped)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.var_dim)
        if terrain_3ch is not None:
            weights = F.softmax(self.terrain_weights, dim=0).view(-1, 1, 1)
            adaptive_terrain_bias = (weights * terrain_3ch).sum(dim=0)
            if self.constrain_physics_sign:
                terrain_scale = -F.softplus(self.terrain_bias_scale)
            else:
                terrain_scale = self.terrain_bias_scale
            scores = scores + terrain_scale * adaptive_terrain_bias
        if wind_batch is not None and self.has_physics:
            wind_feat = wind_batch.permute(0, 1, 3, 2)
            wind_mean = torch.mean(wind_feat, dim=2)
            wind_u = wind_mean[..., 0]
            wind_v = wind_mean[..., 1]
            proj = wind_u.unsqueeze(1) * self.geo_dir_u + wind_v.unsqueeze(1) * self.geo_dir_v
            proj_scale = F.softplus(self.proj_scale) + 1e-06
            wind_effect = F.relu(proj * proj_scale)
            decay_power = F.softplus(self.decay_power) + 1e-06
            dist_decay = 1.0 / (self.geo_dist.pow(decay_power) + 1e-06)
            wind_bias = wind_effect * dist_decay
            wind_bias = torch.tanh(wind_bias / 10.0) * 5.0
            eye = torch.eye(self.num_nodes, device=wind_bias.device, dtype=torch.bool).unsqueeze(0)
            wind_bias = wind_bias.masked_fill(eye, 0.0)
            if self.constrain_physics_sign:
                wind_gate = torch.sigmoid(self.wind_gate)
            else:
                wind_gate = self.wind_gate
            scores = scores + wind_gate * wind_bias.unsqueeze(1)
        attn = self.softmax(scores)
        out = torch.matmul(self.dropout(attn), v)
        out = out.reshape(B, T, N_all)
        return out.transpose(1, 2)

class Time_att(nn.Module):

    def __init__(self, dim_input, dropout, num_head):
        super(Time_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x):
        x = x.unsqueeze(-1).transpose(1, 2)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x)).squeeze(-1).transpose(1, 2)
            k = self.dropout(self.key(x)).squeeze(-1).transpose(1, 2)
            v = self.dropout(self.value(x)).squeeze(-1).transpose(1, 2)
            kd = torch.sqrt(torch.tensor(k.shape[-1], device=k.device, dtype=torch.float32) / self.num_head)
            line = torch.matmul(self.dropout(self.softmax(torch.matmul(q, k.transpose(-2, -1)) / kd)), v)
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result).squeeze(-1)
        x_orig = x.squeeze(-1).transpose(1, 2)
        x = x_orig + result
        return self.laynorm(x)

class cross_att(nn.Module):

    def __init__(self, dim_input, dropout, num_head):
        super(cross_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x, x2):
        x = x.unsqueeze(-1).transpose(1, 2)
        x2 = x2.unsqueeze(-1).transpose(1, 2)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x2)).squeeze(-1).transpose(1, 2)
            k = self.dropout(self.key(x)).squeeze(-1).transpose(1, 2)
            v = self.dropout(self.value(x)).squeeze(-1).transpose(1, 2)
            kd = torch.sqrt(torch.tensor(k.shape[-1], device=k.device, dtype=torch.float32) / self.num_head)
            line = torch.matmul(self.dropout(self.softmax(torch.matmul(q, k.transpose(-2, -1)) / kd)), v)
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result).squeeze(-1)
        x_orig = x.squeeze(-1).transpose(1, 2)
        x = x_orig + result
        return self.laynorm(x)

# PADformer source; data and trained artifacts are not included.

import torch
from torch import nn
from Embedding import FrequencyTimeEmbed
from Physics_block import Physics_TVA_block
from Decoder import TVADE_block
from RevIN import RevIN

def canonical_ablation(ablation):
    aliases = {None: 'none', 'full': 'none', 'baseline': 'none', 'full_v2': 'none_v2', 'full_free_scale_gate': 'none_free_scale_gate', 'no_terrain': 'no_terrain_freq', 'no_wind': 'no_wind_time'}
    return aliases.get(ablation, ablation)

class PADformer(nn.Module):

    def __init__(self, Input_len, out_len, num_id, num_layer, dropout, muti_head, num_samp, IF_node, num_nodes=1, var_dim=4, terrain_3ch=None, lat_lon=None, num_terrain_channels=3, ablation='none', constrain_physics_sign=True):
        super(PADformer, self).__init__()
        self.ablation = canonical_ablation(ablation)
        self.constrain_physics_sign = constrain_physics_sign
        self.num_nodes = num_nodes
        self.var_dim = var_dim
        self.num_terrain_channels = num_terrain_channels
        self.enc_in_dim = 144
        self.enc_out_dim = 144
        self.inputlen = self.enc_out_dim
        self.RevIN = RevIN(num_id)
        self.embed_layer = FrequencyTimeEmbed(Input_len, num_id, d_model=self.enc_in_dim)
        self.encoder_terrain = Physics_TVA_block(Input_len=self.enc_in_dim, output_len=self.enc_out_dim, num_id=num_id, num_layer=num_layer, dropout=dropout, num_head=muti_head, num_samp=num_samp, num_nodes=num_nodes, var_dim=var_dim, lat_lon=None, num_terrain_channels=num_terrain_channels, constrain_physics_sign=constrain_physics_sign)
        self.encoder_wind = Physics_TVA_block(Input_len=self.enc_in_dim, output_len=self.enc_out_dim, num_id=num_id, num_layer=num_layer, dropout=dropout, num_head=muti_head, num_samp=num_samp, num_nodes=num_nodes, var_dim=var_dim, lat_lon=lat_lon, num_terrain_channels=num_terrain_channels, constrain_physics_sign=constrain_physics_sign)
        self.laynorm = nn.LayerNorm([self.inputlen])
        self.decoder = TVADE_block(self.inputlen, num_id, dropout, muti_head)
        self.output = nn.Conv1d(in_channels=self.inputlen, out_channels=out_len, kernel_size=1)
        if terrain_3ch is not None:
            self.register_buffer('terrain_3ch', terrain_3ch)
        else:
            self.register_buffer('terrain_3ch', torch.zeros(num_terrain_channels, num_nodes, num_nodes))
        self.arbitrator = nn.Sequential(nn.Linear(self.enc_out_dim * 2, self.enc_out_dim // 2), nn.ReLU(), nn.Linear(self.enc_out_dim // 2, 1), nn.Sigmoid())
        if self.ablation == 'no_arbitration':
            self.concat_fusion_proj = nn.Linear(self.enc_out_dim * 2, self.enc_out_dim)
        if self.ablation == 'none_v2':
            self.terrain_fusion_norm = nn.LayerNorm(self.enc_out_dim)
            self.wind_fusion_norm = nn.LayerNorm(self.enc_out_dim)
            self.channel_arbitrator = nn.Sequential(nn.Linear(self.enc_out_dim * 2, self.enc_out_dim), nn.ReLU(), nn.Linear(self.enc_out_dim, self.enc_out_dim), nn.Sigmoid())

    def _branch_flags(self):
        use_terrain_branch = self.ablation != 'no_terrain_freq'
        use_wind_branch = self.ablation not in {'no_wind_time', 'no_freq_time_embed'}
        use_terrain_physics = self.ablation not in {'no_terrain_freq', 'no_all'}
        use_wind_physics = self.ablation not in {'no_wind_time', 'no_freq_time_embed', 'no_all'}
        return (use_terrain_branch, use_wind_branch, use_terrain_physics, use_wind_physics)

    def _route_features(self, x_freq, x_time):
        if self.ablation == 'swap_freq_time':
            return (x_time, x_freq)
        return (x_freq, x_time)

    def _fuse(self, x_terrain, x_wind):
        if x_terrain is None:
            return x_wind
        if x_wind is None:
            return x_terrain
        if self.ablation == 'none_v2':
            x_terrain = self.terrain_fusion_norm(x_terrain)
            x_wind = self.wind_fusion_norm(x_wind)
            combined = torch.cat([x_terrain, x_wind], dim=-1)
            gate = self.channel_arbitrator(combined)
            return gate * x_terrain + (1 - gate) * x_wind
        combined = torch.cat([x_terrain, x_wind], dim=-1)
        if self.ablation == 'no_arbitration':
            return self.concat_fusion_proj(combined)
        if self.ablation in {'mean_fusion', 'no_all'}:
            return 0.5 * (x_terrain + x_wind)
        gate = self.arbitrator(combined)
        return gate * x_terrain + (1 - gate) * x_wind

    def forward(self, x, wind_batch=None):
        x = self.RevIN(x, 'norm')
        x = x.transpose(1, 2)
        embed_mode = 'raw' if self.ablation == 'no_all' else 'freq_trend'
        x_freq, x_time = self.embed_layer(x, mode=embed_mode)
        terrain_feat, wind_feat = self._route_features(x_freq, x_time)
        use_terrain_branch, use_wind_branch, use_terrain_physics, use_wind_physics = self._branch_flags()
        x_terrain = None
        if use_terrain_branch:
            terrain_input = self.terrain_3ch if use_terrain_physics else None
            x_terrain = self.encoder_terrain(terrain_feat, terrain_3ch=terrain_input, wind_batch=None)
        x_wind = None
        if use_wind_branch:
            wind_input = wind_batch if use_wind_physics else None
            x_wind = self.encoder_wind(wind_feat, terrain_3ch=None, wind_batch=wind_input)
        x = self._fuse(x_terrain, x_wind)
        x = self.laynorm(x)
        x = self.decoder(x)
        x = self.output(x.transpose(-2, -1))
        x = self.RevIN(x, 'denorm')
        return x

    def active_graph(self):
        use_terrain_branch, use_wind_branch, use_terrain_physics, use_wind_physics = self._branch_flags()
        return {'ablation': self.ablation, 'embedding': 'raw_shared' if self.ablation == 'no_all' else 'frequency+trend', 'terrain_branch': use_terrain_branch, 'terrain_physics': use_terrain_physics and use_terrain_branch, 'wind_branch': use_wind_branch, 'wind_physics': use_wind_physics and use_wind_branch, 'fusion': 'single_branch' if not (use_terrain_branch and use_wind_branch) else 'branch_norm_channel_gate' if self.ablation == 'none_v2' else 'concat_projection' if self.ablation == 'no_arbitration' else 'fixed_mean' if self.ablation in {'mean_fusion', 'no_all'} else 'learned_arbitration', 'routing': 'swapped' if self.ablation == 'swap_freq_time' else 'standard', 'physics_sign': 'constrained' if self.constrain_physics_sign else 'unconstrained'}

"""PADformer: training, evaluation and clean ablations (no dataset included).

Dependencies: torch, numpy, pandas, openpyxl, scikit-learn.
Example (replace paths and station count with your own inputs):
    python main.py --num_nodes 10 --csv_path data/air_quality.csv \
        --station_path data/stations.xlsx --terrain_path data/terrain.npz \
        --ablation none_v2 --seeds 50 --num_workers 0

CSV columns: PM2.5, PM10, O3, NO2, u10(m/s), v10(m/s).
Rows: contiguous time series per station by default; use --csv_layout
time_block for contiguous station groups per time step. All stations must
share aligned timestamps and the same order across every input file.
Station XLSX: latitude/longitude (also accepts lat/lon or Chinese headers).
Terrain NPZ: five [N,N] arrays named relief, climb, slope, tpi, svf;
arr_0 through arr_4 are also supported. SVF is negated before normalization.
Default terrain_channels=3 selects slope, tpi and svf from these five arrays.
Only historical wind is used. Missing terrain/stations retain the original
zero-bias fallback; supply both for the full physical model.

Time steps are split chronologically into 60/20/20 train/validation/test
segments BEFORE windows are generated independently inside each segment.
Windows never cross split boundaries. Input rows must already be in time
order. Only training batches are shuffled. MAE loss, Adam, early stopping,
best checkpoint and positive-target metrics are retained.
The original default experiment is none; explicitly select none_v2 for
branch normalization and channel-wise arbitration. Evaluation uses 24/48/72h
horizons and therefore requires pre_len >= 72. No original observations,
coordinates, matrices, weights, logs or experiment results are distributed.
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import copy
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_preprocess import pro_data, MyDataset
from utils import get_logger, All_Metrics, init_seed
from model import PADformer
MODEL_NAME = 'PADformer'
DATASET_NAME = 'custom_dataset'
POLLUTANTS = ['PM2.5', 'PM10', 'O3', 'NO2']
SINGLE_INDICES = [23, 47, 71]
SINGLE_NAMES = ['24h', '48h', '72h']
WINDOW_LENGTHS = [24, 48, 72]
WINDOW_NAMES = ['24h', '48h', '72h']
NEW_FIVE_ABLATIONS = ['no_all', 'no_freq_time_embed_v2', 'no_terrain_physics', 'no_wind_physics', 'no_arbitration_v2']
CLEAN_V2_ABLATIONS = set(NEW_FIVE_ABLATIONS) - {'no_all'}
PAIRING_ABLATIONS = ['none_v2', 'swap_freq_time_v2', 'both_freq_v2', 'both_trend_v2']
PAIRING_ROUTING = {'none_v2': 'standard', 'swap_freq_time_v2': 'swapped', 'both_freq_v2': 'frequency_frequency', 'both_trend_v2': 'trend_trend'}
CLEAN_V2_ABLATIONS.update(PAIRING_ABLATIONS[1:])
CANONICAL_ABLATIONS = ['none', 'none_free_scale_gate', 'no_freq_time_embed', 'no_terrain_freq', 'no_wind_time', 'no_arbitration', 'swap_freq_time']
ABLATION_CHOICES = CANONICAL_ABLATIONS + ['none_v2', 'full_v2', 'no_all', 'no_terrain', 'no_wind', 'terrain_only', 'wind_only', 'raw_dual_stream', 'mean_fusion', 'unconstrained_physics', 'swapped_streams', 'no_freq_time_embed_v2', 'no_terrain_physics', 'no_wind_physics', 'no_arbitration_v2', 'swap_freq_time_v2', 'both_freq_v2', 'both_trend_v2']
ABLATION_ALIAS = {'full_v2': 'none_v2', 'no_terrain': 'no_terrain_freq', 'no_wind': 'no_wind_time', 'terrain_only': 'no_wind_time', 'wind_only': 'no_terrain_freq', 'raw_dual_stream': 'no_all', 'unconstrained_physics': 'none_free_scale_gate', 'swapped_streams': 'swap_freq_time'}
ABLATION_DESCRIPTIONS = {'swap_freq_time_v2': 'Full_v2 unchanged modules: trend to terrain, frequency to wind.', 'both_freq_v2': 'Full_v2 unchanged modules: frequency to both physical branches.', 'both_trend_v2': 'Full_v2 unchanged modules: trend to both physical branches.', 'none': 'Full: frequency+terrain branch, trend+wind branch, learned arbitration.', 'none_v2': 'Full_v2: full physics branches with branch LayerNorm and channel-wise arbitration gate.', 'none_free_scale_gate': 'Full structure with unconstrained terrain scale and wind gate.', 'no_wind_time': 'Terrain only: frequency+terrain branch; wind branch and arbitration bypassed.', 'no_terrain_freq': 'Wind only: trend+wind branch; terrain branch and arbitration bypassed.', 'no_arbitration': 'Both full physics branches with concatenation followed by a learned 288-to-144 projection; no arbitration gate.', 'mean_fusion': 'Both full physics branches with fixed 0.5/0.5 mean fusion.', 'no_freq_time_embed': 'Frequency+terrain branch only; trend+wind branch and arbitration bypassed.', 'swap_freq_time': 'Both full branches, but trend goes to terrain and frequency goes to wind.', 'no_all': 'Raw dual stream without FFT/trend split, physics biases, or learned arbitration; fixed 0.5/0.5 mean fusion.', 'no_freq_time_embed_v2': 'Full_v2 with raw shared embedding; physics branches, branch LayerNorm, and channel arbitration are retained.', 'no_terrain_physics': 'Full_v2 with only the terrain attention bias disabled; the frequency terrain branch is retained.', 'no_wind_physics': 'Full_v2 with only the dynamic wind attention bias disabled; the trend wind branch is retained.', 'no_arbitration_v2': 'Full_v2 with branch LayerNorm retained and channel arbitration replaced by fixed 0.5/0.5 fusion.'}

def canonical_ablation(ablation):
    return ABLATION_ALIAS.get(ablation, ablation)

class PADformerCleanAblation(PADformer):

    def __init__(self, *args, ablation, **kwargs):
        if ablation not in CLEAN_V2_ABLATIONS:
            raise ValueError(f'Unsupported clean v2 ablation: {ablation}')
        requested_ablation = ablation
        super().__init__(*args, ablation='none_v2', **kwargs)
        self.ablation = requested_ablation

    def _route_features(self, x_freq, x_time):
        if self.ablation == 'swap_freq_time_v2':
            return (x_time, x_freq)
        if self.ablation == 'both_freq_v2':
            return (x_freq, x_freq)
        if self.ablation == 'both_trend_v2':
            return (x_time, x_time)
        return super()._route_features(x_freq, x_time)

    def _branch_flags(self):
        use_terrain_physics = self.ablation != 'no_terrain_physics'
        use_wind_physics = self.ablation != 'no_wind_physics'
        return (True, True, use_terrain_physics, use_wind_physics)

    def _fuse(self, x_terrain, x_wind):
        x_terrain = self.terrain_fusion_norm(x_terrain)
        x_wind = self.wind_fusion_norm(x_wind)
        if self.ablation == 'no_arbitration_v2':
            return 0.5 * (x_terrain + x_wind)
        combined = torch.cat([x_terrain, x_wind], dim=-1)
        gate = self.channel_arbitrator(combined)
        return gate * x_terrain + (1 - gate) * x_wind

    def forward(self, x, wind_batch=None):
        x = self.RevIN(x, 'norm')
        x = x.transpose(1, 2)
        embed_mode = 'raw' if self.ablation == 'no_freq_time_embed_v2' else 'freq_trend'
        x_freq, x_time = self.embed_layer(x, mode=embed_mode)
        terrain_feat, wind_feat = self._route_features(x_freq, x_time)
        _, _, use_terrain_physics, use_wind_physics = self._branch_flags()
        terrain_input = self.terrain_3ch if use_terrain_physics else None
        wind_input = wind_batch if use_wind_physics else None
        x_terrain = self.encoder_terrain(terrain_feat, terrain_3ch=terrain_input, wind_batch=None)
        x_wind = self.encoder_wind(wind_feat, terrain_3ch=None, wind_batch=wind_input)
        x = self._fuse(x_terrain, x_wind)
        x = self.laynorm(x)
        x = self.decoder(x)
        x = self.output(x.transpose(-2, -1))
        return self.RevIN(x, 'denorm')

    def active_graph(self):
        _, _, use_terrain_physics, use_wind_physics = self._branch_flags()
        return {'ablation': self.ablation, 'embedding': 'raw_shared' if self.ablation == 'no_freq_time_embed_v2' else 'frequency+trend', 'terrain_branch': True, 'terrain_physics': use_terrain_physics, 'wind_branch': True, 'wind_physics': use_wind_physics, 'fusion': 'branch_norm_fixed_mean' if self.ablation == 'no_arbitration_v2' else 'branch_norm_channel_gate', 'routing': PAIRING_ROUTING.get(self.ablation, 'standard'), 'physics_sign': 'constrained' if self.constrain_physics_sign else 'unconstrained'}

def parse_args():
    parser = argparse.ArgumentParser(description='PADformer clean five-ablation runner')
    parser.add_argument('--stage', type=str, default='tune_full', choices=['tune_full', 'ablation_suite', 'clean_five', 'pairing_suite'], help='clean_five runs no_all and four single-factor none_v2 ablations.')
    parser.add_argument('--ablation', type=str, default=None, choices=ABLATION_CHOICES, help='Run only one ablation. It overrides --ablations.')
    parser.add_argument('--ablations', nargs='+', default=None, choices=ABLATION_CHOICES, help='Run selected experiments. It overrides --stage.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[50], help='Random seeds.')
    parser.add_argument('--run_tag', type=str, default=None, help='Optional result tag. Defaults to a tag built from hyperparameters.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--var_dim', type=int, default=4)
    parser.add_argument('--lag', type=int, default=72)
    parser.add_argument('--pre_len', type=int, default=72)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--muti_head', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--num_samp', type=int, default=2)
    parser.add_argument('--IF_node', type=eval, default=True)
    parser.add_argument('--terrain_channels', type=int, default=3, choices=[3, 5], help='3=Slope/TPI/SVF; 5=Relief/Climb/Slope/TPI/SVF.')
    parser.add_argument('--csv_path', type=str, default='data/air_quality.csv')
    parser.add_argument('--csv_layout', type=str, default='station_block', choices=['station_block', 'time_block'])
    parser.add_argument('--terrain_path', type=str, default='data/terrain.npz')
    parser.add_argument('--station_path', type=str, default='data/stations.xlsx')
    parser.add_argument('--result_root', type=str, default='./results')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--debug', type=eval, default=False)
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--strict_deterministic', type=eval, default=True, help='True enables slower strict CUDA determinism for final reproducibility checks.')
    parser.add_argument('--early_stop', type=eval, default=True)
    parser.add_argument('--early_stop_patience', type=int, default=30)
    parser.add_argument('--min_delta', type=float, default=0.005, help='Minimum validation-loss improvement required to reset early stopping.')
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    args.num_id = args.num_nodes * args.var_dim
    if args.ablation is not None:
        args.ablations = [args.ablation]
    elif args.ablations is None:
        if args.stage == 'tune_full':
            args.ablations = ['none']
        elif args.stage == 'clean_five':
            args.ablations = NEW_FIVE_ABLATIONS
        elif args.stage == 'pairing_suite':
            args.ablations = PAIRING_ABLATIONS
        else:
            args.ablations = CANONICAL_ABLATIONS
    args.ablations = [canonical_ablation(x) for x in args.ablations]
    if len(set(args.ablations)) != len(args.ablations):
        raise ValueError(f'Duplicate experiments after alias conversion: {args.ablations}')
    if args.run_tag is None:
        lr_tag = f'{args.learning_rate:.0e}'.replace('+', '')
        args.run_tag = f'lr{lr_tag}_drop{args.dropout:g}_pat{args.early_stop_patience}_md{args.min_delta:g}_tc{args.terrain_channels}_det{int(args.strict_deterministic)}'
    args.run_tag = ''.join((char if char.isalnum() or char in {'-', '_', '.'} else '_' for char in args.run_tag))
    return args

def get_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        return torch.device(f'cuda:{args.gpu_id}')
    return torch.device('cpu')

def terrain_channel_names(num_channels):
    if num_channels == 5:
        return ['ReliefRes', 'ClimbRes', 'SlopeRes', 'TPIRes', 'SVFRes']
    return ['SlopeRes', 'TPIRes', 'SVFRes']

def load_sliced_terrain_matrix(file_path, num_nodes, device, num_channels=3):
    if not os.path.exists(file_path):
        print(f'Warning: terrain file not found: {file_path}. Model will use zeros.')
        return None
    try:
        with np.load(file_path) as data:
            keys_map = {'relief': ['relief', 'arr_0'], 'climb': ['climb', 'arr_1'], 'slope': ['slope', 'arr_2'], 'tpi': ['tpi', 'arr_3'], 'svf': ['svf', 'arr_4']}

            def get_m(k_type):
                for key in keys_map[k_type]:
                    for file_key in data.keys():
                        if key in file_key.lower():
                            return data[file_key]
                return None
            m_list = [get_m(k) for k in ['relief', 'climb', 'slope', 'tpi', 'svf']]
            if any((x is None for x in m_list)):
                sorted_keys = sorted(data.keys())
                raw_data = np.stack([data[sorted_keys[i]] for i in range(5)], axis=0)
            else:
                raw_data = np.stack(m_list, axis=0)
            if num_channels == 5:
                sliced_data = raw_data[[0, 1, 2, 3, 4], :, :].astype(np.float32)
                sliced_data[4] = -sliced_data[4]
            else:
                sliced_data = raw_data[[2, 3, 4], :, :].astype(np.float32)
                sliced_data[2] = -sliced_data[2]
        channel_names = terrain_channel_names(num_channels)
        print(f">>> Terrain resistance channels: {', '.join(channel_names)}")
        for i in range(num_channels):
            mean_val = sliced_data[i].mean()
            std_val = sliced_data[i].std()
            if std_val < 1e-06:
                std_val = 1.0
            sliced_data[i] = (sliced_data[i] - mean_val) / std_val
            if sliced_data[i].ndim == 2 and sliced_data[i].shape[0] == sliced_data[i].shape[1]:
                np.fill_diagonal(sliced_data[i], 0.0)
            print(f'   - {channel_names[i]} normalized.')
        if sliced_data.shape[-2:] != (num_nodes, num_nodes):
            raise ValueError(f'Terrain shape {sliced_data.shape} does not match num_nodes={num_nodes}')
        return torch.FloatTensor(sliced_data).to(device)
    except Exception as exc:
        print(f'Terrain loading failed: {exc}. Model will use zeros.')
        return None

def load_raw_arrays(args, device):
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f'CSV not found: {args.csv_path}')
    dataframe = pd.read_csv(args.csv_path)
    if len(dataframe) % args.num_nodes != 0:
        raise ValueError(f'CSV rows {len(dataframe)} cannot be divided by num_nodes={args.num_nodes}')
    data_pollutant_raw = dataframe[POLLUTANTS].values.astype('float32')
    data_wind_raw = dataframe[['u10(m/s)', 'v10(m/s)']].values.astype('float32')
    if args.csv_layout == 'station_block':
        dataset_pollutant = data_pollutant_raw.reshape(args.num_nodes, -1, args.var_dim).transpose(1, 0, 2)
        dataset_wind = data_wind_raw.reshape(args.num_nodes, -1, 2).transpose(1, 0, 2)
    else:
        dataset_pollutant = data_pollutant_raw.reshape(-1, args.num_nodes, args.var_dim)
        dataset_wind = data_wind_raw.reshape(-1, args.num_nodes, 2)
    if os.path.exists(args.station_path):
        df_station = pd.read_excel(args.station_path)
        if {'纬度', '经度'}.issubset(df_station.columns):
            lat_lon_arr = df_station[['纬度', '经度']].iloc[:args.num_nodes].values.astype('float32')
        elif {'latitude', 'longitude'}.issubset(df_station.columns):
            lat_lon_arr = df_station[['latitude', 'longitude']].iloc[:args.num_nodes].values.astype('float32')
        elif {'lat', 'lon'}.issubset(df_station.columns):
            lat_lon_arr = df_station[['lat', 'lon']].iloc[:args.num_nodes].values.astype('float32')
        else:
            raise KeyError(f'Cannot find latitude/longitude columns in {args.station_path}')
        lat_lon_tensor = torch.tensor(lat_lon_arr, dtype=torch.float32).to(device)
    else:
        print(f'Warning: station file not found: {args.station_path}. Wind projection will be disabled.')
        lat_lon_tensor = torch.zeros((args.num_nodes, 2), dtype=torch.float32).to(device)
    print(f'Pollutant shape: {dataset_pollutant.shape} | Wind shape: {dataset_wind.shape}')
    return (dataset_pollutant, dataset_wind, lat_lon_tensor)

def build_dataloaders(args, dataset_pollutant, dataset_wind, seed):
    if len(dataset_pollutant) != len(dataset_wind):
        raise ValueError('Pollutant and wind arrays must have the same time length.')
    if args.lag <= 0 or args.pre_len <= 0:
        raise ValueError('lag and pre_len must be positive.')

    total = len(dataset_pollutant)
    train_end = total * 6 // 10
    val_end = train_end + total * 2 // 10
    segments = [('train', 0, train_end),
                ('validation', train_end, val_end),
                ('test', val_end, total)]
    required = args.lag + args.pre_len
    for name, start, end in segments:
        if end - start < required:
            raise ValueError(
                f'{name} segment has {end - start} time steps; at least '
                f'{required} are required for lag={args.lag}, pre_len={args.pre_len}.'
            )

    datasets = []
    for name, start, end in segments:
        arrays = pro_data(dataset_pollutant[start:end], dataset_wind[start:end], args)
        datasets.append(MyDataset(*arrays))
        print(f'Chronological {name}: time steps [{start}, {end}), '
              f'{end - start} steps, {len(datasets[-1])} windows')
    train_dataset, val_dataset, test_dataset = datasets
    pin_memory = torch.cuda.is_available()
    train_generator = torch.Generator().manual_seed(seed + 10000)
    loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': pin_memory, 'persistent_workers': args.num_workers > 0}
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=train_generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    return (train_loader, val_loader, test_loader)

def uses_terrain_physics(ablation):
    return ablation not in {'no_terrain_freq', 'no_all', 'no_terrain_physics'}

def uses_wind_physics(ablation):
    return ablation not in {'no_wind_time', 'no_freq_time_embed', 'no_all', 'no_wind_physics'}

def uses_unconstrained_scale_gate(ablation):
    return ablation == 'none_free_scale_gate'

def get_wind_input(ablation, data_wind):
    if uses_wind_physics(ablation):
        return data_wind
    return None

def init_model(args, ablation, terrain_input, lat_lon_tensor, device):
    terrain_for_model = terrain_input if uses_terrain_physics(ablation) else None
    model_class = PADformerCleanAblation if ablation in CLEAN_V2_ABLATIONS else PADformer
    return model_class(Input_len=args.lag, out_len=args.pre_len, num_id=args.num_id, num_layer=args.num_layer, dropout=args.dropout, muti_head=args.muti_head, num_samp=args.num_samp, IF_node=args.IF_node, num_nodes=args.num_nodes, var_dim=args.var_dim, terrain_3ch=terrain_for_model, lat_lon=lat_lon_tensor, num_terrain_channels=args.terrain_channels, ablation=ablation, constrain_physics_sign=not uses_unconstrained_scale_gate(ablation)).to(device)

def log_physics_parameters(net, ablation, logger):
    parts = []
    if uses_terrain_physics(ablation):
        try:
            att_module = net.encoder_terrain.space_att
            weights = torch.softmax(att_module.terrain_weights, dim=0).detach().cpu().tolist()
            if uses_unconstrained_scale_gate(ablation):
                scale = att_module.terrain_bias_scale.item()
                scale_name = 'ScaleRaw'
            else:
                scale = -F.softplus(att_module.terrain_bias_scale).item()
                scale_name = 'ScaleEff'
            weight_str = '|'.join((f'{name}:{value:.2f}' for name, value in zip(terrain_channel_names(len(weights)), weights)))
            parts.append(f'[TerrainRes] {weight_str}|{scale_name}:{scale:.4f}')
        except Exception:
            pass
    if uses_wind_physics(ablation):
        try:
            att_module = net.encoder_wind.space_att
            if uses_unconstrained_scale_gate(ablation):
                gate = att_module.wind_gate.item()
                gate_name = 'GateRaw'
            else:
                gate = torch.sigmoid(att_module.wind_gate).item()
                gate_name = 'GateEff'
            decay = F.softplus(att_module.decay_power).item()
            proj_scale = F.softplus(att_module.proj_scale).item()
            parts.append(f'[Wind] {gate_name}:{gate:.4f}|DecayEff:{decay:.3f}|ScaleEff:{proj_scale:.3f}')
        except Exception:
            pass
    if parts:
        logger.info('Physics params: ' + ' || '.join(parts))

def forward_batch(net, data_x, data_wind, args, ablation, device):
    data_x = data_x.to(device, non_blocking=True)
    data_wind = data_wind.to(device, non_blocking=True)
    B, N, feat_dim, T = data_x.shape
    inp = data_x.permute(0, 3, 1, 2).reshape(B, T, N * feat_dim)
    output = net(inp, wind_batch=get_wind_input(ablation, data_wind))
    return output.reshape(B, args.pre_len, N, feat_dim).permute(0, 2, 3, 1)

def evaluate_and_collect_metrics(test_outputs, test_target, logger):
    row = {}
    for t, h_name in zip(SINGLE_INDICES, SINGLE_NAMES):
        logger.info(f'=== [Single Point] {h_name} Prediction ===')
        for idx, pollutant in enumerate(POLLUTANTS):
            mae, rmse, mape, r2 = All_Metrics(test_outputs[:, :, idx, t], test_target[:, :, idx, t])
            logger.info(f'{pollutant} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}')
            key = pollutant.replace('.', '')
            row[f'{key}_sp_{h_name}_MAE'] = mae.item()
            row[f'{key}_sp_{h_name}_RMSE'] = rmse.item()
            row[f'{key}_sp_{h_name}_MAPE'] = mape.item()
            row[f'{key}_sp_{h_name}_R2'] = r2.item()
    for w, h_name in zip(WINDOW_LENGTHS, WINDOW_NAMES):
        logger.info(f'=== [Window Average] {h_name} Horizon ===')
        for idx, pollutant in enumerate(POLLUTANTS):
            mae, rmse, mape, r2 = All_Metrics(test_outputs[:, :, idx, :w], test_target[:, :, idx, :w])
            logger.info(f'{pollutant} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}')
            key = pollutant.replace('.', '')
            row[f'{key}_wa_{h_name}_MAE'] = mae.item()
            row[f'{key}_wa_{h_name}_RMSE'] = rmse.item()
            row[f'{key}_wa_{h_name}_MAPE'] = mape.item()
            row[f'{key}_wa_{h_name}_R2'] = r2.item()
    return row

def run_one_experiment(base_args, ablation, seed, device):
    args = copy.deepcopy(base_args)
    args.seed = seed
    init_seed(seed, strict=args.strict_deterministic)
    result_filename = f'{DATASET_NAME}_{args.run_tag}_{ablation}_seed{seed}'
    log_dir = os.path.join(args.result_root, result_filename)
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, name=f'{MODEL_NAME}_{ablation}_seed{seed}', debug=args.debug)
    logger.info(f'===== Seed: {seed} | Ablation: {ablation} | Device: {device} =====')
    if device.type == 'cuda':
        logger.info(f"CUDA config | visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')} | gpu={torch.cuda.get_device_name(device)} | cuda={torch.version.cuda} | cudnn={torch.backends.cudnn.version()}")
    logger.info(f'Experiment: {ABLATION_DESCRIPTIONS[ablation]}')
    logger.info(f'Training config | lr={args.learning_rate:.6g} | dropout={args.dropout:.3f} | patience={args.early_stop_patience} | epochs={args.epochs} | min_delta={args.min_delta:g} | workers={args.num_workers} | strict_deterministic={args.strict_deterministic} | run_tag={args.run_tag}')
    best_path = os.path.join(log_dir, 'best_model.pth')
    dataset_pollutant, dataset_wind, lat_lon_tensor = load_raw_arrays(args, device)
    train_loader, val_loader, test_loader = build_dataloaders(args, dataset_pollutant, dataset_wind, seed)
    terrain_input = None
    if uses_terrain_physics(ablation):
        terrain_input = load_sliced_terrain_matrix(args.terrain_path, args.num_nodes, device, args.terrain_channels)
    init_seed(seed, strict=args.strict_deterministic)
    initialization_rng = torch.get_rng_state().clone()
    net = init_model(args, ablation, terrain_input, lat_lon_tensor, device)
    if ablation in PAIRING_ABLATIONS:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initialization_rng)
            reference = init_model(args, 'none_v2', terrain_input, lat_lon_tensor, torch.device('cpu'))
        reference_state = reference.state_dict()
        actual_state = net.state_dict()
        if reference_state.keys() != actual_state.keys() or any((not torch.equal(reference_state[key].cpu(), actual_state[key].cpu()) for key in reference_state)):
            raise RuntimeError('Pairing experiment changed full-model initialization or state structure.')
        counts = lambda model: (sum((p.numel() for p in model.parameters())), sum((p.numel() for p in model.parameters() if p.requires_grad)))
        if counts(net) != counts(reference):
            raise RuntimeError('Pairing experiment changed full-model parameter count.')
        if [(name, type(module)) for name, module in net.named_modules() if name] != [(name, type(module)) for name, module in reference.named_modules() if name]:
            raise RuntimeError('Pairing experiment changed full-model child modules.')
        logger.info(f'Pairing audit passed | total/trainable parameters={counts(net)} | identical initial state')
        del reference, reference_state, actual_state
    active_graph = net.active_graph()
    expected_fusion = {'swap_freq_time_v2': 'branch_norm_channel_gate', 'both_freq_v2': 'branch_norm_channel_gate', 'both_trend_v2': 'branch_norm_channel_gate', 'none_v2': 'branch_norm_channel_gate', 'no_arbitration': 'concat_projection', 'mean_fusion': 'fixed_mean', 'no_all': 'fixed_mean', 'no_freq_time_embed_v2': 'branch_norm_channel_gate', 'no_terrain_physics': 'branch_norm_channel_gate', 'no_wind_physics': 'branch_norm_channel_gate', 'no_arbitration_v2': 'branch_norm_fixed_mean'}
    if ablation in expected_fusion and active_graph.get('fusion') != expected_fusion[ablation]:
        raise RuntimeError(f'Ablation {ablation} expected fusion={expected_fusion[ablation]}, but active graph is {active_graph}. Check that model.py is saved and the correct file is being imported.')
    logger.info(f'Active graph: {json.dumps(active_graph, ensure_ascii=False)}')
    if ablation in PAIRING_ABLATIONS and active_graph['routing'] != PAIRING_ROUTING[ablation]:
        raise RuntimeError('Unexpected pairing input routing.')
    criterion = torch.nn.L1Loss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    best_loss = np.inf
    early_stop_reference_loss = np.inf
    not_improved_count = 0
    logger.info(f'Start training | Ablation: {ablation} | Seed: {seed}')
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        net.train()
        total_loss = torch.zeros((), device=device)
        for batch_idx, (data_x, data_y, data_wind) in enumerate(train_loader):
            data_y = data_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = forward_batch(net, data_x, data_wind, args, ablation, device)
            loss = criterion(output, data_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
            current_step = batch_idx + 1
            if current_step % args.log_step == 0 or current_step == len(train_loader):
                logger.info(f'Epoch {epoch}: {current_step:>3}/{len(train_loader)} | Loss: {loss.item():.6f}')
        train_epoch_loss = (total_loss / len(train_loader)).item()
        logger.info(f'Train Epoch {epoch}: Averaged Loss: {train_epoch_loss:.6f}')
        log_physics_parameters(net, ablation, logger)
        net.eval()
        total_val_loss = torch.zeros((), device=device)
        with torch.no_grad():
            for data_x, data_y, data_wind in val_loader:
                data_y = data_y.to(device, non_blocking=True)
                output = forward_batch(net, data_x, data_wind, args, ablation, device)
                total_val_loss += criterion(output, data_y)
        val_epoch_loss = (total_val_loss / len(val_loader)).item()
        epoch_seconds = time.perf_counter() - epoch_start
        logger.info(f'Val Epoch {epoch}: Averaged Loss: {val_epoch_loss:.6f} | Epoch Time: {epoch_seconds:.1f}s')
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(net.state_dict(), best_path)
            logger.info(f'Best model saved. Loss: {best_loss:.6f}')
        if val_epoch_loss < early_stop_reference_loss - args.min_delta:
            early_stop_reference_loss = val_epoch_loss
            not_improved_count = 0
        else:
            not_improved_count += 1
            if args.early_stop and not_improved_count >= args.early_stop_patience:
                logger.info(f'Early stopping after {args.early_stop_patience} epochs without an improvement of at least {args.min_delta:g}.')
                break
    logger.info('Start testing.')
    net.load_state_dict(torch.load(best_path, map_location=device))
    log_physics_parameters(net, ablation, logger)
    net.eval()
    test_output_batches = []
    test_target_batches = []
    with torch.no_grad():
        for data_x, data_y, data_wind in test_loader:
            output = forward_batch(net, data_x, data_wind, args, ablation, device)
            test_output_batches.append(output)
            test_target_batches.append(data_y.to(device, non_blocking=True))
    test_outputs = torch.cat(test_output_batches, dim=0)
    test_target = torch.cat(test_target_batches, dim=0)
    metric_row = evaluate_and_collect_metrics(test_outputs, test_target, logger)
    metric_row = {'Ablation': ablation, 'Seed': seed, 'Run_Tag': args.run_tag, 'Learning_Rate': args.learning_rate, 'Dropout': args.dropout, 'Early_Stop_Patience': args.early_stop_patience, 'Min_Delta': args.min_delta, 'Num_Workers': args.num_workers, 'Strict_Deterministic': args.strict_deterministic, 'Best_Val_Loss': best_loss, 'Terrain_Channels': args.terrain_channels, **metric_row}
    one_run_csv = os.path.join(log_dir, 'metrics.csv')
    pd.DataFrame([metric_row]).to_csv(one_run_csv, index=False, encoding='utf-8-sig')
    logger.info(f'One-run metrics saved: {one_run_csv}')
    del net, optimizer, criterion, test_outputs, test_target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metric_row

def save_summary(rows, args):
    os.makedirs(args.result_root, exist_ok=True)
    df_all = pd.DataFrame(rows)
    all_csv = os.path.join(args.result_root, f'Ablation_All_Results_{args.run_tag}.csv')
    df_all.to_csv(all_csv, index=False, encoding='utf-8-sig')
    print(f'All experiment details saved: {all_csv}')
    statistics = []
    for ablation in df_all['Ablation'].unique():
        ab_csv = os.path.join(args.result_root, f'Ablation_Result_{args.run_tag}_{ablation}.csv')
        df_ablation = df_all[df_all['Ablation'] == ablation]
        df_ablation.to_csv(ab_csv, index=False, encoding='utf-8-sig')
        print(f'{ablation} details saved: {ab_csv}')
        numeric_columns = [column for column in df_ablation.select_dtypes(include=[np.number]).columns if column != 'Seed']
        stat_row = {'Ablation': ablation, 'Run_Tag': args.run_tag}
        for column in numeric_columns:
            stat_row[f'{column}_mean'] = df_ablation[column].mean()
            stat_row[f'{column}_std'] = df_ablation[column].std(ddof=1)
        statistics.append(stat_row)
    statistics_csv = os.path.join(args.result_root, f'Ablation_Statistics_{args.run_tag}.csv')
    pd.DataFrame(statistics).to_csv(statistics_csv, index=False, encoding='utf-8-sig')
    print(f'Ablation mean/std saved: {statistics_csv}')

def main():
    args = parse_args()
    device = get_device(args)
    print(f'Stage: {args.stage} | Run tag: {args.run_tag}')
    print(f'Experiments: {args.ablations} | Seeds: {args.seeds}')
    os.makedirs(args.result_root, exist_ok=True)
    config_path = os.path.join(args.result_root, f'Experiment_Config_{args.run_tag}.json')
    with open(config_path, 'w', encoding='utf-8') as file_obj:
        json.dump({'args': vars(args), 'canonical_ablation_suite': CANONICAL_ABLATIONS, 'clean_five_ablation_suite': NEW_FIVE_ABLATIONS, 'ablation_descriptions': ABLATION_DESCRIPTIONS}, file_obj, ensure_ascii=False, indent=2)
    print(f'Experiment config saved: {config_path}')
    all_rows = []
    for ablation in args.ablations:
        for seed in args.seeds:
            row = run_one_experiment(args, ablation, seed, device)
            all_rows.append(row)
    save_summary(all_rows, args)
if __name__ == '__main__':
    main()

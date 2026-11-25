"""Forecast from the latest historical window using a trained PADformer state dict."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from main import (POLLUTANTS, ABLATION_CHOICES, canonical_ablation, init_model,
                  load_raw_arrays, load_sliced_terrain_matrix,
                  uses_terrain_physics, forward_batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Experiment_Config_*.json from training.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='best_model.pth from the chosen run.')
    parser.add_argument('--ablation', choices=ABLATION_CHOICES, required=True,
                        help='Must match the chosen training run, including full vs ablated variants.')
    parser.add_argument('--csv_path', required=True, help='History CSV; at least lag steps per station.')
    parser.add_argument('--station_path', required=True)
    parser.add_argument('--terrain_path', required=True)
    parser.add_argument('--csv_layout', choices=['station_block', 'time_block'])
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0, etc.')
    parser.add_argument('--output', type=Path, default=Path('predictions/latest.csv'))
    opts = parser.parse_args()
    with opts.config.open(encoding='utf-8') as f:
        args = SimpleNamespace(**json.load(f)['args'])
    ablation = canonical_ablation(opts.ablation)
    if ablation not in args.ablations:
        parser.error('The selected ablation is absent from this training config.')
    for name in ('csv_path', 'station_path', 'terrain_path'):
        if not Path(getattr(opts, name)).is_file():
            parser.error(f'Required input not found: {getattr(opts, name)}')
        setattr(args, name, getattr(opts, name))
    if opts.csv_layout:
        args.csv_layout = opts.csv_layout
    device = torch.device(opts.device)
    pollution, wind, coordinates = load_raw_arrays(args, device)
    if coordinates.shape != (args.num_nodes, 2):
        parser.error('Station coordinates must have exactly num_nodes rows and two columns.')
    if len(pollution) < args.lag:
        parser.error(f'At least {args.lag} historical steps per station are required.')
    terrain = None
    if uses_terrain_physics(ablation):
        terrain = load_sliced_terrain_matrix(args.terrain_path, args.num_nodes, device, args.terrain_channels)
        if terrain is None:
            parser.error('Terrain inputs could not be loaded.')
    net = init_model(args, ablation, terrain, coordinates, device)
    state = torch.load(opts.checkpoint, map_location=device, weights_only=True)
    net.load_state_dict(state, strict=True)
    net.eval()
    x = torch.from_numpy(pollution[-args.lag:].transpose(1, 2, 0).copy()).unsqueeze(0)
    w = torch.from_numpy(wind[-args.lag:].transpose(1, 2, 0).copy()).unsqueeze(0)
    with torch.no_grad():
        prediction = forward_batch(net, x, w, args, ablation, device)[0].cpu().numpy()
    rows = []
    for station in range(args.num_nodes):
        for step in range(args.pre_len):
            rows.append({'station_index': station, 'forecast_step': step + 1,
                         **{name: float(prediction[station, i, step]) for i, name in enumerate(POLLUTANTS)}})
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(opts.output, index=False)
    print(f'Forecast saved: {opts.output.resolve()} ({len(rows)} rows)')


if __name__ == '__main__':
    main()

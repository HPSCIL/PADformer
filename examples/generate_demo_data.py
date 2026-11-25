"""Generate reproducible SYNTHETIC inputs for workflow checks, not experiments."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--steps', type=int, default=730)
    parser.add_argument('--seed', type=int, default=50)
    args = parser.parse_args()
    if args.num_nodes < 2 or args.steps < 720:
        parser.error('Use at least 2 stations and 720 steps for the default 72/72 workflow.')
    args.output.mkdir(parents=True, exist_ok=True)
    files = [args.output / name for name in ('air_quality.csv', 'stations.xlsx', 'terrain.npz')]
    if any(path.exists() for path in files):
        parser.error('Output already contains input files; choose a new directory.')
    rng = np.random.default_rng(args.seed)
    t = np.arange(args.steps)
    frames = []
    for station in range(args.num_nodes):
        phase = t / 24 * 2 * np.pi + station * 0.3
        concentrations = np.maximum(
            np.array([30, 55, 65, 25]) + 8 * np.sin(phase[:, None])
            + rng.normal(0, 2, (args.steps, 4)), 0.1)
        frame = pd.DataFrame(concentrations, columns=['PM2.5', 'PM10', 'O3', 'NO2'])
        frame['u10(m/s)'] = 2 * np.cos(phase)
        frame['v10(m/s)'] = 2 * np.sin(phase)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(files[0], index=False)
    pd.DataFrame({'latitude': 30 + np.arange(args.num_nodes) * 0.05,
                  'longitude': 114 + np.arange(args.num_nodes) * 0.05}).to_excel(files[1], index=False)
    terrain = {}
    for name in ('relief', 'climb', 'slope', 'tpi', 'svf'):
        matrix = rng.uniform(0, 1, (args.num_nodes, args.num_nodes)).astype('float32')
        np.fill_diagonal(matrix, 0)
        terrain[name] = matrix
    np.savez_compressed(files[2], **terrain)
    print(f'SYNTHETIC demonstration data: {args.output.resolve()}')
    print('Station-block layout. These values are not observations or research results.')


if __name__ == '__main__':
    main()

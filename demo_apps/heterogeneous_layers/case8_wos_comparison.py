"""Run case-8 WoSX GPU walks and compare with a saved order-4 FEM matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from demo_apps.heterogeneous_layers.case8_support.full_case8 import make_case
from demo_apps.heterogeneous_layers.case8_support.gpu_wos import implementation_sha256
from demo_apps.heterogeneous_layers.case8_support.reference_io import load_fem_matrix


def run(*args):
    subprocess.run([str(value) for value in args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fem-reference', type=Path, default=ROOT / 'data/case8/fem/refined-order4.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/case8/wos')
    parser.add_argument('--potential-grid', type=int, nargs=2, default=(80, 64), metavar=('NX', 'NZ'))
    parser.add_argument('--potential-paths', type=int, default=128)
    parser.add_argument('--paths', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=73193)
    parser.add_argument('--tolerance', type=float, default=5e-5)
    parser.add_argument('--continuations', type=int, default=200)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--reuse-gpu', action='store_true', help='Reuse matching saved GPU paths')
    args = parser.parse_args()
    if args.paths < 2 or args.potential_paths < 2 or min(args.potential_grid) < 2 or args.tolerance <= 0 or args.continuations < 0:
        parser.error('Invalid walk settings')
    if not args.fem_reference.is_file():
        parser.error(f'Missing FEM result: {args.fem_reference}; run case8_fem_convergence.py --enriched')
    load_fem_matrix(args.fem_reference)
    if args.fem_reference.suffix == '.json':
        fem = json.loads(args.fem_reference.read_text())
        fem = fem[-1] if isinstance(fem, list) else fem
        if fem.get('order') != 4:
            parser.error('The comparison requires an order-4 FEM result')
    deck = ROOT / 'demo_apps/heterogeneous_layers/case8/case8.cap3d'
    pinned = json.loads((deck.parent / 'SOURCE_REVISION.json').read_text())
    if hashlib.sha256(deck.read_bytes()).hexdigest() != pinned['case8_sha256']:
        parser.error('Case-8 deck differs from its pinned source hash')
    if args.build:
        run('cmake', '--build', 'build', '--target', 'heterogeneous_rectilinear_walk_gpu', '--parallel', 2)
    executable = ROOT / 'build/demo_apps/heterogeneous_rectilinear_walk_gpu'
    if not executable.is_file():
        parser.error('Build heterogeneous_rectilinear_walk_gpu first, or pass --build')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu, paths = args.output_dir / 'gpu-wos.json', args.output_dir / 'gpu-paths.npz'
    case_hash = hashlib.sha256((json.dumps(make_case(), indent=2) + '\n').encode()).hexdigest()
    reuse = False
    if args.reuse_gpu and gpu.is_file() and paths.is_file():
        report = json.loads(gpu.read_text())
        settings = report['parameters']
        reuse = (report['case_sha256'] == case_hash and
                 report['gpu_implementation_sha256'] == implementation_sha256(executable) and
                 [row['terminal'] for row in report['rows']] == list(range(1, 9)) and
                 all(row['unfinished'] == 0 for row in report['rows']) and
                 settings['paths'] == args.paths and settings['seed'] == args.seed and
                 settings['tolerance'] == args.tolerance and
                 settings['interface_tolerance'] == args.tolerance and
                 settings['continuations'] == args.continuations)
    if not reuse:
        run(sys.executable, '-m', 'demo_apps.heterogeneous_layers.case8_support.gpu_wos',
            '--rows', *range(1, 9), '--paths', args.paths, '--seed', args.seed,
            '--tolerance', args.tolerance, '--interface-tolerance', args.tolerance,
            '--continuations', args.continuations, '--output', gpu, '--path-output', paths)
    else:
        print(f'Reusing matching GPU paths: {gpu}', flush=True)
    potential = args.output_dir / 'wos-potential.npz'
    potential_seed = args.seed + 4242
    reuse_potential = False
    if args.reuse_gpu and potential.is_file():
        with np.load(potential, allow_pickle=False) as saved:
            reuse_potential = (str(saved['case_sha256']) == case_hash and
                str(saved['gpu_implementation_sha256']) == implementation_sha256(executable) and
                int(saved['paths_per_pixel']) == args.potential_paths and
                int(saved['seed']) == potential_seed and
                float(saved['tolerance']) == args.tolerance and
                int(saved['continuations']) == args.continuations and
                saved['potential'].shape == tuple(reversed(args.potential_grid)))
    if not reuse_potential:
        run(sys.executable, '-m', 'demo_apps.heterogeneous_layers.case8_support.gpu_potential',
            '--output', potential, '--grid', *args.potential_grid,
            '--paths-per-pixel', args.potential_paths, '--seed', potential_seed,
            '--tolerance', args.tolerance, '--continuations', args.continuations)
    else:
        print(f'Reusing matching GPU potential: {potential}', flush=True)
    run(sys.executable, '-m', 'demo_apps.heterogeneous_layers.case8_support.gpu_assess', gpu,
        '--fem-reference', args.fem_reference, '--path-data', paths,
        '--wos-potential', potential,
        '--figure', args.output_dir / 'comparison.png',
        '--output', args.output_dir / 'assessment.json')
    print(f'Saved comparison to {args.output_dir}')


if __name__ == '__main__':
    main()

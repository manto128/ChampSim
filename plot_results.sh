#!/usr/bin/env python3
#
# plot_results.sh - compare replacement policies against an LRU baseline.
#
# Reads the per-benchmark JSON stats produced by run_sim.sh and generates
# two grouped bar plots for SHiP, Hawkeye, and Mockingjay:
#   1) LLC MPKI reduction over LRU (%)  per CRC2 benchmark
#   2) Speedup over LRU (%)             per CRC2 benchmark
#
# The plots are written to plots/ with names derived from the run date/time.
#
# Examples:
#   ./plot_results.sh --lru results/lru --hawkeye results/hawkeye --ship results/ship --mockingjay results/mockingjay
#   ./plot_results.sh --lru results/lru --hawkeye results/hawkeye --ship results/ship --mockingjay results/mockingjay -o myplots

import argparse
import datetime
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CHAMPSIM_ROOT = os.path.dirname(os.path.abspath(__file__))

# Demand accesses that count toward MPKI (write hits/misses are fills, not demands)
MPKI_TYPES = ('LOAD', 'RFO', 'TRANSLATION')

# Categorical series colors (fixed order) and chart chrome
SERIES_COLORS = ['#2a78d6', '#008300', '#e87ba4']   # slots 1-3: blue, green, magenta (CVD-validated order)
SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'

def load_run(results_dir, cache_level):
    '''
    Read every JSON stats file under <results_dir>/json (or the directory
    itself) and return {benchmark: {'mpki': float, 'ipc': float}}.
    '''
    json_dir = os.path.join(results_dir, 'json')
    if not os.path.isdir(json_dir):
        json_dir = results_dir
    if not os.path.isdir(json_dir):
        sys.exit(f'ERROR: results directory not found: {results_dir}')

    stats = {}
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith('.json') or fname == 'config.json':
            continue
        with open(os.path.join(json_dir, fname)) as rfp:
            phases = json.load(rfp)

        # Use the detailed-simulation phase's region of interest
        phase = next((p for p in phases if p.get('name') == 'Simulation'), phases[-1])
        roi = phase['roi']

        core = roi['cores'][0]
        instructions, cycles = core['instructions'], core['cycles']

        cache = roi[cache_level]
        misses = sum(sum(cache[t]['miss']) for t in MPKI_TYPES)

        stats[os.path.splitext(fname)[0]] = {
            'mpki': misses / (instructions / 1000),
            'ipc': instructions / cycles,
        }

    if not stats:
        sys.exit(f'ERROR: no JSON stats files found under {json_dir}')
    return stats

def grouped_bars(ax, benchmarks, series, separate_last=False):
    '''Draw one grouped bar chart. series = [(label, values), ...]'''
    x = np.arange(len(benchmarks))
    slot = 0.76 / len(series)
    width = slot * 0.85   # leftover slot width is the surface gap between touching bars
    offset0 = -0.38 + slot / 2
    for i, (label, values) in enumerate(series):
        ax.bar(x + offset0 + i * slot, values, width, label=label,
               color=SERIES_COLORS[i], edgecolor='none', zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=9, color=INK_SECONDARY)
    ax.axhline(0, color=BASELINE, linewidth=1.0, zorder=3)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='y', length=0, colors=INK_MUTED)
    ax.tick_params(axis='x', length=0)
    ax.legend(frameon=False, loc='best', labelcolor=INK_PRIMARY)

    if separate_last:
        # Set the aggregate group apart: hairline separator and an emphasized label
        ax.axvline(len(benchmarks) - 1.5, color=GRIDLINE, linewidth=0.8, zorder=1)
        last = ax.get_xticklabels()[-1]
        last.set_fontweight('bold')
        last.set_color(INK_PRIMARY)

def make_plot(benchmarks, series, ylabel, title, path, separate_last=False):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    grouped_bars(ax, benchmarks, series, separate_last=separate_last)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, loc='left', fontweight='bold', color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f'wrote {path}')

def parse_args():
    parser = argparse.ArgumentParser(description='Plot MPKI reduction and speedup over LRU for SHiP, Hawkeye, and Mockingjay.')
    parser.add_argument('--lru', required=True, metavar='DIR',
            help='Results directory of the LRU baseline run (a run_sim.sh output directory)')
    parser.add_argument('--hawkeye', required=True, metavar='DIR',
            help='Results directory of the Hawkeye run')
    parser.add_argument('--ship', required=True, metavar='DIR',
            help='Results directory of the SHiP run')
    parser.add_argument('--mockingjay', required=True, metavar='DIR',
            help='Results directory of the Mockingjay run')
    parser.add_argument('-o', '--output', default=os.path.join(CHAMPSIM_ROOT, 'plots'),
            help='Directory to write the plots into (default: plots/)')
    parser.add_argument('--cache', default='LLC',
            help='Cache level whose misses are counted for MPKI (default: LLC)')
    return parser.parse_args()

def main():
    args = parse_args()

    runs = {
        'LRU': load_run(args.lru, args.cache),
        'SHiP': load_run(args.ship, args.cache),
        'Hawkeye': load_run(args.hawkeye, args.cache),
        'Mockingjay': load_run(args.mockingjay, args.cache),
    }

    # Compare only benchmarks present in all runs
    benchmarks = sorted(set.intersection(*(set(r) for r in runs.values())), key=str.lower)
    if not benchmarks:
        sys.exit('ERROR: no common benchmarks across the runs')
    for label, run in runs.items():
        missing = sorted(set(run) - set(benchmarks))
        if missing:
            print(f'warning: skipping benchmarks not in all runs ({label} extras: {", ".join(missing)})')

    lru = runs['LRU']
    policies = ('SHiP', 'Hawkeye', 'Mockingjay')

    def pct(fn):
        return {p: [fn(runs[p][b], lru[b]) * 100 for b in benchmarks] for p in policies}

    # MPKI reduction relative to LRU: positive = fewer misses than LRU
    mpki_red = pct(lambda r, base: 1 - r['mpki'] / base['mpki'] if base['mpki'] > 0 else 0.0)
    # IPC speedup relative to LRU: positive = faster than LRU
    speedup = pct(lambda r, base: r['ipc'] / base['ipc'] - 1)

    # Rightmost aggregate bars: arithmetic mean for MPKI reduction,
    # geometric mean of the speedup ratios for speedup
    mean_mpki_red = {p: float(np.mean(mpki_red[p])) for p in policies}
    gmean_speedup = {
        p: 100 * (float(np.exp(np.mean(np.log([runs[p][b]['ipc'] / lru[b]['ipc'] for b in benchmarks])))) - 1)
        for p in policies
    }

    os.makedirs(args.output, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    make_plot(benchmarks + ['Mean'],
              [(p, mpki_red[p] + [mean_mpki_red[p]]) for p in policies],
              ylabel=f'{args.cache} MPKI reduction over LRU (%)',
              title=f'{args.cache} MPKI reduction over LRU',
              path=os.path.join(args.output, f'mpki_over_lru_{stamp}.png'),
              separate_last=True)

    make_plot(benchmarks + ['Geomean'],
              [(p, speedup[p] + [gmean_speedup[p]]) for p in policies],
              ylabel='Speedup over LRU (%)',
              title='Speedup over LRU',
              path=os.path.join(args.output, f'speedup_over_lru_{stamp}.png'),
              separate_last=True)

    # Table view of the plotted values, next to the plots
    csv_path = os.path.join(args.output, f'summary_{stamp}.csv')
    with open(csv_path, 'w') as wfp:
        wfp.write('benchmark,' + ','.join(f'{p}_mpki_reduction_pct,{p}_speedup_pct' for p in policies) + '\n')
        for i, b in enumerate(benchmarks):
            row = [b]
            for p in policies:
                row += [f'{mpki_red[p][i]:.4f}', f'{speedup[p][i]:.4f}']
            wfp.write(','.join(row) + '\n')
        row = ['mean/geomean']
        for p in policies:
            row += [f'{mean_mpki_red[p]:.4f}', f'{gmean_speedup[p]:.4f}']
        wfp.write(','.join(row) + '\n')
    print(f'wrote {csv_path}')

    # Aggregate summary on stdout
    for p in policies:
        print(f'{p:8s} geomean speedup: {gmean_speedup[p]:+.2f}%   mean MPKI reduction: {mean_mpki_red[p]:+.2f}%')

    return 0

if __name__ == '__main__':
    sys.exit(main())

# vim: set filetype=python:

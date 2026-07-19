#!/usr/bin/env python3
#
# run_sim.sh - configure, build, and run ChampSim on one or more benchmarks.
#
# Each benchmark runs in its own thread, which drives a separate simulator
# process. By default all traces under CRC2_trace/ are run.
#
# Examples:
#   ./run_sim.sh -c champsim_config.json -o results/hawkeye
#   ./run_sim.sh -c myconfig.json -o results/test -b astar_163B
#   ./run_sim.sh -c myconfig.json -o results/test -l benchmarks.txt -j 8
#   ./run_sim.sh -c myconfig.json -o results/quick -b lbm_94B -w 1000000 -i 5000000 --no-build
#
# The LLC replacement policy in the config can be overridden with -r. The
# binary is then named champsim_<policy>, so several policies can be built
# once (--build-only) and their simulations run concurrently (--no-build):
#   ./run_sim.sh -r mockingjay --build-only
#   ./run_sim.sh -r mockingjay --no-build -o results/mockingjay -j 6

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading

CHAMPSIM_ROOT = os.path.dirname(os.path.abspath(__file__))

TRACE_SUFFIXES = ('.trace.xz', '.champsimtrace.xz')

print_lock = threading.Lock()

def log(*args):
    with print_lock:
        print(*args, flush=True)

def resolve_trace(name, trace_dir):
    ''' Resolve a benchmark name, file name, or path to a trace file. '''
    candidates = [name, os.path.join(trace_dir, name)]
    candidates += [os.path.join(trace_dir, name + sfx) for sfx in TRACE_SUFFIXES]
    for candidate in candidates:
        candidate = os.path.expandvars(os.path.expanduser(candidate))
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    sys.exit(f"ERROR: cannot find trace for benchmark '{name}' (searched in {trace_dir})")

def benchmark_name(trace_path):
    name = os.path.basename(trace_path)
    for sfx in ('.xz', '.trace', '.champsimtrace'):
        name = name.removesuffix(sfx)
    return name

def gather_traces(args):
    benchmarks = list(args.benchmark)
    if args.list:
        with open(args.list) as rfp:
            for line in rfp:
                line = line.split('#', 1)[0].strip()
                if line:
                    benchmarks.append(line)

    if benchmarks:
        return [resolve_trace(b, args.trace_dir) for b in benchmarks]

    if not os.path.isdir(args.trace_dir):
        sys.exit(f'ERROR: trace directory not found: {args.trace_dir}')
    traces = sorted(
        os.path.join(args.trace_dir, f)
        for f in os.listdir(args.trace_dir)
        if f.endswith(TRACE_SUFFIXES)
    )
    if not traces:
        sys.exit(f'ERROR: no traces found under {args.trace_dir}')
    return [os.path.abspath(t) for t in traces]

def apply_replacement(config_path, policy):
    ''' Write a copy of the config with the LLC replacement policy overridden.
        The executable is renamed champsim_<policy> so builds don't clobber
        each other. Returns the path of the patched config. '''
    with open(config_path) as rfp:
        config = json.load(rfp)
    config.setdefault('LLC', {})['replacement'] = policy
    config['executable_name'] = f'champsim_{policy}'
    patched = os.path.join(CHAMPSIM_ROOT, f'.champsim_config_{policy}.json')
    with open(patched, 'w') as wfp:
        json.dump(config, wfp, indent=2)
    return patched

def build(config, dry_run):
    configure_cmd = [os.path.join(CHAMPSIM_ROOT, 'config.sh'), config]
    make_cmd = ['make', '-C', CHAMPSIM_ROOT, f'-j{os.cpu_count()}']

    log(f'=== Configuring with {config}')
    if dry_run:
        log('+', ' '.join(configure_cmd))
        log('+', ' '.join(make_cmd))
        return
    subprocess.run(configure_cmd, cwd=CHAMPSIM_ROOT, check=True)
    log('=== Building')
    subprocess.run(make_cmd, check=True)

def run_one(binary, trace, args, outdir):
    ''' Run a single simulation. Returns (benchmark_name, return_code). '''
    name = benchmark_name(trace)
    log_path = os.path.join(outdir, 'logs', f'{name}.log')
    json_path = os.path.join(outdir, 'json', f'{name}.json')

    cmd = [binary,
           '--warmup-instructions', str(args.warmup),
           '--simulation-instructions', str(args.instructions),
           '--json', json_path,
           trace]

    if args.dry_run:
        log('+', ' '.join(cmd), '>', log_path, '2>&1')
        return name, 0

    log(f'  [started] {name}')
    with open(log_path, 'w') as wfp:
        result = subprocess.run(cmd, stdout=wfp, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        log(f'  [done]    {name}')
    else:
        log(f'  [FAILED]  {name} (exit {result.returncode}, see {log_path})')
    return name, result.returncode

def parse_args():
    parser = argparse.ArgumentParser(description='Build and run ChampSim simulations, one thread per benchmark.')
    parser.add_argument('-c', '--config', default=os.path.join(CHAMPSIM_ROOT, 'champsim_config.json'),
            help='ChampSim JSON config file (default: champsim_config.json)')
    parser.add_argument('-r', '--replacement', metavar='POLICY',
            help='Override the LLC replacement policy in the config (e.g. lru, ship, hawkeye, '
                 'mockingjay). The binary is named champsim_<POLICY> so policies can coexist in bin/.')
    parser.add_argument('--build-only', action='store_true',
            help='Configure and build, then exit without running simulations')
    parser.add_argument('-o', '--output',
            help='Output directory for logs and JSON stats (required unless --build-only)')
    parser.add_argument('-b', '--benchmark', action='append', default=[], metavar='NAME',
            help='Run a single benchmark. May be given multiple times. NAME is a trace name '
                 '(astar_163B), file name (astar_163B.trace.xz), or a full path to a trace.')
    parser.add_argument('-l', '--list', metavar='FILE',
            help="File with one benchmark name per line ('#' comments ok)")
    parser.add_argument('-d', '--trace-dir', default=os.path.join(CHAMPSIM_ROOT, 'CRC2_trace'),
            help='Directory to search for traces (default: CRC2_trace). With no -b/-l, all '
                 '*.trace.xz and *.champsimtrace.xz under it are run.')
    parser.add_argument('-w', '--warmup', type=int, default=200_000_000,
            help='Warmup instructions (default: %(default)s)')
    parser.add_argument('-i', '--instructions', type=int, default=500_000_000,
            help='Simulation instructions (default: %(default)s)')
    parser.add_argument('-j', '--jobs', type=int, default=0,
            help='Max concurrent simulations (default: one thread per benchmark)')
    parser.add_argument('--no-build', action='store_false', dest='build',
            help='Skip config.sh + make, use the existing binary')
    parser.add_argument('-n', '--dry-run', action='store_true',
            help='Print the commands without running them')
    return parser.parse_args()

def main():
    args = parse_args()

    if not args.build_only and not args.output:
        sys.exit('ERROR: -o/--output is required unless --build-only is given')

    if not os.path.isfile(args.config):
        sys.exit(f'ERROR: config file not found: {args.config}')
    args.config = os.path.abspath(args.config)

    if args.replacement:
        args.config = apply_replacement(args.config, args.replacement)

    # The binary name comes from "executable_name" in the JSON config
    with open(args.config) as rfp:
        bin_name = json.load(rfp).get('executable_name', 'champsim')
    binary = os.path.join(CHAMPSIM_ROOT, 'bin', bin_name)

    if args.build:
        try:
            build(args.config, args.dry_run)
        except subprocess.CalledProcessError as err:
            sys.exit(f'ERROR: {" ".join(err.cmd)} failed with exit code {err.returncode}')
    if args.build_only:
        log(f'=== Build-only: {binary}')
        return 0

    traces = gather_traces(args)

    if not args.dry_run and not os.access(binary, os.X_OK):
        sys.exit(f'ERROR: binary not found: {binary} (build it or drop --no-build)')

    outdir = os.path.abspath(args.output)
    os.makedirs(os.path.join(outdir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'json'), exist_ok=True)
    if not args.dry_run:
        shutil.copy(args.config, os.path.join(outdir, 'config.json'))  # provenance

    log(f'=== Running {len(traces)} benchmark(s): warmup={args.warmup}, instructions={args.instructions}')
    log(f'=== Output: {outdir}')

    max_workers = args.jobs if args.jobs > 0 else len(traces)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda t: run_one(binary, t, args, outdir), traces))

    if args.dry_run:
        return 0

    failed = [name for name, rc in results if rc != 0]
    log('')
    if failed:
        log(f'=== {len(failed)} of {len(results)} simulation(s) FAILED: {" ".join(failed)}')
        return 1
    log(f'=== All {len(results)} simulation(s) completed successfully.')
    log(f'=== Logs: {outdir}/logs/   JSON stats: {outdir}/json/')
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

# vim: set filetype=python:

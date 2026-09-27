"""Validate launch counts and distinguish profiler warmup copies from simulation."""
import csv, collections, sys
from pathlib import Path

root = Path(sys.argv[1])
summary = []
for path in sorted(root.glob('*-device.csv')):
    stem = path.name.removesuffix('-device.csv')
    device = list(csv.DictReader(path.open()))
    host = list(csv.DictReader(path.with_name(stem+'-host.csv').open()))
    assert all(None not in row for row in device+host), 'malformed CSV'
    kernels = [r for r in device if not r['name'].startswith('[')]
    memory = [r for r in device if r['name'].startswith('[')]
    expected = 2048 if stem.startswith('breakout-') else 384
    assert len(kernels) == expected, (stem, len(kernels))
    start = min(float(r['start']) for r in kernels)
    # CUDA.jl's profiler warms itself with CuArray([1]) before running our closure.
    assert len(memory) == 1 and memory[0]['size'] == '8', (stem, memory)
    assert float(memory[0]['stop']) < start
    calls = collections.Counter(r['name'] for r in host)
    launch = {k:v for k,v in calls.items() if 'LaunchKernel' in k or 'GraphLaunch' in k}
    graph_launches = sum(v for k,v in launch.items() if 'GraphLaunch' in k)
    kernel_launches = sum(v for k,v in launch.items() if 'LaunchKernel' in k)
    graph = '-graph_' in stem
    expected_graphs = (32 if stem.startswith('breakout-') else 2) if graph else 0
    assert graph_launches == expected_graphs, (stem, launch)
    assert kernel_launches == (0 if graph else expected), (stem, launch)
    summary.append(dict(trace=stem, kernels=len(kernels),
        kernel_launch_api_calls=kernel_launches, graph_launch_api_calls=graph_launches,
        simulation_memory_operations=0, profiler_warmup_copy_bytes=8))
    print(stem, dict(sorted(launch.items())))
assert summary, f'no CUPTI device traces found in {root}'
with (root/'summary.csv').open('w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=summary[0].keys())
    writer.writeheader(); writer.writerows(summary)

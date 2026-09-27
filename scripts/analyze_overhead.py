"""Analyze randomized paired timing trials; bootstrap pairs, not individual samples."""
import csv, random, statistics, json, sys
from pathlib import Path

source=Path(sys.argv[1])
rows=list(csv.DictReader(source.open()))
rng=random.Random(20260927)
summary=[]
for env in sorted({r['environment'] for r in rows}):
    group=[r for r in rows if r['environment']==env]
    trials={}
    for r in group:
        trials.setdefault(int(r['replicate']),{})[r['mode']]=r
    for reference, candidate in [('native','julia'),('graph_native','graph_julia'),('julia','graph_julia')]:
        for metric in ['wall_seconds','device_seconds']:
            ratios=[float(t[reference][metric])/float(t[candidate][metric]) for t in trials.values()]
            bootstrap=sorted(statistics.mean(rng.choices(ratios,k=len(ratios))) for _ in range(20000))
            ref=statistics.median(float(t[reference][metric]) for t in trials.values())
            cand=statistics.median(float(t[candidate][metric]) for t in trials.values())
            record=dict(environment=env, reference=reference, candidate=candidate, metric=metric,
                        repeats=len(ratios), mean_paired_speedup=statistics.mean(ratios),
                        paired_speedup_95pct=[bootstrap[500],bootstrap[19500]],
                        reference_median_seconds=ref,candidate_median_seconds=cand,
                        steps=int(group[0]['steps']),agents=int(group[0]['agents']))
            summary.append(record)
            print(env,reference,'->',candidate,metric, f'{record["mean_paired_speedup"]:.6f}',
                  f'95% [{bootstrap[500]:.6f}, {bootstrap[19500]:.6f}]')
source.with_name('analysis.json').write_text(json.dumps(summary,indent=2)+'\n')

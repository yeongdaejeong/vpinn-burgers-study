"""Fixed-seed depth x width VPINN experiment; no FEM training labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from pathlib import Path
import json, csv, hashlib, contextlib, concurrent.futures
import numpy as np
import torch
import reference_core as core
from study import CONFIG, validation_cache

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'architecture'
DEPTHS = [2, 3, 4]
WIDTHS = [16, 32, 64]
SEED = 2026

def run_one(pair):
    depth, width = pair
    out = OUT / 'runs' / f'd{depth}_w{width}_seed{SEED}'
    out.mkdir(parents=True, exist_ok=True)
    config = dict(CONFIG, depth=depth, width=width, seed=SEED,
                  validation_seed=314159, distribution='gauss', sample_count=5120)
    # Normalize line endings so Git checkout on Windows/Linux preserves cache identity.
    fingerprint = hashlib.sha256((Path(__file__).read_text(encoding='utf8') +
                                 (ROOT/'reference_core.py').read_text(encoding='utf8')).encode('utf8')).hexdigest()
    if (out/'complete.json').exists():
        done = json.loads((out/'complete.json').read_text())
        assert done['config'] == config and done['code_sha256'] == fingerprint
        return done['metrics']
    with (out/'execution.log').open('w', encoding='utf8') as log, contextlib.redirect_stdout(log):
        rule = core.make_test_rule(18, 40, 128, config['forcing'])
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = core.VectorDiskNet(width, depth).double()
        initial_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        history, stats = core.train_vpinn(model, rule, config)
        model.eval()
        ref, xy, target, weights, same, rich, strong_xy = validation_cache()
        errors = core.vector_errors(core.predict(model, xy), target, weights)
        residual, _ = core.independent_residual(model, strong_xy, .1, config['forcing'])
        metrics = dict(depth=depth, width=width, seed=SEED,
                       parameters=sum(p.numel() for p in model.parameters()),
                       train_seconds=stats['train_seconds'],
                       lbfgs_iterations=stats['lbfgs_iterations'], lbfgs_closures=stats['lbfgs_closures'])
        for component in ['vector', 'u', 'v']:
            metrics[component+'_relative_l2'] = errors[component]['relative_l2']
            metrics[component+'_relative_linf'] = errors[component]['sampled_relative_linf']
        for j, component in enumerate(['u', 'v']):
            metrics[component+'_strong_rms'] = float(np.sqrt(np.mean(residual[:,j]**2)))
        metrics['strong_rms'] = float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))
        metrics['boundary_max'] = float(np.linalg.norm(core.predict(model,core.circle_points(1024)),axis=1).max())
        for name, test in [('train_loss',rule),('independent_weak',same),('enriched_weak',rich)]:
            metrics[name] = float(core.weak_loss(core.weak_residual(model,test,.1,False).detach(),test))
        assert all(np.isfinite(v) for v in metrics.values())
        torch.save(dict(state_dict=model.state_dict(),config=config),out/'model.pt')
        np.savez_compressed(out/'points.npz',xy=rule.xy.numpy(),weights=rule.weights)
        for name, data in [('config',config),('metrics',metrics),('statistics',stats)]:
            (out/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf8')
        with (out/'history.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.writer(f);writer.writerow(['phase','step_or_closure','weak_loss','seconds']);writer.writerows(history)
        (out/'complete.json').write_text(json.dumps(dict(config=config,code_sha256=fingerprint,
             initial_state_sha256=initial_hash,metrics=metrics),indent=2),encoding='utf8')
    return metrics

def run_all(workers=3):
    OUT.mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(run_one,(d,w)):(d,w) for d in DEPTHS for w in WIDTHS}
        for future in concurrent.futures.as_completed(jobs):
            r=future.result()
            print(f"DONE depth={r['depth']} width={r['width']} L2={r['vector_relative_l2']:.6g} strong={r['strong_rms']:.6g}",flush=True)
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*/metrics.json'))]
    assert len(rows)==9
    with (OUT/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows

if __name__=='__main__':
    run_all()

"""Verify saved checkpoints, controls, metrics and notebook preservation."""
from pathlib import Path
import json, hashlib, csv
import numpy as np
import torch
import reference_core as core
from study import CONFIG

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'architecture'

def main():
    runs=sorted((OUT/'runs').glob('*/complete.json'))
    assert len(runs)==9
    xy,w=core.disk_rule(40,128)
    strong_xy=core.sample_disk(np.random.default_rng(314159),10000)
    checks=[]
    for p in runs:
        done=json.loads(p.read_text());c=done['config'];r=done['metrics']
        for key,value in CONFIG.items():
            if key not in ('depth','width'): assert c[key]==value,(p,key)
        assert c['seed']==2026 and c['validation_seed']==314159
        z=np.load(p.parent/'points.npz');np.testing.assert_array_equal(z['xy'],xy);np.testing.assert_array_equal(z['weights'],w)
        saved=torch.load(p.parent/'model.pt',weights_only=True,map_location='cpu')
        assert saved['config']==c
        m=core.VectorDiskNet(c['width'],c['depth']).double();m.load_state_dict(saved['state_dict']);m.eval()
        assert sum(v.numel() for v in m.parameters())==r['parameters']
        residual,_=core.independent_residual(m,strong_xy,.1,c['forcing'])
        rms=float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))
        np.testing.assert_allclose(rms,r['strong_rms'],rtol=1e-12,atol=1e-14)
        checks.append(dict(depth=c['depth'],width=c['width'],checkpoint_strong_rms_reproduced=True))
    base=json.loads((OUT/'runs/d3_w32_seed2026/metrics.json').read_text())
    previous=json.loads((ROOT/'runs/gauss_N5120_seed2026/metrics.json').read_text())
    baseline_differences={key:base[new]-previous[key] for key,new in [('relative_l2','vector_relative_l2'),('relative_linf','vector_relative_linf'),('strong_rms','strong_rms'),('train_loss','train_loss')]}
    # Same initialization/algorithm should reproduce the prior baseline, allowing roundoff.
    for key,new in [('relative_l2','vector_relative_l2'),('relative_linf','vector_relative_linf'),('strong_rms','strong_rms'),('train_loss','train_loss')]:
        np.testing.assert_allclose(base[new],previous[key],rtol=1e-7,atol=1e-12)
    nb=json.loads((ROOT/'VPINN_sampling_study.ipynb').read_text(encoding='utf8'))
    manifest=json.loads((OUT/'original_notebook_manifest.json').read_text())
    assert [hashlib.sha256(''.join(c['source']).encode()).hexdigest() for c in nb['cells'][:72]]==manifest['source_sha256']
    assert len(nb['cells'])==76
    new=nb['cells'][72:]
    assert sum('image/png' in o.get('data',{}) for c in new for o in c.get('outputs',[]))==3
    assert not [o for c in new for o in c.get('outputs',[]) if o['output_type']=='error']
    result=dict(runs=9,seed=2026,all_common_controls_verified=True,checkpoints=checks,
                baseline_differences=baseline_differences,original_72_sources_preserved=True,
                appended_cells_executed=True,embedded_figures=3)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

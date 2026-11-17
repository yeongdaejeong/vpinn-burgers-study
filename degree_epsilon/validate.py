from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import nbformat
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
import experiment as e
import reference_core as c

def main():
    rows=list(csv.DictReader((ROOT/'results.csv').open(encoding='utf-8-sig')))
    summary=list(csv.DictReader((ROOT/'summary.csv').open(encoding='utf-8-sig')))
    assert len(rows)==45 and len(summary)==9
    reused=0
    for eps in e.EPSILONS:
      for p in e.DEGREES:
        group=[r for r in rows if float(r['epsilon'])==eps and int(r['test_degree'])==p]
        assert sorted(int(r['seed']) for r in group)==e.SEEDS
        for seed in e.SEEDS:
            path=e.folder(eps,p,seed);config=json.loads((path/'config.json').read_text());meta=json.loads((path/'trained.json').read_text())
            assert meta['code_hash']==e.fingerprint()
            assert meta['scalar_test_dimension']==(p+1)*(p+2)//2
            original=BASE/'runs'/f'gauss_N5120_seed{seed}'
            oldconfig=json.loads((original/'config.json').read_text());oldmetrics=json.loads((original/'metrics.json').read_text())
            for key in ['width','depth','forcing','n_radial','n_angular','adam_steps','adam_lr','lbfgs_steps','lbfgs_lr']:
                assert config[key]==oldconfig[key],key
            assert config['epsilon']==eps and config['test_degree']==p
            assert meta['initial_state_sha256']==oldmetrics['initial_state_sha256']
            stats=json.loads((path/'statistics.json').read_text())
            assert stats['adam_updates']==3000 and stats['lbfgs_iterations']<=1500
            if meta['reused_original']:
                reused+=1;row=next(r for r in group if int(r['seed'])==seed)
                np.testing.assert_allclose(float(row['relative_l2']),oldmetrics['relative_l2'],rtol=1e-11,atol=1e-13)
                np.testing.assert_allclose(float(row['strong_rms']),oldmetrics['strong_rms'],rtol=1e-11)
        row=next(r for r in summary if float(r['epsilon'])==eps and int(r['test_degree'])==p)
        for key in ['relative_l2','strong_rms','common_p30_weak_96x256']:
            values=np.array([float(r[key]) for r in group])
            np.testing.assert_allclose(float(row[key+'_mean']),values.mean())
            np.testing.assert_allclose(float(row[key+'_se']),values.std(ddof=1)/np.sqrt(5))
        data=np.load(ROOT/'fields'/f'eps{eps:g}_p{p}.npz')
        assert data['individual'].shape==(5,201*201,2)
        np.testing.assert_allclose(data['mean'],data['individual'].sum(0)/5,equal_nan=True)
        np.testing.assert_allclose(data['se'],data['individual'].std(0,ddof=1)/np.sqrt(5),equal_nan=True)
    assert reused==5
    for p in e.DEGREES:
        rule=c.make_test_rule(p,40,128,'minus_one_minus_sin_pi_y')
        old=np.load(BASE/'runs/gauss_N5120_seed2026/points.npz')
        np.testing.assert_array_equal(rule.xy.numpy(),old['xy']);np.testing.assert_array_equal(rule.weights,old['weights'])
    check=json.loads((ROOT/'p3_solver_validation.json').read_text())
    for v in check.values():
        assert v['original_matlab_relative_residual']<1e-8
        assert v['jacobian_directional_relative_error']<1e-7
        assert v['baseline_nodal_difference']<1e-7
    audit=json.loads((ROOT/'quadrature_refinement.json').read_text())
    assert len(audit)==45
    for r in audit:
        assert np.isfinite(r['common_p30_weak']) and r['common_p30_weak']>=0
        if r['radial']==96:
            baseline=next(x for x in rows if float(x['epsilon'])==.02 and int(x['test_degree'])==r['test_degree'] and int(x['seed'])==r['seed'])
            np.testing.assert_allclose(r['common_p30_weak'],float(baseline['common_p30_weak_96x256']),rtol=1e-8,atol=1e-12)
    assert max(r['relative_residual_difference'] for r in audit if r['radial']==384)<.00025
    saved=json.loads((ROOT/'preservation.json').read_text())
    for name,digest in saved['files'].items():assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest,name
    n=nbformat.read(BASE/'VPINN_sampling_study.ipynb',4)
    assert len(n.cells)==72 and not any(o.output_type=='error' for cell in n.cells for o in cell.get('outputs',[]))
    assert sum('image/png' in o.get('data',{}) for o in n.cells[-1].outputs)==4
    assert any('text/html' in o.get('data',{}) for o in n.cells[-1].outputs)
    revision=BASE/'gauss_baseline/validation.json'
    expected=json.loads(revision.read_text())['notebook_first70_sha256'] if revision.exists() else saved['notebook_cells_hash']
    assert hashlib.sha256(json.dumps(n.cells[:70],sort_keys=True,ensure_ascii=False).encode()).hexdigest()==expected
    html=(BASE/'index.html').read_text(encoding='utf8');assert html.count('id="degree-epsilon"')==1
    import re
    for name in re.findall(r'(?:href|src)="(degree_epsilon/[^"]+)"',html):assert (BASE/name).exists(),name
    report={'status':'passed','seed_results':45,'new_models':40,'reused_models':5,'matched_initialization':True,
            'same_points_and_weights':True,'same_network_and_optimizer_budget':True,'statistics_recomputed':True,
            'mean_fields_and_standard_errors_verified':True,'original_files_preserved':len(saved['files']),
            'authorized_baseline_layout_preserved':True,'new_section_links_verified':True,'browser_visual_check':'not performed'}
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2));print(report,flush=True)

if __name__=='__main__':main()

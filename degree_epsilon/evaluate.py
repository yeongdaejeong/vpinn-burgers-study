"""Common independent diagnostics and P3 comparisons for all 45 checkpoints."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,csv
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE))
import reference_core as c
from experiment import EPSILONS,DEGREES,SEEDS,folder,fingerprint

def load_study_model(out):
    meta=json.loads((out/'trained.json').read_text())
    assert meta['code_hash']==fingerprint()
    expected=json.loads((out/'config.json').read_text())
    checkpoint=torch.load(BASE/meta['model_path'],map_location='cpu',weights_only=True)
    config=checkpoint['config']
    for key in ['epsilon','forcing','width','depth','test_degree','seed']:
        assert config[key]==expected[key],key
    assert config['forcing']=='minus_one_minus_sin_pi_y'
    model=c.VectorDiskNet(config['width'],config['depth']).double()
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    return model

def refpath(eps,h):
    return BASE/f'reference/p3_vector_burgers_sine_eps0p1_h{h}.mat' if eps==.1 else ROOT/f'reference/p3_eps{eps:g}_h{h}.mat'
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def main():
    assert len(list((ROOT/'runs').glob('*/trained.json')))==45
    (ROOT/'fields').mkdir(exist_ok=True)
    xy=c.sample_disk(np.random.default_rng(314159),10000)
    boundary=c.circle_points(1024)
    rich=c.make_test_rule(30,56,160,'minus_one_minus_sin_pi_y')
    finer=c.make_test_rule(30,96,256,'minus_one_minus_sin_pi_y')
    axis=np.linspace(-1,1,201);X,Y=np.meshgrid(axis,axis);grid=np.column_stack((X.ravel(),Y.ravel()))
    rows=[];refinement=[]
    for eps in EPSILONS:
        fine=c.P3VectorReference(refpath(eps,'0p02'),eps)
        coarse=c.P3VectorReference(refpath(eps,'0p04'),eps)
        q,qref,w=fine.quadrature(6);coarse_values=coarse.evaluate(q);valid=np.isfinite(coarse_values).all(1)
        diff=c.vector_errors(coarse_values[valid],qref[valid],w[valid])
        refinement.append({'epsilon':eps,'coarse_h':.04,'fine_h':.02,'relative_l2_difference':diff['vector']['relative_l2'],
            'included_area_fraction':float(w[valid].sum()/w.sum()),'fine_algebraic_residual':fine.algebraic_residual,
            'fine_vector_dofs':2*len(fine.p)})
        refgrid=fine.evaluate(grid);inside=np.isfinite(refgrid).all(1)&((grid**2).sum(1)<1)
        refgrid[~inside]=np.nan
        for p in DEGREES:
            trainrule=c.make_test_rule(p,40,128,'minus_one_minus_sin_pi_y')
            samerule=c.make_test_rule(p,56,160,'minus_one_minus_sin_pi_y')
            fields=[]
            for seed in SEEDS:
                out=folder(eps,p,seed);meta=json.loads((out/'trained.json').read_text())
                model=load_study_model(out)
                assert model.network[0].weight.shape[1]==2
                metrics=out/'evaluation.json'
                if metrics.exists():row=json.loads(metrics.read_text())
                else:
                    error=c.vector_errors(c.predict(model,q),qref,w)
                    r0=c.weak_residual(model,trainrule,eps,False).detach()
                    r1=c.weak_residual(model,samerule,eps,False).detach()
                    r2=c.weak_residual(model,rich,eps,False).detach()
                    r3=c.weak_residual(model,finer,eps,False).detach()
                    residual,_=c.independent_residual(model,xy,eps,'minus_one_minus_sin_pi_y')
                    stats=json.loads((out/'statistics.json').read_text())
                    row={'epsilon':eps,'test_degree':p,'test_dimension':len(trainrule.modes),'seed':seed,
                        'relative_l2':error['vector']['relative_l2'],'relative_linf':error['vector']['sampled_relative_linf'],
                        'u_relative_l2':error['u']['relative_l2'],'v_relative_l2':error['v']['relative_l2'],
                        'strong_rms':float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
                        'boundary_max':float(np.linalg.norm(c.predict(model,boundary),axis=1).max()),
                        'train_loss':float(c.weak_loss(r0,trainrule)),
                        'same_degree_independent_weak':float(c.weak_loss(r1,samerule)),
                        'common_p30_weak_56x160':float(c.weak_loss(r2,rich)),
                        'common_p30_weak_96x256':float(c.weak_loss(r3,finer)),
                        'quadrature_difference_norm':float(torch.sqrt(c.weak_loss(r2-r3,finer))),
                        'train_seconds':stats['train_seconds'],'gram_condition':trainrule.condition,
                        'lbfgs_iterations':stats['lbfgs_iterations'],'reused_original':meta['reused_original']}
                    assert all(np.isfinite(v) for v in row.values() if isinstance(v,float))
                    metrics.write_text(json.dumps(row,indent=2))
                rows.append(row)
                values=c.predict(model,grid);values[~inside]=np.nan;fields.append(values)
                print('Evaluated',len(rows),'/45',eps,p,seed,'relative L2',row['relative_l2'],flush=True)
            arr=np.stack(fields)
            np.savez_compressed(ROOT/'fields'/f'eps{eps:g}_p{p}.npz',axis=axis,reference=refgrid,
                individual=arr,mean=arr.mean(0),se=arr.std(0,ddof=1)/np.sqrt(5))
        del fine,coarse,q,qref,w
    write_csv(ROOT/'results.csv',rows);write_csv(ROOT/'reference_refinement.csv',refinement)
    (ROOT/'reference_refinement.json').write_text(json.dumps(refinement,indent=2))
    summary=[];metrics=['relative_l2','relative_linf','u_relative_l2','v_relative_l2','strong_rms','train_loss',
        'same_degree_independent_weak','common_p30_weak_56x160','common_p30_weak_96x256','quadrature_difference_norm','train_seconds']
    for eps in EPSILONS:
      for p in DEGREES:
        group=[r for r in rows if r['epsilon']==eps and r['test_degree']==p]
        assert sorted(r['seed'] for r in group)==SEEDS
        row={'epsilon':eps,'test_degree':p,'test_dimension':(p+1)*(p+2)//2,'n_seeds':5}
        for name in metrics:
            values=np.array([r[name] for r in group]);row[name+'_mean']=float(values.mean());row[name+'_se']=float(values.std(ddof=1)/np.sqrt(5))
        summary.append(row)
    write_csv(ROOT/'summary.csv',summary)
    paired=[]
    for eps in EPSILONS:
      base={r['seed']:r for r in rows if r['epsilon']==eps and r['test_degree']==18}
      for p in [22,26]:
        group={r['seed']:r for r in rows if r['epsilon']==eps and r['test_degree']==p}
        values=np.array([group[s]['relative_l2']-base[s]['relative_l2'] for s in SEEDS])*100
        paired.append({'epsilon':eps,'comparison':f'p{p} - p18','difference_pp':float(values.mean()),'se_pp':float(values.std(ddof=1)/np.sqrt(5))})
    write_csv(ROOT/'paired_degree_differences.csv',paired)
    print('Evaluation complete',flush=True)

if __name__=='__main__':main()

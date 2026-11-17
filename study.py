"""Controlled original VPINN experiment. No training on FEM or strong residuals."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from types import SimpleNamespace
import json, hashlib, csv, contextlib, concurrent.futures, time
import numpy as np
import torch
import reference_core as core

ROOT=Path(__file__).resolve().parent
SEEDS=[2026,2027,2028,2029,2030]
COUNTS=[1280,2560,5120]
ALPHAS={'uniform':0.,'boundary_weak':.3,'boundary_strong':.7,'gauss':None,'gauss_boundary_weak':.3,'gauss_boundary_strong':.7}
PRE_BOUNDARY_GAUSS_HASH='746359f8557096f5e204d014677e0bd7d644973a88eb4a9651f9d75bc7542256'
GAUSS_RULES={1280:(20,64),2560:(20,128),5120:(40,128)}
LEGACY_RANDOM_HASH='5bbabfae615326661b16452a406aa21c17c031b4d8b7ff699f14dd815146faac'
CONFIG=dict(epsilon=.1,forcing='minus_one_minus_sin_pi_y',width=32,depth=3,test_degree=18,
    n_radial=40,n_angular=128,adam_steps=3000,adam_lr=.001,lbfgs_steps=1500,lbfgs_lr=1.,
    validation_degree=22,validation_radial=56,validation_angular=160,eval_points=10000)
def run_path(n,mode,seed): return ROOT/'runs'/f'{mode}_N{n}_seed{seed}'
def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'reference_core.py').read_bytes()).hexdigest()

def random_points(n,mode,seed):
    # Shared uniforms and nested sample prefixes across counts and distributions.
    z=np.random.default_rng(seed+900000).random((max(COUNTS),3))[:n]
    a=ALPHAS[mode]
    s=np.where(z[:,0]<a,z[:,1]**.2,z[:,1])
    angle=2*np.pi*z[:,2]
    xy=np.column_stack((np.sqrt(s)*np.cos(angle),np.sqrt(s)*np.sin(angle)))
    density=(1-a)+5*a*s**4
    weights=np.pi/(n*density)
    return xy,weights

def mapped_gauss_points(n,alpha):
    """Gauss in t=F(s), F(s)=(1-alpha)*s+alpha*s**5; ds=dt/q(s)."""
    nr,na=GAUSS_RULES[n]
    z,w=np.polynomial.legendre.leggauss(nr)
    t=(z+1)/2; w=w/2
    lo=np.zeros_like(t);hi=np.ones_like(t)
    for _ in range(60):
        mid=(lo+hi)/2
        below=(1-alpha)*mid+alpha*mid**5<t
        lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    s=(lo+hi)/2
    q=(1-alpha)+5*alpha*s**4
    angle=2*np.pi*np.arange(na)/na
    xy=np.column_stack(((np.sqrt(s)[:,None]*np.cos(angle)).ravel(),
                        (np.sqrt(s)[:,None]*np.sin(angle)).ravel()))
    weights=np.repeat(np.pi*w/(na*q),na)
    return xy,weights

def make_random_rule(n,mode,seed):
    if mode=='gauss':
        nr,na=GAUSS_RULES[n]
        return core.make_test_rule(CONFIG['test_degree'],nr,na,CONFIG['forcing'])
    xy,w=mapped_gauss_points(n,ALPHAS[mode]) if mode.startswith("gauss_") else random_points(n,mode,seed)
    psi,dx,dy,modes=core.disk_tests(xy,CONFIG['test_degree'])
    G=psi.T@(w[:,None]*psi)+dx.T@(w[:,None]*dx)+dy.T@(w[:,None]*dy)
    G=(G+G.T)/2
    T=lambda a:torch.as_tensor(np.ascontiguousarray(a),dtype=torch.float64)
    points=T(xy); Wpsi=T((w[:,None]*psi).T)
    return SimpleNamespace(xy=points,weights=w,psi=psi,dx=dx,dy=dy,modes=modes,
        Wpsi=Wpsi,Wdx=T((w[:,None]*dx).T),Wdy=T((w[:,None]*dy).T),
        load=Wpsi@core.source_tensor(points,CONFIG['forcing']),chol=T(np.linalg.cholesky(G)),gram=G,
        condition=float(np.linalg.cond(G)),degree=18,
        n_radial=GAUSS_RULES[n][0] if mode.startswith("gauss") else 0,
        n_angular=GAUSS_RULES[n][1] if mode.startswith("gauss") else 0)

_CACHE=None
def validation_cache():
    global _CACHE
    if _CACHE is None:
        ref=core.P3VectorReference(ROOT/'reference/p3_vector_burgers_sine_eps0p1_h0p02.mat',.1)
        qxy,qref,w=ref.quadrature(6)
        same=core.make_test_rule(18,56,160,CONFIG['forcing'])
        rich=core.make_test_rule(22,56,160,CONFIG['forcing'])
        xy=core.sample_disk(np.random.default_rng(314159),10000)
        _CACHE=(ref,qxy,qref,w,same,rich,xy)
    return _CACHE

def run_one(task):
    n,mode,seed=task; out=run_path(*task);out.mkdir(parents=True,exist_ok=True)
    fingerprint=code_hash()
    if (out/'complete.json').exists():
        done=json.loads((out/'complete.json').read_text())
        if done['code_sha256']==fingerprint: return done['result']
        if mode in ('uniform','boundary_weak','boundary_strong','gauss') and done['code_sha256']==PRE_BOUNDARY_GAUSS_HASH:
            original=(ROOT/'study_before_boundary_gauss.py').read_bytes()
            assert hashlib.sha256(original+(ROOT/'reference_core.py').read_bytes()).hexdigest()==PRE_BOUNDARY_GAUSS_HASH
            return done['result']
        if mode in ('uniform','boundary_weak','boundary_strong') and done['code_sha256']==LEGACY_RANDOM_HASH:
            original=(ROOT/'study_random_original.py').read_bytes()
            assert hashlib.sha256(original+(ROOT/'reference_core.py').read_bytes()).hexdigest()==LEGACY_RANDOM_HASH
            return done['result']
        raise RuntimeError(f'Existing run has different code: {out}')
    started=time.perf_counter()
    with (out/'execution.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log):
        rule=make_random_rule(n,mode,seed)
        config={**CONFIG,'seed':seed,'sample_count':n,'distribution':mode,'alpha':ALPHAS[mode],
            'sampling_seed':seed+900000 if not mode.startswith('gauss') else None,'validation_seed':314159,'method':'VPINN','problem_id':'steady_vector_burgers_2d'}
        config['actual_quadrature_shape']=list(GAUSS_RULES[n]) if mode.startswith('gauss') else None
        torch.manual_seed(seed);np.random.seed(seed)
        model=core.VectorDiskNet(32,3).double()
        init_hash=hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in model.parameters())).hexdigest()
        history,stats=core.train_vpinn(model,rule,config)
        model.eval()
        ref,qxy,qref,w,same,rich,xy=validation_cache()
        pred=core.predict(model,qxy)
        errs=core.vector_errors(pred,qref,w)
        residual,div=core.independent_residual(model,xy,.1,config['forcing'])
        rtrain=core.weak_residual(model,rule,.1,False).detach()
        rsame=core.weak_residual(model,same,.1,False).detach()
        rrich=core.weak_residual(model,rich,.1,False).detach()
        boundary=core.predict(model,core.circle_points(1024))
        result={'N':n,'distribution':mode,'seed':seed,'relative_l2':errs['vector']['relative_l2'],
            'relative_linf':errs['vector']['sampled_relative_linf'],
            'u_relative_l2':errs['u']['relative_l2'],'v_relative_l2':errs['v']['relative_l2'],
            'strong_rms':float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
            'strong_max':float(np.linalg.norm(residual,axis=1).max()),
            'boundary_max':float(np.linalg.norm(boundary,axis=1).max()),
            'train_loss':float(core.weak_loss(rtrain,rule)),
            'independent_weak':float(core.weak_loss(rsame,same)),
            'enriched_weak':float(core.weak_loss(rrich,rich)),
            'moment_difference':float(torch.sqrt(core.weak_loss(rtrain-rsame,same))),
            'train_seconds':stats['train_seconds'],'gram_condition':rule.condition,
            'weight_sum':float(rule.weights.sum()),'effective_sample_size':float(rule.weights.sum()**2/(rule.weights**2).sum()),
            'boundary_fraction_r_gt_09':float((np.linalg.norm(rule.xy.numpy(),axis=1)>.9).mean()),
            'lbfgs_iterations':stats['lbfgs_iterations'],'lbfgs_closures':stats['lbfgs_closures'],
            'initial_state_sha256':init_hash,'total_seconds':time.perf_counter()-started}
        for v in result.values():
            if isinstance(v,float): assert np.isfinite(v)
        torch.save({'state_dict':model.state_dict(),'config':config},out/'model.pt')
        np.savez_compressed(out/'points.npz',xy=rule.xy.numpy(),weights=rule.weights)
        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
        (out/'statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
        (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        with (out/'history.csv').open('w',newline='',encoding='utf-8-sig') as f:
            wr=csv.writer(f);wr.writerow(['phase','step_or_closure','weak_loss','seconds']);wr.writerows(history)
        (out/'complete.json').write_text(json.dumps({'code_sha256':fingerprint,'result':result},indent=2),encoding='utf8')
    return result

def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('*/metrics.json'))]
    if not records: return []
    def write_csv(name,rows):
        with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    write_csv('results.csv',records)
    metrics=['relative_l2','relative_linf','u_relative_l2','v_relative_l2','strong_rms','train_loss','independent_weak','enriched_weak','train_seconds','gram_condition','effective_sample_size']
    rows=[]
    for n in COUNTS:
      for mode in ALPHAS:
        g=[r for r in records if r['N']==n and r['distribution']==mode]
        if not g: continue
        row={'N':int(n),'distribution':mode,'n_seeds':len(g)}
        for key in metrics:
            v=np.array([r[key] for r in g])
            row[key+'_mean']=float(v.mean());row[key+'_se']=float(v.std(ddof=1)/np.sqrt(len(g)))
        rows.append(row)
    summary=rows;write_csv('summary.csv',rows)
    paired=[]
    for n in COUNTS:
        b={r['seed']:r for r in records if r['N']==n and r['distribution']=='uniform'}
        for mode in [m for m in ALPHAS if m!='uniform']:
            a={r['seed']:r for r in records if r['N']==n and r['distribution']==mode}
            d=np.array([a[s]['relative_l2']-b[s]['relative_l2'] for s in sorted(a.keys()&b.keys())])
            if len(d): paired.append({'N':n,'comparison':mode+' - uniform','n_pairs':len(d),'difference_mean':float(d.mean()),'difference_se':float(d.std(ddof=1)/np.sqrt(len(d)))})
    if paired: write_csv('paired_differences.csv',paired)
    (ROOT/'figures').mkdir(exist_ok=True)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(metric,title,factor) in zip(axs.ravel(),[('relative_l2','P3 relative L2 (%)',100),('strong_rms','Independent strong residual RMS',1),('independent_weak','Independent weak loss',1),('train_seconds','Training seconds (concurrent execution)',1)]):
        for mode,color in zip(ALPHAS,['#245d86','#be772a','#804283','#23824e','#d34f35','#555555']):
            g=sorted([r for r in summary if r['distribution']==mode],key=lambda r:r['N'])
            ax.errorbar([r['N'] for r in g],[r[metric+'_mean']*factor for r in g],yerr=[r[metric+'_se']*factor for r in g],marker='o',capsize=4,label=mode,color=color)
        ax.set(xlabel='Integration points N',ylabel=title,title=title,xticks=COUNTS);ax.grid(alpha=.2)
    for ax in axs.ravel()[:3]: ax.set_yscale('log')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Five paired seeds: mean ± standard error (SD / sqrt(5))',fontsize=14)
    fig.savefig(ROOT/'figures/aggregate.png',dpi=150);plt.close(fig)
    return summary

def run_all(workers=3):
    tasks=[(n,mode,seed) for n in COUNTS for mode in ALPHAS for seed in SEEDS]
    # Pilot sanity checks do not train or change experimental conditions.
    for mode in ALPHAS:
        r=make_random_rule(1280,mode,2026)
        assert len(r.xy)==1280 and np.all(r.weights>0)
        assert np.isfinite(r.gram).all()
        if mode.startswith('gauss'): continue
        a,w=random_points(1280,mode,2026);b,_=random_points(5120,mode,2026)
        np.testing.assert_array_equal(a,b[:1280])
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_one,t):t for t in tasks}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            r=f.result();print(f"{i}/{len(tasks)} {futures[f]} L2={r['relative_l2']:.6g} strong={r['strong_rms']:.6g}",flush=True)
    summary=summarize()
    assert len(summary)==len(COUNTS)*len(ALPHAS) and all(r['n_seeds']==5 for r in summary)
    return summary

if __name__=='__main__':
    run_all()

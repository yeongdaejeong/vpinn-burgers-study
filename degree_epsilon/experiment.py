"""3 test degrees x 3 epsilon values x 5 paired seeds; original Gauss 5120 fixed."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,csv,hashlib,contextlib,concurrent.futures
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(BASE))
import reference_core as core
import study
DEGREES=[18,22,26];EPSILONS=[.1,.05,.02];SEEDS=study.SEEDS
def folder(eps,p,seed):return ROOT/'runs'/f'eps{eps:g}_p{p}_seed{seed}'
def fingerprint():return hashlib.sha256(Path(__file__).read_bytes()+(BASE/'reference_core.py').read_bytes()).hexdigest()
def train(task):
    eps,p,seed=task;out=folder(*task);out.mkdir(parents=True,exist_ok=True)
    if (out/'trained.json').exists():
        d=json.loads((out/'trained.json').read_text());assert d['code_hash']==fingerprint();return task
    config={**study.CONFIG,'epsilon':eps,'test_degree':p,'seed':seed,'sample_count':5120,'distribution':'gauss',
            'validation_degree':30,'validation_radial':56,'validation_angular':160}
    torch.manual_seed(seed);np.random.seed(seed)
    model=core.VectorDiskNet(32,3).double()
    init_hash=hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in model.parameters())).hexdigest()
    old=study.run_path(5120,'gauss',seed)
    assert init_hash==json.loads((old/'metrics.json').read_text())['initial_state_sha256']
    rule=core.make_test_rule(p,40,128,config['forcing'])
    original=np.load(old/'points.npz')
    np.testing.assert_array_equal(rule.xy.numpy(),original['xy']);np.testing.assert_array_equal(rule.weights,original['weights'])
    if eps==.1 and p==18:
        model_path=old/'model.pt';stats=json.loads((old/'statistics.json').read_text());reused=True
    else:
        with (out/'training.log').open('w',encoding='utf8') as f,contextlib.redirect_stdout(f):
            history,stats=core.train_vpinn(model,rule,config)
        model_path=out/'model.pt';reused=False
        torch.save({'state_dict':model.state_dict(),'config':config},model_path)
        with (out/'history.csv').open('w',newline='') as f:
            wr=csv.writer(f);wr.writerow(['phase','step_or_closure','weak_loss','seconds']);wr.writerows(history)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'statistics.json').write_text(json.dumps(stats,indent=2))
    (out/'trained.json').write_text(json.dumps({'code_hash':fingerprint(),'model_path':str(model_path.relative_to(BASE)),
        'reused_original':reused,'initial_state_sha256':init_hash,'scalar_test_dimension':len(rule.modes)},indent=2))
    return task

def train_all():
    tasks=[(e,p,s) for e in EPSILONS for p in DEGREES for s in SEEDS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(train,t):t for t in tasks}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            f.result();print('Trained',i,'/45',futures[f],flush=True)

if __name__=='__main__':train_all()

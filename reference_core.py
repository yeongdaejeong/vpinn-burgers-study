from pathlib import Path
from types import SimpleNamespace
import time, json, csv
import numpy as np
import torch
from torch import nn
from scipy.io import loadmat
from scipy.special import eval_jacobi
from numpy.polynomial.legendre import leggauss
torch.set_num_threads(1)


def monomials(xy):
    """Same complete cubic space as P3 reference triangle, in local coordinates."""
    x, y = np.asarray(xy).T
    return np.column_stack((np.ones_like(x), x, y, x*x, x*y, y*y,
                            x**3, x*y*y, x*x*y, y**3))

def triangle_rule(order=6):
    """Positive tensor Gauss/Duffy rule on (0,0),(1,0),(0,1); weights sum to 1/2."""
    if order < 2:
        raise ValueError("Quadrature order must be >= 2")
    s, w = leggauss(order)
    s, w = (s+1)/2, w/2
    a, b = np.meshgrid(s, s, indexing="ij")
    wa, wb = np.meshgrid(w, w, indexing="ij")
    return np.column_stack((a.ravel(), ((1-a)*b).ravel())), (wa*wb*(1-a)).ravel()

class P3VectorReference:
    """MATLAB columns 1:3 are vertices; columns 4:10 can use any nodal ordering.

    All ten nodes must be affine P3 nodes. Curved/isoparametric geometry is rejected.
    Required metadata prevents comparison with a different PDE or forcing vector.
    """
    def __init__(self, path, epsilon, forcing='minus_one_minus_sin_pi_y'):
        self.path = str(Path(path).resolve())
        try:
            data = loadmat(path, squeeze_me=True)
        except NotImplementedError as exc:
            raise ValueError("Save MATLAB reference with '-v7', not '-v7.3'.") from exc
        required = ("p", "t", "U", "epsilon", "problem_id", "advection_form", "forcing", "boundary_value", "relative_algebraic_residual")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"필수 MATLAB 필드가 없습니다: {missing}. matlab/fem_p3_vector_burgers.m을 사용하십시오.")
        if not np.isclose(float(data["epsilon"]), epsilon, rtol=1e-12, atol=1e-14):
            raise ValueError("FEM 참조해와 VPINN의 epsilon이 다릅니다.")
        if str(data["problem_id"]).strip() != "steady_vector_burgers_2d" or str(data["advection_form"]).strip() != "(u*d_x+v*d_y)[u,v]":
            raise ValueError("Reference must solve the steady 2D vector Burgers equation")
        self.algebraic_residual = float(data["relative_algebraic_residual"])
        if not np.isfinite(self.algebraic_residual) or self.algebraic_residual > 1e-7:
            raise ValueError("The nonlinear FEM reference has not converged sufficiently")
        if not np.array_equal(np.asarray(data["boundary_value"]).ravel(),[0,0]):
            raise ValueError("Both boundary components must be zero")
        if not isinstance(forcing,str) or str(data['forcing']).strip()!=forcing:
            raise ValueError('FEM 참조해의 공간 의존 우변과 VPINN 우변이 다릅니다.')
        if forcing!='minus_one_minus_sin_pi_y':
            raise ValueError('이 노트북은 f=(-1,-sin(pi*y)) 참조해를 사용합니다.')
        self.p = np.asarray(data["p"], dtype=float)
        raw_t = np.atleast_2d(data["t"])
        if raw_t.ndim != 2 or raw_t.shape[1] != 10 or not np.all(raw_t == np.floor(raw_t)):
            raise ValueError("t must contain 10 integer, one-based node IDs per triangle")
        self.t = raw_t.astype(np.int64)-1
        self.u = np.asarray(data["U"], dtype=float)
        if self.u.ndim != 2 or self.u.shape[1] != 2:
            raise ValueError("U must have two vector components")
        if self.p.ndim != 2 or self.p.shape != (len(self.u), 2):
            raise ValueError("p must be N-by-2 and U must contain N nodal values")
        if not (np.isfinite(self.p).all() and np.isfinite(self.u).all()):
            raise ValueError("Non-finite FEM coordinates/solution")
        if self.t.size == 0 or self.t.min() < 0 or self.t.max() >= len(self.p):
            raise ValueError("Invalid MATLAB connectivity")
        if np.max(np.linalg.norm(self.p, axis=1)) > 1+1e-6:
            raise ValueError("Reference mesh extends outside the unit disk")
        vertices = self.p[self.t[:, :3]]
        self.origin = vertices[:, 0]
        self.jac = np.stack((vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]), axis=2)
        self.det = np.abs(np.linalg.det(self.jac))
        if np.any(self.det < 1e-15):
            raise ValueError("Degenerate FEM triangle")
        self.invjac = np.linalg.inv(self.jac)
        local = np.einsum("eij,ekj->eki", self.invjac, self.p[self.t]-self.origin[:, None, :])
        bary = np.concatenate((1-local.sum(axis=2, keepdims=True), local), axis=2)
        if not np.allclose(3*bary, np.round(3*bary), atol=2e-6) or bary.min() < -1e-6:
            raise ValueError("Expected straight-sided affine P3 nodes; curved elements need another evaluator")
        vander = monomials(local.reshape(-1, 2)).reshape(-1, 10, 10)
        self.coefficients = np.linalg.solve(vander, self.u[self.t])
        self.area = float(self.det.sum()/2)
        if not 2.0 < self.area <= np.pi+1e-6:
            raise ValueError("Reference does not appear to cover a polygonal approximation of the unit disk")
        self.fem_times = {k: float(data[k]) for k in ("mesh_seconds", "assembly_seconds", "solve_seconds") if k in data}

    def quadrature(self, order=6):
        local, weights = triangle_rule(order)
        xy = self.origin[:, None, :] + np.einsum("eij,qj->eqi", self.jac, local)
        values = np.einsum("eik,qi->eqk", self.coefficients, monomials(local))
        w = self.det[:, None]*weights[None, :]
        return xy.reshape(-1, 2), values.reshape(-1,2), w.ravel()

    def evaluate(self, xy):
        """Evaluate the full P3 polynomial. Points outside the FEM mesh return NaN."""
        import matplotlib.tri as mtri
        xy = np.asarray(xy)
        # TriFinder uses vertices only; unused P3 nodes are excluded explicitly.
        ids, inverse = np.unique(self.t[:, :3], return_inverse=True)
        tri = mtri.Triangulation(self.p[ids, 0], self.p[ids, 1], inverse.reshape(-1, 3))
        element = tri.get_trifinder()(xy[:, 0], xy[:, 1])
        inside = element >= 0
        out = np.full((len(xy),2), np.nan)
        e = element[inside]
        local = np.einsum("eij,ej->ei", self.invjac[e], xy[inside]-self.origin[e])
        out[inside] = np.einsum("ni,nik->nk",monomials(local),self.coefficients[e])
        return out

def vector_errors(prediction,reference,weights):
    p=np.asarray(prediction);r=np.asarray(reference);w=np.asarray(weights)
    def metric(diff,ref):
        e2=np.sum(diff*diff,axis=-1);r2=np.sum(ref*ref,axis=-1)
        l2=float(np.sqrt(w@e2));rl2=float(np.sqrt(w@r2))
        linf=float(np.sqrt(e2.max()));rlinf=float(np.sqrt(r2.max()))
        return {'absolute_l2':l2,'reference_l2':rl2,'relative_l2':l2/rl2 if rl2>0 else None,
                'sampled_absolute_linf':linf,'sampled_reference_linf':rlinf,
                'sampled_relative_linf':linf/rlinf if rlinf>0 else None}
    return {'vector':metric(p-r,r),'u':metric(p[:,0:1]-r[:,0:1],r[:,0:1]),
            'v':metric(p[:,1:2]-r[:,1:2],r[:,1:2])}

class VectorDiskNet(nn.Module):
    """공유 은닉층과 독립된 두 출력. u=v를 구조적으로 강제하지 않습니다."""
    def __init__(self, width=32, depth=3):
        super().__init__()
        layers=[]
        for i in range(depth):
            layer=nn.Linear(2 if i==0 else width,width)
            nn.init.xavier_normal_(layer.weight);nn.init.zeros_(layer.bias)
            layers.extend((layer,nn.Tanh()))
        layer=nn.Linear(width,2)
        nn.init.xavier_normal_(layer.weight);nn.init.zeros_(layer.bias)
        self.network=nn.Sequential(*layers,layer)

    def forward(self,xy):
        return (1-xy.square().sum(dim=1,keepdim=True))*self.network(xy)

def source_tensor(xy,forcing):
    if callable(forcing): return forcing(xy)
    if isinstance(forcing,str):
        if forcing!='minus_one_minus_sin_pi_y':
            raise ValueError('지원하지 않는 우변 식별자: '+forcing)
        return torch.cat((-torch.ones_like(xy[:,0:1]),-torch.sin(torch.pi*xy[:,1:2])),dim=1)
    return torch.as_tensor(forcing,dtype=xy.dtype,device=xy.device).reshape(1,2).expand(len(xy),2)

def disk_rule(n_radial=32,n_angular=96):
    """s=r^2에서 Gauss 적분, 각도에서 등간격 주기 적분. 총 가중치=pi."""
    if n_radial<2 or n_angular<8: raise ValueError('적분점 수가 너무 작습니다.')
    g,w=leggauss(n_radial);s=(g+1)/2;w=w/2
    theta=2*np.pi*np.arange(n_angular)/n_angular
    r=np.sqrt(s)[:,None]
    xy=np.column_stack(((r*np.cos(theta)).ravel(),(r*np.sin(theta)).ravel()))
    weights=np.repeat(w*np.pi/n_angular,n_angular)
    return xy,weights

def disk_tests(xy,degree):
    """원판 다항식에 (1-r^2)를 곱한 고정 H0^1 시험함수와 해석적 1차 미분."""
    x,y=np.asarray(xy).T;s=x*x+y*y;z=x+1j*y
    values=[];dx=[];dy=[];modes=[]
    for m in range(degree+1):
        H=z**m
        Hx=m*z**(m-1) if m else np.zeros_like(z)
        Hy=1j*Hx
        for k in range((degree-m)//2+1):
            P=eval_jacobi(k,0,m,2*s-1)
            Ps=(k+m+1)*eval_jacobi(k-1,1,m+1,2*s-1) if k else np.zeros_like(s)
            for kind in (('cos',) if m==0 else ('cos','sin')):
                h,hx,hy=(a.real if kind=='cos' else a.imag for a in (H,Hx,Hy))
                values.append((1-s)*h*P)
                dx.append(-2*x*h*P+(1-s)*(hx*P+2*x*h*Ps))
                dy.append(-2*y*h*P+(1-s)*(hy*P+2*y*h*Ps))
                modes.append((m,k,kind))
    return np.column_stack(values),np.column_stack(dx),np.column_stack(dy),modes

def make_test_rule(degree,n_radial,n_angular,forcing):
    xy,weights=disk_rule(n_radial,n_angular)
    psi,dx,dy,modes=disk_tests(xy,degree)
    gram=psi.T@(weights[:,None]*psi)+dx.T@(weights[:,None]*dx)+dy.T@(weights[:,None]*dy)
    gram=(gram+gram.T)/2
    chol=np.linalg.cholesky(gram)
    if not np.isfinite(gram).all(): raise ValueError('유한하지 않은 Gram 행렬')
    T=lambda a:torch.as_tensor(np.ascontiguousarray(a),dtype=torch.float64)
    points=T(xy)
    value_matrix=T((weights[:,None]*psi).T)
    return SimpleNamespace(xy=points,weights=weights,psi=psi,dx=dx,dy=dy,modes=modes,
        Wpsi=value_matrix,Wdx=T((weights[:,None]*dx).T),Wdy=T((weights[:,None]*dy).T),
        load=value_matrix@source_tensor(points,forcing),chol=T(chol),gram=gram,
        condition=float(np.linalg.cond(gram)),degree=degree,n_radial=n_radial,n_angular=n_angular)

def first_derivatives(model,points,training=True):
    xy=points.detach().requires_grad_(True)
    uv=model(xy)
    du=torch.autograd.grad(uv[:,0].sum(),xy,create_graph=training,retain_graph=True)[0]
    dv=torch.autograd.grad(uv[:,1].sum(),xy,create_graph=training,retain_graph=training)[0]
    return uv,du,dv

def weak_residual(model,rule,epsilon,training=True):
    uv,du,dv=first_derivatives(model,rule.xy,training)
    grad_x=torch.stack((du[:,0],dv[:,0]),dim=1)
    grad_y=torch.stack((du[:,1],dv[:,1]),dim=1)
    convection=uv[:,0:1]*grad_x+uv[:,1:2]*grad_y
    return epsilon*(rule.Wdx@grad_x+rule.Wdy@grad_y)+rule.Wpsi@convection-rule.load

def weak_loss(residual,rule):
    """두 잔차 성분의 이산 H1 쌍대 노름 제곱. 역행렬은 만들지 않습니다."""
    whitened=torch.linalg.solve_triangular(rule.chol,residual,upper=False)
    return whitened.square().sum()

def sample_disk(rng,count):
    r=np.sqrt(rng.random(count));angle=2*np.pi*rng.random(count)
    return np.column_stack((r*np.cos(angle),r*np.sin(angle)))

def circle_points(count):
    t=2*np.pi*np.arange(count)/count
    return np.column_stack((np.cos(t),np.sin(t)))

def predict(model,xy,batch=4096):
    out=[]
    with torch.no_grad():
        for i in range(0,len(xy),batch):
            out.append(model(torch.as_tensor(xy[i:i+batch],dtype=torch.float64)).numpy())
    return np.concatenate(out)

def strong_residual(model,points,epsilon,forcing):
    """독립 진단용입니다. 학습 손실에는 이 함수의 결과를 사용하지 않습니다."""
    xy=points.detach().requires_grad_(True);uv=model(xy)
    gradients=[];laplacians=[]
    for k in range(2):
        grad=torch.autograd.grad(uv[:,k].sum(),xy,create_graph=True,retain_graph=True)[0]
        xx=torch.autograd.grad(grad[:,0].sum(),xy,retain_graph=True)[0][:,0]
        yy=torch.autograd.grad(grad[:,1].sum(),xy,retain_graph=True)[0][:,1]
        gradients.append(grad);laplacians.append(xx+yy)
    du,dv=gradients
    adv=torch.stack((uv[:,0]*du[:,0]+uv[:,1]*du[:,1],uv[:,0]*dv[:,0]+uv[:,1]*dv[:,1]),dim=1)
    residual=adv-epsilon*torch.stack(laplacians,dim=1)-source_tensor(xy,forcing)
    return residual,du[:,0]+dv[:,1]

def independent_residual(model,xy,epsilon,forcing,batch=2048):
    rs=[];div=[]
    for i in range(0,len(xy),batch):
        r,d=strong_residual(model,torch.as_tensor(xy[i:i+batch],dtype=torch.float64),epsilon,forcing)
        rs.append(r.detach().numpy());div.append(d.detach().numpy())
    return np.concatenate(rs),np.concatenate(div)

def train_vpinn(model,rule,config):
    history=[];start=time.perf_counter()
    opt=torch.optim.Adam(model.parameters(),lr=config['adam_lr'])
    for step in range(1,config['adam_steps']+1):
        opt.zero_grad(set_to_none=True)
        loss=weak_loss(weak_residual(model,rule,config['epsilon']),rule)
        if not torch.isfinite(loss): raise FloatingPointError('유한하지 않은 약형 손실')
        loss.backward();opt.step()
        if step==1 or step%100==0 or step==config['adam_steps']:
            history.append(('Adam',step,float(loss.detach()),time.perf_counter()-start))
            print(f'Adam {step}: 약형 손실={float(loss.detach()):.4e}',flush=True)
    adam_seconds=time.perf_counter()-start;closures=0;iterations=0
    lbfgs_start=time.perf_counter()
    if config['lbfgs_steps']:
        opt=torch.optim.LBFGS(model.parameters(),lr=config['lbfgs_lr'],max_iter=config['lbfgs_steps'],
            max_eval=2*config['lbfgs_steps'],history_size=50,tolerance_grad=1e-10,
            tolerance_change=1e-13,line_search_fn='strong_wolfe')
        def closure():
            nonlocal closures
            opt.zero_grad(set_to_none=True)
            loss=weak_loss(weak_residual(model,rule,config['epsilon']),rule)
            if not torch.isfinite(loss): raise FloatingPointError('유한하지 않은 약형 손실')
            loss.backward();closures+=1
            if closures==1 or closures%100==0:
                history.append(('LBFGS_closure',closures,float(loss.detach()),time.perf_counter()-start))
                print(f'L-BFGS closure {closures}: 약형 손실={float(loss.detach()):.4e}',flush=True)
            return loss
        opt.step(closure)
        iterations=int(opt.state[next(iter(model.parameters()))].get('n_iter',0))
    lbfgs_seconds=time.perf_counter()-lbfgs_start
    final=float(weak_loss(weak_residual(model,rule,config['epsilon'],False),rule).detach())
    history.append(('final',iterations,final,time.perf_counter()-start))
    return history,{'adam_seconds':adam_seconds,'lbfgs_seconds':lbfgs_seconds,
        'train_seconds':time.perf_counter()-start,'adam_updates':config['adam_steps'],
        'lbfgs_iterations':iterations,'lbfgs_closures':closures,'final_weak_loss':final}

def validation_diagnostics(model,rule,config,reference):
    epsilon=config['epsilon'];forcing=config['forcing']
    finer=make_test_rule(rule.degree,config['validation_radial'],config['validation_angular'],forcing)
    enriched=make_test_rule(config['validation_degree'],config['validation_radial'],config['validation_angular'],forcing)
    r0=weak_residual(model,rule,epsilon,False).detach()
    r1=weak_residual(model,finer,epsilon,False).detach()
    r2=weak_residual(model,enriched,epsilon,False).detach()
    rng=np.random.default_rng(config['seed']+1000003)
    xy=sample_disk(rng,config['eval_points']);uv=predict(model,xy)
    residual,divergence=independent_residual(model,xy,epsilon,forcing)
    boundary_xy=circle_points(1024);boundary=predict(model,boundary_xy)
    qxy,qref,w=reference.quadrature(6);qpred=predict(model,qxy)
    metrics={'weak_training_loss_recomputed':float(weak_loss(r0,rule)),
        'weak_same_tests_finer_quadrature_loss':float(weak_loss(r1,finer)),
        'weak_enriched_tests_loss':float(weak_loss(r2,enriched)),
        'weak_quadrature_difference_norm':float(torch.sqrt(weak_loss(r1-r0,rule))),
        'strong_residual_vector_rms':float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
        'strong_residual_vector_sampled_max':float(np.linalg.norm(residual,axis=1).max()),
        'strong_residual_component_rms':np.sqrt(np.mean(residual**2,axis=0)).tolist(),
        'boundary_vector_sampled_max':float(np.linalg.norm(boundary,axis=1).max()),
        'component_difference_rms_diagnostic_only':float(np.sqrt(np.mean((uv[:,0]-uv[:,1])**2))),
        'divergence_rms_diagnostic_only':float(np.sqrt(np.mean(divergence**2))),
        'errors':vector_errors(qpred,qref,w),'comparison_points':len(w),
        'reference_file':reference.path,'reference_polygon_area':reference.area,
        'component_sampled_min':uv.min(axis=0).tolist(),'component_sampled_max':uv.max(axis=0).tolist()}
    data={'xy':xy,'uv':uv,'residual':residual,'divergence':divergence,'boundary_xy':boundary_xy,
          'boundary':boundary,'qxy':qxy,'qref':qref,'qpred':qpred,'weights':w,
          'weak_r_training':r0.numpy(),'weak_r_finer_quad':r1.numpy(),'weak_r_enriched':r2.numpy()}
    return metrics,data

def save_run(output,model,config,history,statistics,metrics,data):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config={**config,'problem_id':'steady_vector_burgers_2d','method':'VPINN',
        'training_objective':'weak residual only; fixed test space; H1 Gram dual norm',
        'reference_used_as_training_labels':False,'torch_version':str(torch.__version__)}
    (output/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
    (output/'metrics.json').write_text(json.dumps({**statistics,**metrics},indent=2,ensure_ascii=False,allow_nan=False))
    torch.save({'state_dict':model.state_dict(),'config':config},output/'model.pt')
    np.savez_compressed(output/'evaluation.npz',**data)
    with (output/'history.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['phase','step_or_closure','weak_loss','seconds']);writer.writerows(history)
    with (output/'errors.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['성분','상대_L2','표본_상대_Linf','절대_L2','표본_절대_Linf'])
        for key,e in metrics['errors'].items():
            writer.writerow([key,e['relative_l2'],e['sampled_relative_linf'],e['absolute_l2'],e['sampled_absolute_linf']])

def load_checkpoint(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    c=checkpoint['config']
    if c.get('problem_id')!='steady_vector_burgers_2d' or c.get('method')!='VPINN':
        raise ValueError('vector Burgers VPINN 체크포인트가 아닙니다.')
    model=VectorDiskNet(c['width'],c['depth']).double()
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,c
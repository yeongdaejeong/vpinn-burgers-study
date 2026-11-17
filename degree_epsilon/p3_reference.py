"""Coupled P3 Newton solver on the original MATLAB meshes, with epsilon continuation."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,time
import numpy as np
from scipy.io import loadmat,savemat
from scipy.sparse import coo_matrix,bmat
from scipy.sparse.linalg import spsolve
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(BASE))
import reference_core as core

class Solver:
    def __init__(self,path):
        self.data=loadmat(path,squeeze_me=True)
        self.p=self.data['p'];self.t=self.data['t'].astype(int)-1
        self.n=len(self.p);self.free=np.setdiff1d(np.arange(self.n),self.data['b'].astype(int)-1)
        self.free2=np.r_[self.free,self.free+self.n]
        v=self.p[self.t[:,:3]];jac=np.stack((v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=2)
        inv=np.linalg.inv(jac);det=np.abs(np.linalg.det(jac))
        nodes=np.einsum('eij,ekj->eki',inv,self.p[self.t]-v[:,None,0])
        coeff=np.linalg.inv(core.monomials(nodes.reshape(-1,2)).reshape(-1,10,10))
        q,w=core.triangle_rule(6);x,y=q.T;z=np.zeros_like(x);o=np.ones_like(x)
        mx=np.column_stack((z,o,z,2*x,y,z,3*x*x,y*y,2*x*y,z))
        my=np.column_stack((z,z,o,z,x,2*y,z,2*x*y,x*x,3*y*y))
        self.V=np.einsum('qi,eij->eqj',core.monomials(q),coeff)
        xi=np.einsum('qi,eij->eqj',mx,coeff);eta=np.einsum('qi,eij->eqj',my,coeff)
        self.Dx=inv[:,None,0,0,None]*xi+inv[:,None,1,0,None]*eta
        self.Dy=inv[:,None,0,1,None]*xi+inv[:,None,1,1,None]*eta
        self.W=det[:,None]*w
        self.I=np.broadcast_to(self.t[:,:,None],(len(self.t),10,10)).ravel()
        self.J=np.broadcast_to(self.t[:,None,:],(len(self.t),10,10)).ravel()
        self.K=self.sparse(self.integrate(self.Dx,self.Dx)+self.integrate(self.Dy,self.Dy))
        xy=v[:,None,0]+np.einsum('eij,qj->eqi',jac,q)
        f=np.stack((-np.ones_like(xy[:,:,0]),-np.sin(np.pi*xy[:,:,1])),axis=-1)
        fe=np.einsum('eqi,eq,eqk->eik',self.V,self.W,f)
        self.F=np.stack([np.bincount(self.t.ravel(),weights=fe[:,:,k].ravel(),minlength=self.n) for k in range(2)]).ravel()
        self.scale=np.linalg.norm(self.F[self.free2])

    def integrate(self,a,b):return np.einsum('eqi,eq,eqj->eij',a,self.W,b,optimize=True)
    def sparse(self,a):return coo_matrix((a.ravel(),(self.I,self.J)),shape=(self.n,self.n)).tocsr()
    def system(self,state,eps,jacobian=True):
        U=state.reshape(2,self.n).T;u=np.einsum('eqi,eik->eqk',self.V,U[self.t])
        dx=np.einsum('eqi,eik->eqk',self.Dx,U[self.t]);dy=np.einsum('eqi,eik->eqk',self.Dy,U[self.t])
        adv=u[:,:,0,None]*dx+u[:,:,1,None]*dy
        local=np.einsum('eqi,eq,eqk->eik',self.V,self.W,adv)
        r=np.stack([eps*(self.K@U[:,k])+np.bincount(self.t.ravel(),weights=local[:,:,k].ravel(),minlength=self.n) for k in range(2)]).ravel()-self.F
        if not jacobian:return r
        advbasis=u[:,:,0,None]*self.Dx+u[:,:,1,None]*self.Dy
        A=eps*self.K+self.sparse(self.integrate(self.V,advbasis+dx[:,:,0,None]*self.V))
        B=self.sparse(self.integrate(self.V,dy[:,:,0,None]*self.V))
        C=self.sparse(self.integrate(self.V,dx[:,:,1,None]*self.V))
        D=eps*self.K+self.sparse(self.integrate(self.V,advbasis+dy[:,:,1,None]*self.V))
        return r,bmat([[A,B],[C,D]],format='csr')

    def check(self):
        original=self.data['U'].T.ravel()
        r=self.system(original,.1,False)
        residual=float(np.linalg.norm(r[self.free2])/self.scale)
        assert residual<1e-8,residual
        rng=np.random.default_rng(721);state=np.zeros(2*self.n);direction=state.copy()
        state[self.free2]=rng.normal(0,.05,len(self.free2));direction[self.free2]=rng.normal(size=len(self.free2))
        _,J=self.system(state,.05);delta=1e-6
        fd=(self.system(state+delta*direction,.05,False)-self.system(state-delta*direction,.05,False))/(2*delta)
        jd=J@direction;error=float(np.linalg.norm((fd-jd)[self.free2])/np.linalg.norm(jd[self.free2]))
        assert error<1e-7,error
        return {'original_matlab_relative_residual':residual,'jacobian_directional_relative_error':error}

    def solve(self,eps,initial):
        state=initial.copy();history=[];start=time.perf_counter()
        for step in range(41):
            r,J=self.system(state,eps);norm=np.linalg.norm(r[self.free2])/self.scale
            print('epsilon',eps,'Newton',step,'residual',norm,flush=True)
            if norm<1e-10:return state,history,time.perf_counter()-start,float(norm)
            if step==40:break
            delta=np.zeros_like(state);delta[self.free2]=spsolve(J[self.free2][:,self.free2].tocsc(),-r[self.free2])
            for k in range(21):
                damping=2.**(-k);trial=state+damping*delta
                nn=np.linalg.norm(self.system(trial,eps,False)[self.free2])/self.scale
                if nn<=(1-1e-4*damping)*norm:break
            else:raise RuntimeError('P3 Newton line search failed')
            history.append([eps,step,float(norm),damping]);state=trial
        raise RuntimeError('P3 Newton convergence failed')

def main():
    (ROOT/'reference').mkdir(exist_ok=True);checks={}
    for h in ['0p04','0p02']:
        print('Mesh',h,flush=True)
        solver=Solver(BASE/f'reference/p3_vector_burgers_sine_eps0p1_h{h}.mat')
        checks[h]=solver.check();print(checks[h],flush=True)
        original=solver.data['U'].T.ravel()
        reproduced,_,_,_=solver.solve(.1,.8*original)
        checks[h]['baseline_nodal_difference']=float(np.max(np.abs(reproduced-original)))
        assert checks[h]['baseline_nodal_difference']<1e-7
        state=original
        for eps in [.075,.05,.035,.02]:
            dest=ROOT/'reference'/f'p3_eps{eps:g}_h{h}.mat'
            if dest.exists():state=loadmat(dest)['U'].T.ravel();continue
            state,hist,elapsed,residual=solver.solve(eps,state)
            data={k:v for k,v in solver.data.items() if not k.startswith('__')}
            data.update(U=state.reshape(2,-1).T,epsilon=eps,relative_algebraic_residual=residual,
                        newton_history=np.asarray(hist),solve_seconds=elapsed,
                        solver='Python P3; same original nodes, connectivity and 6x6 Duffy quadrature',
                        jacobian_check_relative=checks[h]['jacobian_directional_relative_error'])
            savemat(dest,data,do_compression=True)
            core.P3VectorReference(dest,eps)
            print('Saved',dest,flush=True)
        del solver
    (ROOT/'p3_solver_validation.json').write_text(json.dumps(checks,indent=2))
    print('P3 reference generation complete',flush=True)

if __name__=='__main__':main()

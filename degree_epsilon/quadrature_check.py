"""Post-training refinement audit for epsilon .02; no optimizer updates."""
import evaluate as e
import numpy as np
import torch,json

def main():
    models=[(p,s,e.load_study_model(e.folder(.02,p,s))) for p in e.DEGREES for s in e.SEEDS]
    rule=e.c.make_test_rule(30,56,160,'minus_one_minus_sin_pi_y')
    rows=[];previous=None
    for nr,na in [(96,256),(192,512),(384,1024)]:
        xy,w=e.c.disk_rule(nr,na);res=np.zeros((15,496,2))
        for start in range(0,len(xy),4096):
            pts=xy[start:start+4096];weights=w[start:start+4096,None]
            psi,dx,dy,_=e.c.disk_tests(pts,30)
            A=(weights*psi).T;B=(weights*dx).T;C=(weights*dy).T
            points=torch.as_tensor(pts,dtype=torch.float64)
            load=A@e.c.source_tensor(points,'minus_one_minus_sin_pi_y').numpy()
            for i,(p,s,model) in enumerate(models):
                uv,du,dv=e.c.first_derivatives(model,points,False)
                uv=uv.detach().numpy();du=du.detach().numpy();dv=dv.detach().numpy()
                gx=np.column_stack((du[:,0],dv[:,0]));gy=np.column_stack((du[:,1],dv[:,1]))
                res[i]+=.02*(B@gx+C@gy)+A@(uv[:,0,None]*gx+uv[:,1,None]*gy)-load
        for i,(p,s,_) in enumerate(models):
            loss=float(e.c.weak_loss(torch.tensor(res[i]),rule))
            row={'epsilon':.02,'test_degree':p,'seed':s,'radial':nr,'angular':na,'common_p30_weak':loss}
            if previous is not None:
                delta=float(torch.sqrt(e.c.weak_loss(torch.tensor(res[i]-previous[i]),rule)))
                row['residual_difference_norm']=delta
                row['relative_residual_difference']=delta/np.sqrt(loss)
            else:row.update(residual_difference_norm=None,relative_residual_difference=None)
            rows.append(row)
        previous=res
        print('Audited',nr,na,'max relative change',max((r['relative_residual_difference'] or 0) for r in rows[-15:]),flush=True)
    e.write_csv(e.ROOT/'quadrature_refinement.csv',rows)
    (e.ROOT/'quadrature_refinement.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()

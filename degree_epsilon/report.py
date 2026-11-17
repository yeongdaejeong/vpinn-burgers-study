from pathlib import Path
import sys,csv,json,html,hashlib,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
EPS=[.1,.05,.02];P=[18,22,26]

def integrate():
    fragment=ROOT/'report_fragment.html'
    if not fragment.exists():return
    p=BASE/'index.html';s=p.read_text(encoding='utf8')
    import re
    s=re.sub(r'<!-- DEGREE_EPSILON_START -->.*?<!-- DEGREE_EPSILON_END -->','',s,flags=re.S)
    s=s.replace('<section id="limits">','<!-- DEGREE_EPSILON_START -->'+fragment.read_text(encoding='utf8')+'<!-- DEGREE_EPSILON_END --><section id="limits">')
    anchor='<a href="#degree-epsilon">5.1 시험공간 차원과 ε</a>'
    s=re.sub(r'<a href="#degree-epsilon">.*?</a>','',s).replace('<a href="#limits">',anchor+'<a href="#limits">')
    s=s.replace('5개 시드 · 90회 통제 실험<br>','5개 시드 · 배치 90회 + 차원·ε 45회 분석<br>')
    p.write_text(s,encoding='utf8')
    if (BASE/'figures/visual_sampling.png').exists():
        sys.path.insert(0,str(BASE))
        from visual_report import apply
        apply()

def main():
    F=ROOT/'figures';F.mkdir(exist_ok=True)
    rows=list(csv.DictReader((ROOT/'summary.csv').open(encoding='utf-8-sig')))
    refs=json.loads((ROOT/'reference_refinement.json').read_text())
    assert len(rows)==9
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    metrics=[('relative_l2','P3 relative L2 (%)',100),('strong_rms','Independent strong residual RMS',1),
             ('common_p30_weak_96x256','Degree-30 weak loss (96 x 256)',1),('train_seconds','Recorded training seconds',1)]
    for eps,color in zip(EPS,['#245d86','#be772a','#804283']):
        group=sorted([r for r in rows if float(r['epsilon'])==eps],key=lambda r:int(r['test_degree']))
        for ax,(key,title,factor) in zip(axs.ravel(),metrics):
            ax.errorbar([int(r['test_dimension']) for r in group],[float(r[key+'_mean'])*factor for r in group],
                yerr=[float(r[key+'_se'])*factor for r in group],marker='o',capsize=4,label=f'epsilon={eps}',color=color)
            ax.set(title=title,xlabel='Scalar test-space dimension',ylabel=title,xticks=[190,276,378]);ax.grid(alpha=.2)
    for ax in axs.ravel()[:3]:ax.set_yscale('log')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
    fig.suptitle('Fixed Gauss 5120 and optimizer budget | five seeds: mean ± SE')
    fig.savefig(F/'dimension_epsilon.png',dpi=150,bbox_inches='tight');plt.close(fig)
    for eps in EPS:
        data=[np.load(ROOT/'fields'/f'eps{eps:g}_p{p}.npz') for p in P]
        axis=data[0]['axis'];X,Y=np.meshgrid(axis,axis)
        lo=min(np.nanmin(d[k][:,0]) for d in data for k in ['mean','reference'])
        hi=max(np.nanmax(d[k][:,0]) for d in data for k in ['mean','reference'])
        errors=[np.linalg.norm(d['mean']-d['reference'],axis=1) for d in data];emax=max(np.nanmax(x) for x in errors)
        fig,axs=plt.subplots(3,3,figsize=(12,11),layout='constrained')
        for i,(p,d,error) in enumerate(zip(P,data,errors)):
            for j,(value,title) in enumerate([(d['mean'][:,0],'Mean VPINN u'),(d['reference'][:,0],'P3 FEM u'),(error,'Norm of mean vector difference')]):
                im=axs[i,j].pcolormesh(X,Y,value.reshape(X.shape),shading='auto',cmap='viridis' if j<2 else 'magma',vmin=lo if j<2 else 0,vmax=hi if j<2 else emax)
                axs[i,j].set(title=f'p={p}, dim={(p+1)*(p+2)//2}\n'+title,xlabel='x',ylabel='y',aspect='equal')
                fig.colorbar(im,ax=axs[i,j],shrink=.8)
        fig.suptitle(f'epsilon={eps} | 201 x 201 display grid | shared scales within this figure')
        fig.savefig(F/f'fields_eps{eps:g}.png',dpi=130,bbox_inches='tight');plt.close(fig)
    def fmt(r,key,factor=1):return f"{float(r[key+'_mean'])*factor:.5g} ± {float(r[key+'_se'])*factor:.3g}"
    h='<section id="degree-epsilon"><h2>5.1 시험공간 차원과 작은 ε: 추가 통제 실험</h2>'
    h+='<p>기본 Gauss 5,120점(40×128), 시드 2026–2030, 32×3 tanh 네트워크, Adam 3,000회 및 L-BFGS 최대 1,500회를 고정했습니다. 시험 차수 18·22·26과 ε=0.1·0.05·0.02를 교차한 9개 조건, 45개 실행입니다. 기존 기본 Gauss/ε=0.1/p=18의 5개 모델을 재사용하고 40개 모델만 새로 학습했습니다. 기존 90회 배치 실험은 보존했습니다.</p>'
    h+='<p><strong>이번에는 학습 시험공간을 바꿉니다.</strong> 스칼라 시험함수는 190·276·378개, 두 방정식의 잔차 성분은 380·552·756개입니다. 신경망의 함수 공간과 매개변수 수는 그대로입니다. p는 경계 인자를 곱하기 전 다항식 지수이며 FEM 참조해의 P3 차수와 구별합니다. 이 실험은 기본 Gauss에 한정하며 여섯 배치 모두를 다시 실험한 것은 아닙니다.</p>'
    h+='<p>모든 모델을 공통 차수 30의 시험공간(스칼라 496개)에서 평가하고, 56×160 및 96×256 적분 결과를 비교했습니다. 공통 내부점 10,000개, 원주점 1,024개 및 각 ε에 맞는 P3 요소별 적분점도 사용했습니다. 평균 ± SE는 개별 모델 오차 5개의 통계이며 평균 해의 오차와 다릅니다.</p>'
    h+='<div class="scroll"><table><tr><th>ε</th><th>시험 차수</th><th>차원</th><th>P3 상대 L2 (%)</th><th>강형 RMS</th><th>공통 p=30 약형 손실</th><th>추가 적분 검증 차이 노름</th></tr>'
    for r in rows:h+=f"<tr><td>{r['epsilon']}</td><td>{r['test_degree']}</td><td>{r['test_dimension']}</td><td>{fmt(r,'relative_l2',100)}</td><td>{fmt(r,'strong_rms')}</td><td>{fmt(r,'common_p30_weak_96x256')}</td><td>{fmt(r,'quadrature_difference_norm')}</td></tr>"
    h+='</table></div><h3>관측 결과</h3><ul>'
    for eps in EPS:
        group=[r for r in rows if float(r['epsilon'])==eps];best=min(group,key=lambda r:float(r['relative_l2_mean']))
        base=next(r for r in group if r['test_degree']=='18')
        h+=f"<li>ε={eps}: p=18의 상대 L2는 {fmt(base,'relative_l2',100)}%, 관측 평균이 가장 작은 차수는 p={best['test_degree']}로 {fmt(best,'relative_l2',100)}%입니다. 이는 이 예산에서의 평균 순서입니다.</li>"
    h+='</ul><p>차원 증가가 단조로운 개선을 보장하지 않습니다. 더 많은 잔차 성분을 최소화해야 하지만 최적화 예산은 같기 때문입니다. ε 감소는 다른 PDE와 더 얇은 층을 만들 수 있으므로 참조해도 다시 계산했습니다. 학습 손실만으로 우열을 판단하지 않고 P3 차이와 공통 독립 잔차를 함께 읽으십시오.</p>'
    h+='<h3>동일 시드에서 시험공간 확장의 효과</h3><p>같은 ε에서 p=22 또는 26의 상대 L2에서 p=18의 값을 뺀 차이입니다. 음수이면 확대된 시험공간의 오차가 작습니다. SE는 시드별 차이 5개의 표준오차입니다.</p><div class="scroll"><table><tr><th>ε</th><th>비교</th><th>차이 ± SE (%p)</th></tr>'
    for r in csv.DictReader((ROOT/'paired_degree_differences.csv').open(encoding='utf-8-sig')):h+=f"<tr><td>{r['epsilon']}</td><td>{r['comparison']}</td><td>{float(r['difference_pp']):.5g} ± {float(r['se_pp']):.3g}</td></tr>"
    h+='</table></div><h3>P3 참조해의 재계산과 세분화 검사</h3><p>원본과 같은 직선 P3 메시, 결합 비선형 약형, 6×6 Gauss–Duffy 적분을 Python으로 구현했습니다. 기존 ε=0.1 MATLAB 해의 잔차와 Newton Jacobian을 검사하고 ε를 단계적으로 낮추며 참조해를 계산했습니다. 각 ε에서 h=0.04와 0.02를 공통 영역에서 비교했습니다.</p><div class="scroll"><table><tr><th>ε</th><th>P3 세분화 상대 L2 차이 (%)</th><th>미세 Newton 잔차</th><th>공통 면적 비율</th></tr>'
    for r in refs:h+=f"<tr><td>{r['epsilon']}</td><td>{100*r['relative_l2_difference']:.5g}</td><td>{r['fine_algebraic_residual']:.3g}</td><td>{r['included_area_fraction']:.8f}</td></tr>"
    h+='</table></div><p><strong>참조해의 한계:</strong> 세분화 차이는 정확해 오차 상한이 아닙니다. VPINN과 P3의 차이가 이 변화량과 비슷하거나 작으면 연속 PDE 정확도 개선을 확정하지 않습니다. ε에 따라 안정성 상수와 해의 크기가 달라질 수 있으므로 잔차를 정확해 오차와 동일시하지 않습니다. 경계조건은 모든 경우 구조적으로 부과하며, 경계 오차가 작다고 내부 층이 정확한 것은 아닙니다.</p>'
    h+='<p><a href="degree_epsilon/results.csv">45개 개별 결과</a> · <a href="degree_epsilon/summary.csv">평균·표준오차</a> · <a href="degree_epsilon/reference_refinement.csv">P3 세분화</a> · <a href="degree_epsilon/experiment.py">학습 코드</a> · <a href="degree_epsilon/evaluate.py">독립 평가 코드</a> · <a href="degree_epsilon/p3_reference.py">P3 솔버</a></p>'
    h+='<h3>작은 ε에서 드러난 적분 해상도의 한계</h3><p>ε=0.02에서 차원을 190→276→378로 늘렸을 때 P3 상대 L2 평균은 약 2.91%→4.92%→4.91%였습니다. 반면 독립 강형 RMS 평균은 약 1.79→1.79→1.32였습니다. 잔차 감소가 참조해 오차 감소와 같은 뜻이 아님을 보여 줍니다. 같은 시험공간에서도 학습 적분과 독립 적분의 손실이 크게 달라, 고정된 5,120점 적분과 최적화 예산으로 작은 ε를 충분히 해결했다고 할 수 없습니다. 차원 확대 자체의 일반적인 열세를 뜻하지도 않습니다.</p>'
    audit=json.loads((ROOT/'quadrature_refinement.json').read_text())
    h+='<p>ε=0.02의 15개 저장 모델에 대해 공통 차수 30 잔차를 96×256, 192×512, 384×1024 적분으로 다시 평가했습니다. 아래 변화율은 이전 적분 대비 잔차 벡터 차이의 H1 쌍대 노름을 현재 잔차 노름으로 나눈 값이며, 15개 모델 중 최대입니다. 모든 검사는 재학습 없이 수행했습니다.</p><table><tr><th>검증 적분</th><th>p=18 약형 손실 평균</th><th>p=22 평균</th><th>p=26 평균</th><th>잔차 벡터 변화율 최대</th></tr>'
    for nr,na in [(96,256),(192,512),(384,1024)]:
        group=[r for r in audit if r['radial']==nr]
        means=[np.mean([r['common_p30_weak'] for r in group if r['test_degree']==p]) for p in P]
        change='—' if nr==96 else f"{100*max(r['relative_residual_difference'] for r in group):.4g}%"
        h+=f'<tr><td>{nr}×{na}</td>'+''.join(f'<td>{v:.5g}</td>' for v in means)+f'<td>{change}</td></tr>'
    h+='</table><p>이 세분화 검사는 관측된 적분 안정성을 보여 주며 엄밀한 오차 상한은 아닙니다. 위 전체 비교 표와 곡선의 공통 약형 손실은 96×256 값입니다. <a href="degree_epsilon/quadrature_refinement.csv">추가 적분 검증 전체 결과</a></p>'
    (ROOT/'report_tables.html').write_text(h+'</section>',encoding='utf8')
    image_names=['dimension_epsilon.png']+[f'fields_eps{e:g}.png' for e in EPS]
    h+='<h3>시험공간 확장과 ε 감소의 결과 그림</h3><p>공간 지도는 5개 시드의 평균 u, P3의 u, 두 성분을 함께 사용한 평균 벡터 차이의 크기입니다. 같은 ε의 그림 안에서는 색 범위를 공유합니다. 기록된 학습 시간은 기존 기준 모델과 새 모델의 실행 시점 및 동시 작업 부하가 달라 엄밀한 속도 비교로 해석하지 않습니다.</p>'
    h+=''.join(f'<img class="wide" style="display:block;width:100%;height:auto;margin:24px 0" src="degree_epsilon/figures/{name}" alt="시험공간과 epsilon 비교 {name}">' for name in image_names)
    (ROOT/'report_fragment.html').write_text(h+'</section>',encoding='utf8');integrate()
    n=nbformat.read(BASE/'VPINN_sampling_study.ipynb',4)
    # Preserve the original 70 cells; this extension owns only the following pair.
    n.cells=n.cells[:70]
    n.cells.append(nbformat.v4.new_markdown_cell('## 36. 학습 시험공간 차원과 작은 epsilon 추가 실험\n\n기본 Gauss 5120점을 고정하고 시험 차수 18/22/26(차원 190/276/378), epsilon 0.1/0.05/0.02, 시드 2026–2030을 교차 비교합니다. 45개 결과 중 기존 기준 모델 5개를 재사용했고 40개를 새로 학습했습니다. 아래 표는 공통 차수 30 검증과 각 epsilon에 맞게 재계산한 P3 비교입니다.\n\n전체 재현 명령은 `python degree_epsilon/run_all.py`입니다. 완료 모델을 재사용하며 새 epsilon의 P3 참조해와 세분화 검사도 포함합니다. 학습은 `degree_epsilon/experiment.py`, 독립 평가는 `degree_epsilon/evaluate.py`, P3 솔버는 `degree_epsilon/p3_reference.py`에 있습니다. 기본 해설은 Gauss 기준이며, 랜덤 배치 비교와 구분합니다.'))
    code="from pathlib import Path\nfrom IPython.display import HTML, Image, display\ndisplay(HTML(Path('degree_epsilon/report_tables.html').read_text(encoding='utf8')))\nfor name in "+repr(image_names)+":\n    display(Image(filename=str(Path('degree_epsilon/figures')/name)))"
    from IPython.core.interactiveshell import InteractiveShell
    from unittest.mock import patch
    import os
    # Execute only this display cell, capturing actual MIME output without a kernel connection.
    formatter=InteractiveShell.instance().display_formatter
    outputs=[]
    def capture_display(*objects,**kwargs):
        for obj in objects:
            data,metadata=formatter.format(obj)
            outputs.append(nbformat.v4.new_output('display_data',data=data,metadata=metadata))
    previous=os.getcwd();os.chdir(BASE)
    try:
        with patch('IPython.display.display',capture_display):exec(compile(code,'degree_epsilon_display_cell','exec'),{})
    finally:os.chdir(previous)
    cell=nbformat.v4.new_code_cell(code,execution_count=1,outputs=outputs)
    cell.metadata['execution_note']='Only this display cell executed in Python with IPython MIME capture; prior cells preserved.'
    assert sum('image/png' in o.get('data',{}) for o in cell.outputs)==4
    assert any('text/html' in o.get('data',{}) for o in cell.outputs)
    n.cells.append(cell);nbformat.validate(n);nbformat.write(n,BASE/'VPINN_sampling_study.ipynb')
    saved=json.loads((ROOT/'preservation.json').read_text())
    for name,digest in saved['files'].items():assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest,name
    revision=BASE/'gauss_baseline/validation.json'
    expected=json.loads(revision.read_text())['notebook_first70_sha256'] if revision.exists() else saved['notebook_cells_hash']
    assert hashlib.sha256(json.dumps(n.cells[:70],sort_keys=True,ensure_ascii=False).encode()).hexdigest()==expected
    report={'completed_conditions':9,'completed_seed_results':45,'new_training_runs':40,'reused_baseline_runs':5,
            'previous_files_unchanged':len(saved['files']),'prior_70_cells_preserved_during_refresh':True,'baseline_layout':'Gauss first','display_grid':[201,201]}
    (ROOT/'delivery_validation.json').write_text(json.dumps(report,indent=2));print(report,flush=True)

if __name__=='__main__':main()

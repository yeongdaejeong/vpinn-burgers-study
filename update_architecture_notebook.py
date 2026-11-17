"""Build fixed-seed figures and execute only the appended notebook section."""
from pathlib import Path
import json, csv, base64, copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'architecture'

def build():
    rows=list(csv.DictReader((OUT/'results.csv').open(encoding='utf-8-sig')))
    rows=[{k:float(v) for k,v in row.items()} for row in rows]
    assert len(rows)==9 and len({(r['depth'],r['width']) for r in rows})==9
    depths=[2,3,4]; widths=[16,32,64]
    for r in rows:
        d,w=int(r['depth']),int(r['width'])
        assert r['seed']==2026 and r['parameters']==(d-1)*w*w+(d+4)*w+2
        assert all(np.isfinite(v) for v in r.values())
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    measures=[('vector_relative_l2','Vector relative L2 (%)',100),
              ('vector_relative_linf','Vector sampled relative Linf (%)',100),
              ('strong_rms','Independent strong residual RMS',1),
              ('parameters','Trainable parameters',1),
              ('train_loss','Training weak loss',1),
              ('train_seconds','Training seconds (pool: up to 3 workers)',1)]
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,(key,title,factor) in zip(axs.ravel(),measures):
        a=np.array([[next(r[key] for r in rows if r['depth']==d and r['width']==w)*factor for w in widths] for d in depths])
        im=ax.imshow(a,cmap='YlGnBu',aspect='auto')
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f'{a[i,j]:.3g}',ha='center',va='center',color='white' if a[i,j]>(a.min()+a.max())/2 else '#111')
        ax.set(xticks=range(3),xticklabels=widths,yticks=range(3),yticklabels=depths,xlabel='Width (neurons / hidden layer)',ylabel='Depth (hidden layers)',title=title)
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Architecture study | Gauss 5120 | seed 2026 | epsilon 0.1',fontsize=16)
    fig.savefig(figures/'overview.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for i,component in enumerate(['u','v']):
        for j,(metric,title,factor) in enumerate([('relative_l2','Relative L2 (%)',100),('relative_linf','Sampled relative Linf (%)',100),('strong_rms','Strong residual RMS',1)]):
            ax=axs[i,j]
            for d in depths:
                group=sorted([r for r in rows if r['depth']==d],key=lambda r:r['width'])
                ax.plot(widths,[r[component+'_'+metric]*factor for r in group],marker='o',label=f'depth {d}')
            ax.set(title=f'{component}: {title}',xlabel='Width',yscale='log',xticks=widths)
            ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Component accuracy | one fixed seed, no error bars',fontsize=16)
    fig.savefig(figures/'components.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for r in rows:
        d,w=int(r['depth']),int(r['width'])
        hist=list(csv.DictReader((OUT/'runs'/f'd{d}_w{w}_seed2026'/'history.csv').open()))
        for ax,phase in zip(axs,['Adam','LBFGS_closure']):
            h=[v for v in hist if v['phase']==phase]
            ax.semilogy([float(v['step_or_closure']) for v in h],
                        [float(v['weak_loss']) for v in h],label=f'd{d} / w{w}')
            ax.set(xlabel='Adam updates' if phase=='Adam' else 'L-BFGS closure calls',ylabel='Training weak loss',title=phase)
            ax.grid(alpha=.25)
    axs[1].legend(ncol=3,fontsize=8)
    fig.savefig(figures/'training.png',dpi=160);plt.close(fig)
    best=min(rows,key=lambda r:r['vector_relative_l2'])
    baseline=next(r for r in rows if r['depth']==3 and r['width']==32)
    intro=r'''## 37. 신경망 depth와 width: 시드 2026 고정 실험

**depth 2·3·4 × width 16·32·64의 9개 조합**을 새로 학습했습니다. depth는 은닉층 수, width는 각 은닉층의 뉴런 수입니다. 입력은 좌표 2개, 출력은 해의 두 성분이며 모든 은닉층은 tanh를 사용합니다.

기본 구조는 depth 3 / width 32입니다. 매개변수 수는 depth를 $d$, width를 $w$라 할 때
$$P(d,w)=(d-1)w^2+(d+4)w+2$$
입니다. 경계 인자 $(1-x^2-y^2)$에는 학습 매개변수가 없습니다.

**고정 조건:** epsilon 0.1, 기본 Gauss 40×128=5,120점, 시험 차수 18(스칼라 시험함수 190개), Adam 3,000회, L-BFGS 최대 1,500회, CPU float64. 네트워크 생성 직전에 seed 2026을 설정합니다. 구조가 달라지면 가중치 배열의 크기도 달라지므로 같은 seed가 같은 초기 함수를 뜻하지는 않습니다.

**공통 검증:** 동일한 P3 h=0.02 참조해와 요소 적분점, 동일한 독립 내부점 10,000개(seed 314159), 동일 시험공간의 56×160 적분 및 차수 22 확장 검증. 벡터와 u/v 각각의 상대 L2, 표본 상대 Linf, 강형 잔차 RMS를 기록합니다.

이 절은 단일 시드 결과이며 평균·표준오차를 계산하지 않습니다. 같은 반복 상한을 사용하지만 매개변수 수·연산량·선 탐색 호출 수는 다릅니다. 시간은 최대 3개 작업을 동시에 실행한 벽시계 시간으로, 실행별 시스템 부하가 다를 수 있어 독점 실행 성능 비교가 아닙니다.
'''
    execution='''from pathlib import Path
import subprocess, sys
from IPython.display import display, Image, HTML

# False이면 저장된 9개 결과만 읽습니다. 전체 재학습에는 기존 architecture/runs를 별도 보관한 뒤 True로 설정합니다.
RUN_ARCHITECTURE_EXPERIMENTS = False
if RUN_ARCHITECTURE_EXPERIMENTS:
    subprocess.run([sys.executable, "architecture_study.py"], check=True)

display(Image(filename="architecture/figures/overview.png"))
display(Image(filename="architecture/figures/components.png"))
display(Image(filename="architecture/figures/training.png"))
'''
    tablecode='''import csv
architecture_rows = list(csv.DictReader(Path("architecture/results.csv").open(encoding="utf-8-sig")))
columns = [("depth", "Depth", 1), ("width", "Width", 1), ("parameters", "매개변수", 1),
           ("vector_relative_l2", "벡터 L2 (%)", 100), ("vector_relative_linf", "벡터 표본 Linf (%)", 100),
           ("u_relative_l2", "u L2 (%)", 100), ("v_relative_l2", "v L2 (%)", 100),
           ("u_relative_linf", "u 표본 Linf (%)", 100), ("v_relative_linf", "v 표본 Linf (%)", 100),
           ("strong_rms", "강형 RMS", 1), ("independent_weak", "독립 약형 손실", 1),
           ("enriched_weak", "확장 약형 손실", 1), ("train_seconds", "학습 초", 1)]
table = "<table><thead><tr>" + "".join("<th>"+label+"</th>" for _,label,_ in columns) + "</tr></thead><tbody>"
for row in architecture_rows:
    table += "<tr>" + "".join("<td>"+format(float(row[key])*factor,".5g")+"</td>" for key,_,factor in columns) + "</tr>"
display(HTML(table+"</tbody></table>"))
'''
    conclusion=f'''### 결과 해석

- 이 9개 실행에서 벡터 상대 L2가 가장 작은 구조는 **depth {int(best['depth'])} / width {int(best['width'])}**로, **{100*best['vector_relative_l2']:.5f}%**입니다.
- 기본 depth 3 / width 32는 벡터 상대 L2 **{100*baseline['vector_relative_l2']:.5f}%**, 표본 상대 Linf **{100*baseline['vector_relative_linf']:.5f}%**, 독립 강형 RMS **{baseline['strong_rms']:.6g}**입니다.
- L2가 가장 작은 구조가 표본 Linf 또는 강형 잔차에서도 가장 작다고 가정하지 않습니다. 각 지표와 u/v 결과를 따로 비교합니다.
- 동일 시드·학습 예산에서 관찰한 구조 민감도입니다. 더 깊거나 넓으면 항상 정확해진다는 결론이나 여러 초기화에서의 평균 성능 순위는 이 실험만으로 주장하지 않습니다. P3 차이는 정확해 오차의 인증값이 아니며 표본 Linf는 연속 영역 최대값이 아닙니다.

### 코드와 저장 파일

`architecture_study.py`의 `run_one`은 `torch.manual_seed(2026)` 이후 `VectorDiskNet(width, depth)`를 생성하고 기존 `train_vpinn`을 그대로 호출합니다. `validation_cache()`의 공통 평가 데이터를 모든 구조에 사용합니다. `run_all`은 9개 조합을 실행하며 설정과 코드 해시가 일치하는 완료 실행만 재사용합니다.

각 실행의 `architecture/runs/d*_w*_seed2026/`에는 모델, 설정, 적분점·가중치, 학습 이력, 실제 최적화 횟수, 검증 지표가 있습니다. 전체 수치는 [architecture/results.csv](architecture/results.csv)에 있습니다. 재현 명령은 `python architecture_study.py`입니다.
'''
    cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_code_cell(execution),nbformat.v4.new_code_cell(tablecode),nbformat.v4.new_markdown_cell(conclusion)]
    small=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    # Execute the appended display cells in a fresh in-process IPython namespace.
    # This does not require opening a Jupyter kernel network port.
    import os
    os.chdir(ROOT)
    shell=InteractiveShell.instance()
    count=0
    for cell in small.cells:
        if cell.cell_type!='code': continue
        count+=1
        with capture_output() as captured:
            result=shell.run_cell(cell.source)
        result.raise_error()
        cell.execution_count=count
        cell.outputs=[]
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
    for cell in small.cells:
        cell.metadata['architecture_study']=True
    # Preserve every pre-existing source cell and output, including public-path sanitization.
    paths=[ROOT/'VPINN_sampling_study.ipynb']
    public=ROOT.parent/'vpinn-burgers-study/VPINN_sampling_study.ipynb'
    if public.exists() and public.resolve()!=paths[0].resolve(): paths.append(public)
    for path in paths:
        nb=nbformat.read(path,as_version=4)
        old=[c for c in nb.cells if not c.metadata.get('architecture_study')]
        nb.cells=old+copy.deepcopy(small.cells)
        nbformat.validate(nb)
        nbformat.write(nb,path)
        saved=nbformat.read(path,as_version=4)
        assert saved.cells[:len(old)]==old
    print(json.dumps({'runs':len(rows),'best_l2':best,'baseline':baseline},indent=2))

if __name__=='__main__':build()

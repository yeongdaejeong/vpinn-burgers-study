"""Reproduce the extension; --postprocess requires all trained checkpoints already present."""
from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
def main():
    scripts=[]
    if '--postprocess' not in sys.argv:
        scripts=['p3_reference.py','experiment.py']
    else:
        assert len(list((ROOT/'runs').glob('*/trained.json')))==45
        assert (ROOT/'p3_solver_validation.json').exists()
    scripts+=['evaluate.py','quadrature_check.py','report.py','validate.py']
    for script in scripts:
        print('START',script,flush=True)
        with (ROOT/(script+'.run.log')).open('w',encoding='utf8') as f:
            subprocess.run([sys.executable,'-u','-X','utf8',str(ROOT/script)],cwd=BASE,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('END',script,flush=True)
    subprocess.run([sys.executable,'-X','utf8',str(BASE/'delivery_check.py')],cwd=BASE,check=True)
    print('Degree and epsilon extension completed.',flush=True)
if __name__=='__main__':main()

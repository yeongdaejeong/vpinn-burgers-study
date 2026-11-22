"""Render the preceding chat answer verbatim; only adapt markup and math typesetting."""
from pathlib import Path
import re, html
import mistune
from bs4 import BeautifulSoup
from latex2mathml.converter import convert
ROOT=Path(__file__).resolve().parent
STYLE="""#qa h3{margin-top:32px;line-height:1.6}#qa p{max-width:950px}#qa .qa-equation{overflow-x:auto;padding:16px;background:#f1f5f7;border-radius:6px;margin:20px 0}#qa math[display=block]{margin:8px auto;font-size:1.2em}#qa math[display=inline]{display:inline-block;margin:0 .1em;font-size:1.05em;vertical-align:middle}#qa math[display=inline] mi{font-style:normal}#qa pre{background:#152839;color:#e7eef4;padding:18px;line-height:1.7;max-height:none}#qa pre code{background:none;color:inherit}#qa .qa-comment{color:#83d9b2}#qa .qa-key{display:block;background:#fff0b8;color:#172533}#qa table td,#qa table th{white-space:normal}#qa hr{border:0;border-top:1px solid #d9dce0;margin:28px 0}@media(max-width:600px){#qa math[display=block]{font-size:1.05em}}"""

def rendered_answer():
    source=(ROOT/'qa_conversation.md').read_text(encoding='utf8')
    equations=[]
    def replace(m):
        block=m.group(1) is not None
        tex=(m.group(1) if block else m.group(2)).strip()
        math=convert(tex,display='block' if block else 'inline')
        parsed=BeautifulSoup(math,'html.parser');parsed.math['data-latex']=tex
        tag=str(parsed.math)
        if block:tag='<div class="qa-equation">'+tag+'</div>'
        key=f'QAMATHPLACEHOLDER{len(equations):03d}END'
        equations.append((key,tag))
        return key
    marked=re.sub(r'\\\[(.*?)\\\]|\\\((.*?)\\\)',replace,source,flags=re.S)
    content=mistune.create_markdown(escape=False,plugins=['table'])(marked)
    for key,value in equations:
        content=content.replace('<p>'+key+'</p>',value).replace(key,value)
    soup=BeautifulSoup(content,'html.parser')
    for para in list(soup.find_all('p')):
        if len(para.contents)==1 and para.strong and re.match(r'^[1-6]\. ',para.get_text()):
            para.name='h3';para.strong.unwrap()
    for pre in soup.find_all('pre'):
        original=pre.code.get_text();lines=[]
        for line in original.splitlines():
            code,sep,comment=line.partition('#')
            marked=html.escape(code)+(('<span class="qa-comment">'+html.escape(sep+comment)+'</span>') if sep else '')
            if line.startswith(('loss.backward','optimizer.step')):marked='<span class="qa-key">'+marked+'</span>'
            lines.append(marked)
        pre.code.clear();pre.code.append(BeautifulSoup('\n'.join(lines)+'\n','html.parser'))
        assert pre.code.get_text()==original
    assert len(soup.find_all('h3'))==6
    assert len(soup.find_all('table'))==3
    assert len(soup.find_all('pre'))==2
    assert len(soup.find_all('math'))==len(equations)
    return str(soup)

def apply(soup):
    old=soup.find(id='qa')
    if old:old.decompose()
    sec=soup.find('section',id='reproduce')
    sec.find('h2').string='10. 노트북과 원자료'
    for summary in sec.find_all('summary'):summary.string='10. 노트북과 원자료 — 상세 보기'
    qa=BeautifulSoup('<section id="qa"><h2>9. Q&amp;A — 제조해와 신경망 학습</h2>'+rendered_answer()+'</section>','html.parser').section
    sec.insert_before(qa)
    for a in soup.select('nav a[href="#qa"]'):a.decompose()
    link=soup.select_one('nav a[href="#reproduce"]');link.string='10. 노트북과 원자료'
    a=soup.new_tag('a',href='#qa');a.string='9. Q&A — 제조해와 신경망 학습';link.insert_before(a)
    old=soup.find('style',id='qa-style')
    if old:old.decompose()
    style=soup.new_tag('style',id='qa-style');style.string=STYLE;soup.head.append(style)

if __name__=='__main__':
    paths=[ROOT/'index.html']
    public=ROOT.parent/'vpinn-burgers-study/index.html'
    if public.exists() and public.resolve()!=paths[0].resolve():paths.append(public)
    for p in paths:
        soup=BeautifulSoup(p.read_text(encoding='utf8'),'html.parser');apply(soup)
        p.write_text(str(soup),encoding='utf8')
        print(p.parent.name, '6 original subsections, 3 tables, 2 unchanged code blocks,',len(soup.select('#qa math')),'equations')

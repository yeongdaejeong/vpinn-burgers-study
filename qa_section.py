"""Append the mathematical Q&A before source files; preserve existing report data."""
from pathlib import Path
from bs4 import BeautifulSoup

HTML = r'''<section id="qa">
<h2>9. Q&amp;A — 제조해와 신경망 학습</h2>
<p class="visual-lead">제조해는 잔차 계산 코드를 검사합니다. 실제 학습은 고정된 PDE에서 신경망 가중치를 바꾸어 해를 찾습니다.</p>
<div class="visual-cards"><div><strong>① 제조해로 검사</strong><p>답을 아는 함수와 그 함수에 맞춘 우변 → 예상한 잔차와 코드 결과 비교</p></div><div><strong>② 가중치를 학습</strong><p>실제 우변은 고정 → 후보 함수의 약형 잔차 계산 → 가중치 갱신</p></div><div><strong>③ 결과를 검증</strong><p>독립 적분·강형 잔차·P3 참조해로 정확도 확인</p></div></div>
<article id="qa-manufactured"><h3>Q1. 제조해로 정확히 무엇을 검사하나요?</h3>
<p><strong>손으로 계산한 답과 코드가 계산한 답이 일치하는지 검사합니다.</strong> 예를 들어 다음 함수를 먼저 정합니다.</p>
<div class="qa-equation"><math display="block"><msub><mi>u</mi><mo>*</mo></msub><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mi>x</mi><mo>(</mo><mn>1</mn><mo>−</mo><mi>x</mi><mo>)</mo><mo>,</mo><mspace width="1em"/><mo>−</mo><msubsup><mi>u</mi><mo>*</mo><mo>″</mo></msubsup><mo>=</mo><mn>2</mn><mo>=</mo><msub><mi>f</mi><mo>*</mo></msub></math></div>
<p>이 함수는 구간 (0,1)의 양 끝에서 0입니다. 손으로 계산하면 2차 미분은 −2이므로, 검증용 우변을 2로 정합니다. 정상 코드의 잔차는 −(−2)−2=0입니다. 확산항 부호를 반대로 구현하면 −2−2=−4가 되어 오류가 드러납니다. 이 단계에서는 최적화나 가중치 갱신을 하지 않습니다.</p>
<details class="qa-detail"><summary>현재 Burgers 노트북의 네 가지 검사</summary><ol><li>실제 우변: y=−0.5, 0, 0.5에서 두 번째 우변이 1, 0, −1인지 확인합니다.</li><li>강형 잔차: 제조해와 제조 우변으로 두 검사점에서 잔차가 0인지 확인합니다(절대 허용오차 10⁻¹²).</li><li>약형 잔차: 시험 차수 8, Gauss 20×64점에서 모든 잔차 성분의 절댓값이 2×10⁻¹¹ 미만인지 확인합니다.</li><li>강형·약형 일치: 같은 제조 함수에 실제 우변을 사용해, 직접 조립한 약형 잔차와 강형 잔차를 시험함수에 곱해 적분한 값이 일치하는지 확인합니다. 이때 잔차는 0일 필요가 없습니다.</li></ol><p>공간 미분, 대류의 성분 결합, 확산항 부호, 하중과 수치적분 조립을 함께 검사합니다. 이 검사만으로 모든 함수에서의 구현, 가중치 역전파, 최적화 성공까지 증명되지는 않습니다.</p></details></article>
<article id="qa-two"><h3>Q2. 왜 v=2u인가요? 3u나 다른 함수는 안 되나요?</h3><p>2라는 숫자에 특별한 의미는 없습니다. 원판에서 다음 함수는 경계조건을 만족하고 두 성분이 달라서 간단한 검증에 유용합니다.</p>
<div class="qa-equation"><math display="block"><mi>w</mi><mo>=</mo><mn>1</mn><mo>−</mo><msup><mi>x</mi><mn>2</mn></msup><mo>−</mo><msup><mi>y</mi><mn>2</mn></msup><mo>,</mo><mspace width="1em"/><msub><mi>u</mi><mo>*</mo></msub><mo>=</mo><mi>w</mi><mo>,</mo><mspace width="1em"/><msub><mi>v</mi><mo>*</mo></msub><mo>=</mo><mi>c</mi><mi>w</mi></math></div>
<div class="qa-equation"><math display="block"><msub><mi>f</mi><mrow><mn>1</mn><mo>,</mo><mo>*</mo></mrow></msub><mo>=</mo><mn>4</mn><mi>ε</mi><mo>−</mo><mn>2</mn><mi>w</mi><mo>(</mo><mi>x</mi><mo>+</mo><mi>c</mi><mi>y</mi><mo>)</mo><mo>,</mo><mspace width="1em"/><msub><mi>f</mi><mrow><mn>2</mn><mo>,</mo><mo>*</mo></mrow></msub><mo>=</mo><mi>c</mi><msub><mi>f</mi><mrow><mn>1</mn><mo>,</mo><mo>*</mo></mrow></msub></math></div>
<p>c=3도 가능합니다. 다만 함수를 바꾸면 제조 우변도 다시 계산해야 합니다. 충분히 매끄럽고 해당 경계조건을 만족하는 다른 함수도 가능합니다. 예를 들어 u=w(1+x), v=w(1+2y)는 서로 비례하지 않아 더 다양한 오류를 검사하는 추가 예시가 됩니다. 이 추가 예시는 현재 실행된 검사와 구분됩니다.</p><p><strong>실제 학습에는 v=2u를 강제하지 않습니다.</strong> 실제 우변은 (−1, −sin(πy))이며, 제조해는 학습 단계마다 바뀌지 않습니다.</p></article>
<article id="qa-theta"><h3>Q3. 가중치를 조정하면 어떻게 원하는 함수를 찾나요?</h3><p>쉬운 예로 −u″=2, u(0)=u(1)=0을 풀어 봅니다. 후보 함수를 u<sub>θ</sub>(x)=θx(1−x)로 두면 모든 θ에서 경계조건은 만족하고, θ가 포물선의 높이를 결정합니다. 잔차와 손실은 다음과 같습니다.</p>
<div class="qa-equation"><math display="block"><msub><mi>r</mi><mi>θ</mi></msub><mo>=</mo><mn>2</mn><mi>θ</mi><mo>−</mo><mn>2</mn><mo>,</mo><mspace width="1em"/><mi>J</mi><mo>(</mo><mi>θ</mi><mo>)</mo><mo>=</mo><msup><mrow><mo>(</mo><mn>2</mn><mi>θ</mi><mo>−</mo><mn>2</mn><mo>)</mo></mrow><mn>2</mn></msup><mo>,</mo><mspace width="1em"/><msup><mi>J</mi><mo>′</mo></msup><mo>(</mo><mi>θ</mi><mo>)</mo><mo>=</mo><mn>8</mn><mo>(</mo><mi>θ</mi><mo>−</mo><mn>1</mn><mo>)</mo></math></div>
<p>학습률 0.1로 θ를 갱신하면 다음과 같습니다. 정답 θ=1을 학습 데이터로 주지 않아도, 방정식의 잔차가 이동 방향을 정합니다.</p>
<div class="scroll"><table><thead><tr><th>현재 θ</th><th>손실의 기울기</th><th>다음 θ</th><th>현재 손실 → 다음 손실</th></tr></thead><tbody><tr><td>0</td><td>−8</td><td>0.8</td><td>4 → 0.16</td></tr><tr><td>0.8</td><td>−1.6</td><td>0.96</td><td>0.16 → 0.0064</td></tr><tr><td>0.96</td><td>−0.32</td><td>0.992</td><td>0.0064 → 0.000256</td></tr></tbody></table></div>
<p>θ=2에서 시작하면 2 → 1.2 → 1.04 → 1.008로 감소합니다. 가중치를 무조건 줄이는 것이 아니라 <strong>손실을 줄이는 방향으로 가중치를 늘리거나 줄입니다.</strong></p></article>
<article id="qa-weak"><h3>Q4. 이 예시를 약형 손실로 학습하면 어떻게 되나요?</h3><p>시험함수 ψ=x(1−x)를 사용하면, 부분적분 후 잔차는 다음과 같습니다.</p>
<div class="qa-equation"><math display="block"><mi>R</mi><mo>(</mo><mi>θ</mi><mo>)</mo><mo>=</mo><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><msubsup><mi>u</mi><mi>θ</mi><mo>′</mo></msubsup><msup><mi>ψ</mi><mo>′</mo></msup><mi mathvariant="normal">d</mi><mi>x</mi><mo>−</mo><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><mn>2</mn><mi>ψ</mi><mi mathvariant="normal">d</mi><mi>x</mi><mo>=</mo><mfrac><mrow><mi>θ</mi><mo>−</mo><mn>1</mn></mrow><mn>3</mn></mfrac></math></div>
<div class="qa-equation"><math display="block"><mi>G</mi><mo>=</mo><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><mo>(</mo><msup><mi>ψ</mi><mn>2</mn></msup><mo>+</mo><msup><mrow><mo>(</mo><msup><mi>ψ</mi><mo>′</mo></msup><mo>)</mo></mrow><mn>2</mn></msup><mo>)</mo><mi mathvariant="normal">d</mi><mi>x</mi><mo>=</mo><mfrac><mn>11</mn><mn>30</mn></mfrac><mo>,</mo><mspace width="1em"/><msub><mi>J</mi><mtext>weak</mtext></msub><mo>=</mo><mfrac><msup><mi>R</mi><mn>2</mn></msup><mi>G</mi></mfrac><mo>=</mo><mfrac><mn>10</mn><mn>33</mn></mfrac><msup><mrow><mo>(</mo><mi>θ</mi><mo>−</mo><mn>1</mn><mo>)</mo></mrow><mn>2</mn></msup></math></div>
<p>이 예시에서는 강형·약형 손실 모두 θ=1에서 최소입니다. 다만 손실의 크기와 기울기가 다르므로 같은 학습률에서 갱신 궤적까지 같지는 않습니다. 실제 VPINN은 여러 시험함수의 잔차를 Gram 행렬로 결합합니다. 약형 잔차는 배열이고, 약형 손실은 그 쌍대 노름의 제곱인 하나의 숫자입니다.</p><p>쌍대 노름은 정규화된 시험함수의 모든 선형결합 중 가장 크게 감지되는 잔차입니다. 개별 잔차 성분의 최대값이나 공간의 점별 PDE 잔차 최대값과는 다릅니다.</p></article>
<article id="qa-network"><h3>Q5. 실제 신경망에서는 θ의 각 성분이 무엇을 바꾸나요?</h3><p>단순 모델의 θ는 높이 하나만 바꿉니다. 신경망의 θ는 모든 층의 가중치와 편향을 모은 것으로, 함수의 높이·변화 위치·가파름을 함께 바꿉니다. 아래는 원리를 설명하는 1차원 작은 모델입니다.</p>
<div class="qa-equation"><math display="block"><msub><mi>u</mi><mi>θ</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mi>x</mi><mo>(</mo><mn>1</mn><mo>−</mo><mi>x</mi><mo>)</mo><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mn>2</mn></munderover><msub><mi>a</mi><mi>j</mi></msub><mi mathvariant="normal">tanh</mi><mo>(</mo><msub><mi>b</mi><mi>j</mi></msub><mi>x</mi><mo>+</mo><msub><mi>c</mi><mi>j</mi></msub><mo>)</mo></math></div>
<p>a는 각 항의 기여 크기, b는 가파름, c는 변화 위치에 영향을 줍니다. 실제 원판 모델은 좌표 (x,y)를 입력받아 두 성분을 출력하고, 경계 인자 1−x²−y²를 곱합니다. 공간 미분은 PDE를 평가하는 데 쓰고, 가중치 미분은 손실을 줄이는 갱신 방향을 정하는 데 씁니다.</p>
<pre class="qa-code"><code><span class="qa-comment"># 현재 후보 함수의 약형 잔차와 손실</span>
R = weak_residual(model, rule, epsilon)
loss = weak_loss(R, rule)

optimizer.zero_grad(set_to_none=True)
<span class="qa-key">loss.backward()  # 각 가중치에 대한 손실의 미분</span>
<span class="qa-key">optimizer.step()  # 가중치와 편향 갱신</span></code></pre>
<p>위 코드는 Adam 단계의 핵심입니다. 현재 구현은 Adam 후 L-BFGS를 사용하며, L-BFGS는 한 갱신에서 손실과 기울기를 여러 번 평가할 수 있습니다. 가중치 학습 중 실제 우변·시험함수·적분점은 고정됩니다.</p></article>
<article id="qa-descent"><h3>Q6. 경사하강법은 상승하는 방향을 처리하지 못하나요?</h3>
<div class="qa-equation"><math display="block"><msubsup><mi>θ</mi><mi>k</mi><mtext>new</mtext></msubsup><mo>=</mo><msub><mi>θ</mi><mi>k</mi></msub><mo>−</mo><mi>η</mi><mfrac><mrow><mo>∂</mo><mi>J</mi></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mi>k</mi></msub></mrow></mfrac></math></div>
<p>기울기는 손실이 증가하는 방향을 알려주므로 그 반대 방향으로 이동합니다. 미분값이 음수이면 가중치를 늘리고, 양수이면 줄입니다. 줄이려는 것은 기울기 자체가 아니라 손실입니다. 학습률이 너무 크면 손실이 오히려 증가할 수도 있습니다.</p><p>다른 의미의 한계는 있습니다. 더 좋은 최솟값에 도달하려면 중간의 손실 장벽을 넘어야 하는 경우, 기본 경사하강법은 이를 보장하지 않습니다. Adam과 L-BFGS도 비선형 문제의 전역 최솟값을 보장하지 않습니다.</p></article>
<article id="qa-target"><h3>Q7. 작은 손실이면 원하는 타겟 함수를 찾았다고 할 수 있나요?</h3><p><strong>PDE·우변·경계조건이 원하는 해를 규정합니다. 제조해가 실제 학습의 타겟으로 들어가는 것은 아닙니다.</strong> 앞의 1차원 예시는 후보 함수 안에 정확해가 있고 손실이 단순해서 정답에 접근합니다. 실제 VPINN에서는 유한한 시험공간, 수치적분 오차, 표현력과 최적화 오차가 남습니다.</p><p>따라서 학습은 정답 함수와의 거리를 직접 줄이는 과정이 아니라, 그 함수가 만족해야 할 조건의 위반을 줄이는 과정입니다. 작은 잔차에서 작은 해 오차를 결론 내리려면 PDE의 안정성·유일성 등도 필요합니다. 독립 적분, 더 풍부한 시험공간, 강형 잔차와 P3 비교가 이를 보완하는 검증 자료입니다.</p></article>
</section>'''
STYLE='''#qa article{padding:20px 0}#qa h3{line-height:1.6}#qa .qa-equation{overflow-x:auto;padding:12px;background:#f1f5f7;border-radius:6px;margin:16px 0}#qa math{margin:8px auto;font-size:1.2em}#qa .qa-detail{padding:16px;background:#f5f7f9;border:1px solid #d9dce0}#qa summary{cursor:pointer;font-weight:600}#qa .qa-code{background:#152839;color:#e7eef4;padding:18px;line-height:1.7}#qa .qa-code code{background:none;color:inherit}#qa .qa-comment{color:#83d9b2}#qa .qa-key{display:block;background:#fff0b8;color:#172533}#qa p{max-width:950px}@media(max-width:600px){#qa math{font-size:1.05em}}'''

def apply(soup):
    old=soup.find(id='qa')
    if old:old.decompose()
    section=soup.find('section',id='reproduce')
    section.find('h2').string='10. 노트북과 원자료'
    for summary in section.find_all('summary'):
        summary.string='10. 노트북과 원자료 — 상세 보기'
    section.insert_before(BeautifulSoup(HTML,'html.parser').section)
    for a in soup.select('nav a[href="#qa"]'):a.decompose()
    link=soup.select_one('nav a[href="#reproduce"]')
    link.string='10. 노트북과 원자료'
    q=soup.new_tag('a',href='#qa');q.string='9. Q&A — 제조해와 신경망 학습';link.insert_before(q)
    old=soup.find('style',id='qa-style')
    if old:old.decompose()
    style=soup.new_tag('style',id='qa-style');style.string=STYLE;soup.head.append(style)

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    for p in [root/'index.html',root.parent/'vpinn-burgers-study/index.html']:
        soup=BeautifulSoup(p.read_text(encoding='utf8'),'html.parser');apply(soup)
        p.write_text(str(soup),encoding='utf8')
        assert [s.get('id') for s in soup.select('main > section')][-2:]==['qa','reproduce']
        print(p.name, 'Q&A added;',len(soup.select('#qa article')),'questions')

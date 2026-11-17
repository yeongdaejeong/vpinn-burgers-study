'use strict';
const D=window.FIELD_DATA;
const byId=id=>document.getElementById(id);
const labels={uniform:'균등 랜덤',boundary_weak:'약한 경계 강조',boundary_strong:'강한 경계 강조',gauss:'기본 Gauss',gauss_boundary_weak:'Gauss · 약한 경계 강조',gauss_boundary_strong:'Gauss · 강한 경계 강조'};
const valid=x=>x!==null&&Number.isFinite(x);
const fmt=x=>Math.abs(x)<.001&&x!==0?x.toExponential(2):Number(x.toPrecision(4)).toString();
const extent=a=>{const v=a.filter(valid);return [Math.min(...v),Math.max(...v)]};
function drawMap(id,values,palette,limits,cut){
  const canvas=byId(id),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);
  const x0=52,y0=20,size=350,n=D.axis.length,colors=D.palettes[palette];
  const [lo,hi]=limits;const span=hi-lo||1;
  for(let iy=0;iy<n;iy++)for(let ix=0;ix<n;ix++){
    const v=values[iy*n+ix];if(!valid(v)||D.reference[iy*n+ix][0]===null)continue;
    const t=Math.max(0,Math.min(255,Math.round(255*(v-lo)/span))),rgb=colors[t];
    ctx.fillStyle=`rgb(${rgb.map(x=>Math.round(255*x)).join(',')})`;
    ctx.fillRect(x0+ix*size/n,y0+(n-1-iy)*size/n,size/n+.3,size/n+.3);
  }
  ctx.strokeStyle='#596575';ctx.lineWidth=.7;ctx.beginPath();ctx.arc(x0+size/2,y0+size/2,size/2,0,Math.PI*2);ctx.stroke();
  ctx.save();ctx.setLineDash([5,4]);ctx.strokeStyle='#172b3e';ctx.lineWidth=1.2;ctx.beginPath();
  if(cut.orientation==='horizontal'){const y=y0+(1-cut.location)*size/2;ctx.moveTo(x0,y);ctx.lineTo(x0+size,y)}
  else{const x=x0+(1+cut.location)*size/2;ctx.moveTo(x,y0);ctx.lineTo(x,y0+size)}ctx.stroke();ctx.restore();
  ctx.fillStyle='#233244';ctx.font='16px Segoe UI';ctx.textAlign='center';
  for(const t of [-1,0,1]){ctx.fillText(t,x0+(t+1)*size/2,y0+size+20);ctx.fillText(t,x0-23,y0+(1-t)*size/2+4)}
  ctx.fillText('x',x0+size/2,y0+size+40);ctx.fillText('y',18,y0+size/2);
  for(let i=0;i<256;i++){ctx.fillStyle=`rgb(${colors[i].map(x=>Math.round(x*255)).join(',')})`;ctx.fillRect(x0+i*size/256,425,size/256+1,13)}
  ctx.fillStyle='#233244';ctx.textAlign='left';ctx.fillText(fmt(lo),x0,458);ctx.textAlign='right';ctx.fillText(fmt(hi),x0+size,458);
}
function drawSlice(id,t,pred,ref,se,isError){
  const c=byId(id),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='white';ctx.fillRect(0,0,w,h);
  const x0=75,y0=28,pw=w-100,ph=h-95;
  const a=pred.map((v,i)=>valid(v)&&valid(ref[i])?(isError?v-ref[i]:v):null);
  const b=isError?ref.map(v=>valid(v)?0:null):ref;
  let values=a.concat(b);if(se)values=values.concat(a.map((v,i)=>valid(v)?v+se[i]:null),a.map((v,i)=>valid(v)?v-se[i]:null));
  let [lo,hi]=extent(values),pad=(hi-lo)*.12||1e-9;lo-=pad;hi+=pad;
  const xx=t=>x0+(t+1)*pw/2,yy=v=>y0+(hi-v)/(hi-lo)*ph;
  ctx.font='15px Segoe UI';ctx.strokeStyle='#e0e4e6';ctx.lineWidth=1;
  for(let j=0;j<5;j++){const v=lo+(hi-lo)*j/4,y=yy(v);ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x0+pw,y);ctx.stroke();ctx.fillStyle='#233244';ctx.textAlign='right';ctx.fillText(fmt(v),x0-8,y+4)}
  for(const v of [-1,-.5,0,.5,1]){ctx.textAlign='center';ctx.fillText(v,xx(v),y0+ph+23)}
  if(se){let start=-1;for(let i=0;i<=a.length;i++){if(i<a.length&&valid(a[i])&&valid(se[i])){if(start<0)start=i}else if(start>=0){ctx.beginPath();ctx.moveTo(xx(t[start]),yy(a[start]+se[start]));for(let k=start+1;k<i;k++)ctx.lineTo(xx(t[k]),yy(a[k]+se[k]));for(let k=i-1;k>=start;k--)ctx.lineTo(xx(t[k]),yy(a[k]-se[k]));ctx.closePath();ctx.fillStyle='#245d8635';ctx.fill();start=-1}}}
  function line(v,color,dash){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash);ctx.beginPath();let on=false;v.forEach((z,i)=>{if(!valid(z)){on=false;return}if(!on){ctx.moveTo(xx(t[i]),yy(z));on=true}else ctx.lineTo(xx(t[i]),yy(z))});ctx.stroke();ctx.setLineDash([])}
  line(b,'#b66325',[6,4]);line(a,'#245d86',[]);
  ctx.fillStyle='#245d86';ctx.textAlign='left';ctx.fillText(isError?'VPINN − FEM':'VPINN',x0,h-20);ctx.fillStyle='#b66325';ctx.fillText(isError?'기준선 0':'P3 FEM (점선)',x0+150,h-20);
  if(se){ctx.fillStyle='#596575';ctx.fillText('음영: 평균 ± 1 SE',x0+310,h-20)}
}
const pendingCases = new Map();
let renderVersion = 0;
function loadCase(key) {
  if (D.cases[key]) return Promise.resolve();
  if (pendingCases.has(key)) return pendingCases.get(key);
  const request = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'field_cases/' + encodeURIComponent(key) + '.js';
    script.onload = () => { script.remove(); resolve(); };
    script.onerror = () => { script.remove(); pendingCases.delete(key); reject(new Error('Data load failed: '+key)); };
    document.head.appendChild(script);
  });
  pendingCases.set(key, request);
  return request;
}
async function render(){
  const version = ++renderVersion;
  const n=byId('count').value,mode=byId('mode').value,seed=byId('seed').value,k=+byId('component').value,diag=byId('diagnostic').value,view=byId('view').value,cutIndex=+byId('cut').value;
  const key=`${mode}_N${n}_${seed==='mean'?'mean':'seed'+seed}`;
  if (!D.cases[key]) {
    byId('selection-note').textContent='선택한 실험 데이터를 불러오는 중입니다.';
    try { await loadCase(key); } catch(error) {
      if (version === renderVersion) byId('selection-note').textContent='데이터를 불러오지 못했습니다. 페이지를 새로고침해 주세요.';
      console.error(error); return;
    }
  }
  if (version !== renderVersion) return;
  const data=D.cases[key];if(!data)throw Error('Missing case '+key);
  const pred=data.uv.map(v=>v[k]),ref=D.reference.map(v=>v[k]);
  let diagnostic=pred.map((v,i)=>valid(v)&&valid(ref[i])?v-ref[i]:null),palette='RdBu_r',title='VPINN − FEM';
  if(diag==='absolute'){diagnostic=diagnostic.map(x=>valid(x)?Math.abs(x):null);palette='magma';title='절대 차이'}
  if(diag==='residual'){diagnostic=data.residual.map(x=>x[k]);palette='magma';title=seed==='mean'?'평균 |강형 잔차|':'|강형 잔차|'}
  if(diag==='se'){diagnostic=seed==='mean'?data.se.map(x=>x[k]):pred.map(()=>null);palette='magma';title=seed==='mean'?'예측값 표준오차':'개별 시드: SE 정의 안 함'}
  const ranges=extent(pred.concat(ref)),ev=extent(diagnostic);let erange=diag==='signed'?[-Math.max(Math.abs(ev[0]),Math.abs(ev[1])),Math.max(Math.abs(ev[0]),Math.abs(ev[1]))]:[0,ev[1]];
  if(!erange.every(Number.isFinite))erange=[0,1];
  byId('prediction-title').textContent=(seed==='mean'?'평균 VPINN':'VPINN · seed '+seed)+' · '+(k?'v':'u');
  byId('diagnostic-title').textContent=title;
  byId('selection-note').textContent=`${labels[mode]} / ${Number(n).toLocaleString()}점 / ${seed==='mean'?'5개 시드 평균':'seed '+seed}. `+(view==='vector'?'벡터 보기는 항상 5개 시드의 성분별 평균입니다.':seed==='mean'?'평균 해의 그림과 평균 개별 오차를 구분하십시오.':'개별 실행을 표시합니다.');
  byId('maps').hidden=view!=='maps';byId('spatial-image').hidden=view==='maps';
  if(view!=='maps'){byId('spatial-image').src='figures/'+(view==='vector'?`${mode}_N${n}_mean_vector.png`:`${key}_surface.png`);return}
  drawMap('prediction',pred,'viridis',ranges,D.cuts[cutIndex]);drawMap('reference',ref,'viridis',ranges,D.cuts[cutIndex]);drawMap('difference',diagnostic,palette,erange,D.cuts[cutIndex]);
  const cut=D.cuts[cutIndex],cp=data.cuts[cutIndex].map(v=>v[k]),cr=cut.ref.map(v=>v[k]),se=seed==='mean'?data.cutse[cutIndex].map(v=>v[k]):null;
  drawSlice('slice',cut.t,cp,cr,se,false);drawSlice('slice-error',cut.t,cp,cr,se,true);
}
for(const id of ['count','mode','seed','component','diagnostic','view','cut'])byId(id).addEventListener('change',render);
document.querySelectorAll('nav a').forEach(a=>a.addEventListener('click',()=>{document.querySelectorAll('nav a').forEach(x=>x.classList.remove('active'));a.classList.add('active')}));
render();

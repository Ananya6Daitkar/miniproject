'use strict';
const $=id=>document.getElementById(id);
const names={sac:'Trained SAC',cp_sat:'CP-SAT (one-window)',threshold:'Threshold rule',no_maintenance:'No maintenance'};
const colors={sac:'#64dfb5',cp_sat:'#7ba8ff',threshold:'#efbc66',no_maintenance:'#ed7582'};
let data=null, policy='sac', step=0, timer=null, report=null, activeScenario=null;
const fmt=(x,d=1)=>Number(x).toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:d});
const color=c=>c>6?'#ed7582':c>=3?'#efbc66':'#52d4ad';
function notice(message,error=false){$('notice').textContent=message;$('notice').classList.toggle('error',error);}
async function request(url,options){const r=await fetch(url,options);if(!r.ok){let j;try{j=await r.json()}catch{}throw new Error(typeof j?.detail==='string'?j.detail:`Request failed (${r.status}). Check your input or server.`)}return r;}
const jsonOptions=x=>({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(x)});
function params(){return {seed:Number($('seed').value),horizon:60,deterioration:Number($('deterioration').value),closures:Number($('closures').value)}}
function pause(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ Play';}
function chart(target,series,{ylabel='Score',xlabel='Window'}={}){
 const all=series.flatMap(s=>s.values.map(v=>v[1]));if(!all.length){$(target).textContent='No measurements available.';return;}
 const W=590,H=240,L=62,R=18,T=15,B=38;let lo=Math.min(...all),hi=Math.max(...all);const pad=Math.max(.5,(hi-lo)*.12);lo-=pad;hi+=pad;
 const xmax=Math.max(1,...series.flatMap(s=>s.values.map(v=>v[0]))),x=v=>L+v/xmax*(W-L-R),y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${ylabel} over ${xlabel}">`;
 for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,py=y(v);svg+=`<line x1="${L}" x2="${W-R}" y1="${py}" y2="${py}" stroke="#263341"/><text x="${L-8}" y="${py+4}" fill="#94a6b9" font-size="10" text-anchor="end">${fmt(v,hi-lo>20?0:1)}</text>`}
 for(let i=0;i<5;i++){const v=xmax*i/4;svg+=`<text x="${x(v)}" y="${H-18}" text-anchor="middle" fill="#94a6b9" font-size="10">${Math.round(v)}</text>`}
 for(const s of series)svg+=`<polyline points="${s.values.map(v=>`${x(v[0])},${y(v[1])}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2.2" stroke-linejoin="round"/>`;
 svg+=`<text x="${W/2}" y="${H-1}" fill="#94a6b9" font-size="10" text-anchor="middle">${xlabel}</text></svg><div class="chart-legend">${series.map(s=>`<span><i style="background:${s.color}"></i>${s.name}</span>`).join('')}</div>`;$(target).innerHTML=svg;
}
function rail(row){
 const n=row.after.length,W=800,start=43,gap=77,y=133;
 let svg=`<svg viewBox="0 0 ${W} 225" role="img" aria-label="Track conditions after window ${step+1}"><path d="M0 90 Q90 20 175 65 T360 45 T570 80 T800 30 L800 225 L0 225Z" fill="#172633"/><path d="M0 113 Q140 66 280 95 T550 85 T800 98 L800 225 L0 225Z" fill="#1a2d37"/><line x1="20" x2="780" y1="${y-5}" y2="${y-5}" stroke="#56707e" stroke-width="3"/><line x1="20" x2="780" y1="${y+5}" y2="${y+5}" stroke="#56707e" stroke-width="3"/>`;
 for(let x=20;x<780;x+=10)svg+=`<line x1="${x}" x2="${x}" y1="${y-10}" y2="${y+10}" stroke="#45606d" stroke-width="2"/>`;
 row.after.forEach((c,i)=>{const x=start+i*gap;svg+=`<rect x="${x-25}" y="${y-13}" width="50" height="26" rx="5" fill="${color(c)}" opacity=".9"/><text x="${x}" y="${y+4}" text-anchor="middle" font-size="12" font-weight="700" fill="#10231f">${fmt(c)}</text><text x="${x}" y="${y+37}" text-anchor="middle" font-size="12" fill="#d4e3ea">S${String(i+1).padStart(2,'0')}</text><text x="${x}" y="${y+56}" text-anchor="middle" font-size="10" fill="#94a6b9">${i*5}–${(i+1)*5} km</text>`;if(!row.mask[i+1])svg+=`<text x="${x}" y="${y-23}" fill="#95a6b7" font-size="10" text-anchor="middle">unavailable</text>`;});
 const x=start+row.next_position*gap;svg+=`<g transform="translate(${x-23},${y-65})"><rect x="0" y="5" width="46" height="25" rx="5" fill="#64dfb5"/><rect x="28" y="0" width="16" height="22" rx="3" fill="#64dfb5"/><rect x="31" y="4" width="9" height="9" rx="2" fill="#183c36"/><circle cx="10" cy="31" r="5" fill="#91b2b2"/><circle cx="35" cy="31" r="5" fill="#91b2b2"/><path d="M20 38 L26 38 L23 43Z" fill="#64dfb5"/></g></svg>`;$('rail-map').innerHTML=svg;
}
function renderStep(){if(!data)return;const run=data.runs[policy],row=run.trace[step];$('timeline').value=step;$('step-label').textContent=`WINDOW ${String(step+1).padStart(2,'0')} / ${run.trace.length}`;rail(row);
 $('decision-title').textContent=row.action?`Tamp segment ${String(row.action).padStart(2,'0')}`:'Wait this window';
 $('decision-detail').textContent=policy==='sac'?'Highest-probability feasible action from the trained actor.':policy==='cp_sat'?'Lowest expected one-window penalty among feasible actions.':policy==='threshold'?'Service the worst feasible segment when its score reaches 4.5.':'Control policy: no maintenance is performed.';
 $('decision-stats').innerHTML=[['Available window',`${fmt(row.available_minutes,0)} min`],['Travel + work',`${fmt(row.minutes_used)} min`],['Maintenance cost',`${fmt(row.cost)} units`],['Observed reward',fmt(row.reward)],['Constraint check',row.mask[row.action]&&row.minutes_used<=row.available_minutes?'✓ Feasible':'Violation']].map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join('');
 $('prob-title').textContent=row.probabilities?'ACTOR PROBABILITIES':'FEASIBLE ACTIONS';
 const probs=row.probabilities;const indices=probs?probs.map((_,i)=>i).sort((a,b)=>probs[b]-probs[a]).slice(0,4):row.mask.map((_,i)=>i).filter(i=>row.mask[i]).slice(0,5);
 $('probabilities').innerHTML=indices.map(i=>`<div class="prob-row"><span>${i?'S'+String(i).padStart(2,'0'):'Wait'}</span><div class="prob-track"><div class="prob-bar" style="width:${probs?probs[i]*100:100}%"></div></div><span>${probs?fmt(probs[i]*100,0)+'%':'✓'}</span></div>`).join('');
 $('prob-note').textContent=probs?'Top four actions. Probabilities are model outputs, not confidence estimates.':'Actions must fit the window and avoid unavailable track units.';
}
function render(){if(!data)return;const m=data.runs[policy].metrics;
 $('metrics').innerHTML=[['Episode return',fmt(m.return),'Higher is better'],['Mean condition',fmt(m.mean_condition,2),'Lower is better'],['Maintenance cost',fmt(m.cost),'Synthetic cost units'],['Overdue exposure',fmt(m.overdue_segment_windows,0),'Segment-windows above threshold']].map(([k,v,h])=>`<div class="metric"><span>${k}</span><strong>${v}</strong><small>${h}</small></div>`).join('');
 $('comparison').innerHTML=Object.entries(data.runs).map(([p,r])=>{const v=r.metrics;return `<tr class="${p===policy?'selected':''}"><td>${names[p]}</td><td>${fmt(v.return)}</td><td>${fmt(v.cost)}</td><td>${fmt(v.mean_condition,2)}</td><td>${v.overdue_segment_windows}</td><td>${fmt(v.travel_km)}</td><td>${v.constraint_violations}</td></tr>`}).join('');
 chart('condition-chart',Object.entries(data.runs).map(([p,r])=>({name:names[p],color:colors[p],values:r.trace.map((x,i)=>[i+1,x.mean_condition])})),{ylabel:'Mean condition'});
 const rows=data.runs[policy].trace,W=600,H=205,L=32,T=8,cw=(W-L)/rows.length,ch=17;let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Maintenance actions by segment and window">`;
 for(let i=0;i<10;i++){svg+=`<text x="0" y="${T+i*ch+11}" fill="#94a6b9" font-size="9">S${i+1}</text>`;rows.forEach((r,j)=>{svg+=`<rect x="${L+j*cw}" y="${T+i*ch}" width="${cw-1}" height="${ch-2}" rx="1" fill="${r.action===i+1?colors[policy]:'#1d2b38'}"><title>Window ${j+1}, Segment ${i+1}${r.action===i+1?': tamped':''}</title></rect>`})}
 [1,15,30,45,60].forEach(v=>svg+=`<text x="${L+(v-1)*cw}" y="${H-9}" font-size="9" fill="#94a6b9">${v}</text>`);$('schedule').innerHTML=svg+'</svg>';renderStep();
}
async function simulate(){pause();const button=$('run');button.disabled=true;button.textContent='Running all four methods…';notice('Running learned SAC and three baselines on identical scenario data…');try{const chosen=params();data=await(await request('/api/simulate',jsonOptions(chosen))).json();activeScenario=chosen;step=0;render();notice(`Scenario ${chosen.seed} complete · 60 windows × 4 methods · synthetic data · episode totals shown`);}catch(e){notice(e.message,true)}finally{button.disabled=false;button.innerHTML='Run comparison <span>↗</span>';}}
function renderEvaluation(){if(!report)return;const suite=$('suite').value,s=report.summary[suite];$('evaluation').innerHTML=Object.entries(s).map(([p,m])=>`<tr class="${p==='sac'?'selected':''}"><td>${names[p]}</td><td>${fmt(m.return.mean)} ± ${fmt(m.return.std)}</td><td>${fmt(m.cost.mean)}</td><td>${fmt(m.mean_condition.mean,2)}</td><td>${fmt(m.overdue_segment_windows.mean)}</td><td>${m.return.n}</td></tr>`).join('');const ranking=Object.entries(s).sort((a,b)=>b[1].return.mean-a[1].return.mean);$('finding').textContent=`Measured finding: ${names[ranking[0][0]]} has the highest mean return in this suite. SAC's mean return is ${fmt(s.sac.return.mean)}; CP-SAT's is ${fmt(s.cp_sat.return.mean)}. These results describe this synthetic environment, not field performance.`;}
async function evidence(){try{const [r,t]=await Promise.all([request('/api/evaluation'),request('/api/training')]);report=await r.json();const logs=await t.json();$('experiment-info').textContent=`${report.training_runs.length} training runs · ${report.test_seeds.length} held-out scenarios per suite · Standard + stress evaluation · Demo selected using validation: seed ${report.selected_demo_seed}`;
 const palette=['#64dfb5','#7ba8ff','#efbc66'];chart('training-chart',logs.map((run,i)=>({name:`Seed ${run.seed}`,color:palette[i%3],values:run.episodes.map((r,j)=>[r.step,run.episodes.slice(Math.max(0,j-9),j+1).reduce((a,b)=>a+b.return,0)/Math.min(j+1,10)])})),{ylabel:'Training return',xlabel:'Training steps'});
 chart('validation-chart',logs.map((run,i)=>({name:`Seed ${run.seed}`,color:palette[i%3],values:run.validation.map(v=>[v.step,v.mean_return])})),{ylabel:'Validation return',xlabel:'Training steps'});
 $('run-cards').innerHTML=report.training_runs.map(r=>`<div class="run-card"><b>Training seed ${r.seed}</b><p>${r.requested_steps.toLocaleString()} interaction steps</p><p>${r.episodes} completed episodes</p><p>Best validation: ${fmt(r.best_validation_return)}</p><p>Test return: ${fmt(report.per_training_seed_returns[r.seed].standard.mean)}</p></div>`).join('');renderEvaluation();}catch(e){$('experiment-info').textContent=e.message;}}
$('run').addEventListener('click',simulate);$('suite').addEventListener('change',renderEvaluation);
$('timeline').addEventListener('input',e=>{pause();step=Number(e.target.value);renderStep()});
$('next').addEventListener('click',()=>{pause();if(data){step=(step+1)%data.runs[policy].trace.length;renderStep()}});
$('play').addEventListener('click',()=>{if(timer){pause();return}if(!data)return;$('play').textContent='Ⅱ Pause';timer=setInterval(()=>{step++;if(step>=data.runs[policy].trace.length){step=0;pause()}renderStep()},650)});
for(const b of document.querySelectorAll('.policy'))b.addEventListener('click',()=>{policy=b.dataset.policy;document.querySelectorAll('.policy').forEach(x=>x.classList.toggle('active',x===b));render()});
for(const b of document.querySelectorAll('.tab'))b.addEventListener('click',()=>{pause();document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.pane').forEach(x=>x.classList.toggle('active',x.id===b.dataset.tab))});
$('export').addEventListener('click',async()=>{if(!activeScenario)return;try{const r=await request(`/api/export/${policy}`,jsonOptions(activeScenario));const url=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=url;a.download=`${policy}-${activeScenario.seed}-schedule.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}catch(e){notice(e.message,true)}});
simulate();evidence();

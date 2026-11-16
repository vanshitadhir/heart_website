const $=s=>document.querySelector(s), api=u=>fetch(u).then(r=>r.json()), pct=v=>(v*100).toFixed(1)+'%';
const COL=['#5b5bf0','#8b5cf6','#f472b6','#22c1c3','#f59e0b','#10b981'];
Chart.defaults.font.family='Inter,system-ui,sans-serif';Chart.defaults.plugins.legend.labels.boxWidth=12;
const mk=(id,type,labels,sets,opt={})=>new Chart($(id),{type,data:{labels,datasets:sets},options:{responsive:true,...opt}});
const card=(t,v,s='')=>`<div class="card"><small>${t}</small><b>${v}</b><span>${s}</span></div>`;

async function init(){
 const [o,p,e,m]=await Promise.all([api('/api/overview'),api('/api/preprocessing'),api('/api/eda'),api('/api/models')]);
 $('#glance').innerHTML=card('DATASET SIZE',o.rows_raw,'patients, '+o.features+' features')+card('DUPLICATES',o.duplicates,'removed in cleaning')+
  card('CLEAN ROWS',o.rows_clean,pct(o.positive)+' have heart disease')+card('MODELS',o.models,'compared')+card('BEST ACCURACY',pct(o.best_acc),o.best);
 $('#prepcards').innerHTML=card('RAW ROWS',p.rows_raw)+card('DUPLICATE ROWS',p.duplicates,'dropped')+card('CHOLESTEROL = 0',p.chol_zero,'treated as missing')+
  card('BP = 0',p.bp_zero,'treated as missing')+card('TRAIN / TEST',p.train+' / '+p.test,'80/20 stratified split');
 $('#steps').innerHTML=[['1. Remove duplicates','Identical rows would appear in both train and test sets and inflate accuracy.'],
  ['2. Fix impossible zeros','Cholesterol and resting BP cannot be 0. They are set to missing and filled with the training-set median.'],
  ['3. Scale numeric features','Age, BP, cholesterol, heart rate and oldpeak are standardized (needed for KNN, SVM, regression).'],
  ['4. Encode categories','Chest pain type, resting ECG and ST slope are one-hot encoded. Binary columns pass through.'],
  ['5. Split and validate','Stratified 80/20 split plus 5-fold cross-validation. All steps live inside a scikit-learn Pipeline, so nothing leaks.']]
  .map(s=>`<div class="step"><b>${s[0]}</b>${s[1]}</div>`).join('');
 $('#coltable').innerHTML='<tr><th>Column</th><th>Type</th><th>Min</th><th>Max</th><th>Mean</th><th>Missing</th><th>Invalid zeros</th></tr>'+
  p.columns.map(c=>`<tr><td>${c.column}</td><td>${c.dtype}</td><td>${c.min}</td><td>${c.max}</td><td>${c.mean}</td><td>${c.missing}</td><td>${c.invalid}</td></tr>`).join('');
 eda(e);models(m,o);predictForm();findings(o,e,m);
}
function eda(e){
 mk('#c1','doughnut',['No disease','Heart disease'],[{data:[e.target[0],e.target[1]],backgroundColor:[COL[0],COL[2]]}]);
 mk('#c2','bar',e.age.labels,[{label:'Patients',data:e.age.values,backgroundColor:COL[0],borderRadius:6}],{plugins:{legend:{display:false}}});
 mk('#c3','bar',['Female (0)','Male (1)'],[{label:'No disease',data:[e.sex[0][0],e.sex[1][0]],backgroundColor:COL[0],borderRadius:6},{label:'Heart disease',data:[e.sex[0][1],e.sex[1][1]],backgroundColor:COL[2],borderRadius:6}]);
 const cp={1:'Typical angina',2:'Atypical angina',3:'Non-anginal',4:'Asymptomatic'};
 mk('#c4','pie',Object.keys(e.chest).map(k=>cp[k]),[{data:Object.values(e.chest),backgroundColor:COL}]);
 mk('#c5','line',e.chol.labels,[{label:'Avg cholesterol',data:e.chol.values,borderColor:COL[1],backgroundColor:COL[1],tension:.3}],{plugins:{legend:{display:false}},scales:{y:{title:{display:true,text:'mg/dl'}},x:{title:{display:true,text:'Age group'}}}});
 mk('#c6','bar',e.corr.labels,[{data:e.corr.values,backgroundColor:e.corr.values.map(v=>v>=0?COL[2]:COL[0]),borderRadius:4}],{indexAxis:'y',plugins:{legend:{display:false}}});
}
function models(m,o){
 $('#split').textContent=`Trained on ${m.train} patients, tested on ${m.test}. Linear Regression predicts a value that is thresholded at 0.5.`;
 const R=m.results,top=Math.max(...R.map(r=>r.accuracy));
 $('#modeltable').innerHTML='<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>5-fold CV</th></tr>'+
  R.map(r=>`<tr><td>${r.name}</td><td class="${r.accuracy==top?'best':''}">${pct(r.accuracy)}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${pct(r.f1)}</td><td>${r.auc.toFixed(3)}</td><td>${pct(r.cv)}</td></tr>`).join('');
 mk('#m1','bar',R.map(r=>r.name),[{data:R.map(r=>r.accuracy*100),backgroundColor:R.map(r=>r.accuracy==top?COL[0]:'#c7c9f8'),borderRadius:6}],{plugins:{legend:{display:false}},scales:{y:{min:60,max:100}}});
 mk('#m2','line',null,R.map((r,i)=>({label:r.name,data:r.roc.fpr.map((x,j)=>({x,y:r.roc.tpr[j]})),borderColor:COL[i],pointRadius:0,tension:0})),{scales:{x:{type:'linear',min:0,max:1,title:{display:true,text:'False positive rate'}},y:{min:0,max:1,title:{display:true,text:'True positive rate'}}}});
 $('#cmsel').innerHTML=R.map(r=>`<option>${r.name}</option>`).join('');
 const cm=()=>{const c=R.find(r=>r.name==$('#cmsel').value).cm,l=[['True negative',c[0][0],'#e7f6ec'],['False positive',c[0][1],'#fde8ef'],['False negative',c[1][0],'#fde8ef'],['True positive',c[1][1],'#e7f6ec']];
  $('#cm').innerHTML=l.map(x=>`<div style="background:${x[2]}">${x[1]}<small>${x[0]}</small></div>`).join('')};$('#cmsel').onchange=cm;cm();
 mk('#m3','bar',m.importance.map(i=>i.feature),[{data:m.importance.map(i=>i.importance),backgroundColor:COL[1],borderRadius:4}],{indexAxis:'y',plugins:{legend:{display:false}}});
 const ra=m.raw_accuracy;
 $('#leak').innerHTML=`<b>Why accuracy differs from the original notebook:</b> with the ${o.duplicates} duplicate rows kept, Random Forest scores ${pct(ra['Random Forest'])} and SVM ${pct(ra['SVM'])}, because copies of the same patient land in both train and test sets. After cleaning, Random Forest scores ${pct(R.find(r=>r.name=='Random Forest').accuracy)}. The cleaned numbers are the honest ones.`;
}
const F=[['age','Age (years)',55],['sex','Sex',[[1,'Male'],[0,'Female']]],['chest pain type','Chest pain type',[[4,'4 Asymptomatic'],[3,'3 Non-anginal'],[2,'2 Atypical angina'],[1,'1 Typical angina']]],
 ['resting bp s','Resting BP (mm Hg)',130],['cholesterol','Cholesterol (mg/dl)',240],['fasting blood sugar','Fasting sugar > 120',[[0,'No'],[1,'Yes']]],
 ['resting ecg','Resting ECG (0-2)',[[0,'0 Normal'],[1,'1 ST-T abnormality'],[2,'2 LV hypertrophy']]],['max heart rate','Max heart rate',140],
 ['exercise angina','Exercise angina',[[0,'No'],[1,'Yes']]],['oldpeak','Oldpeak (ST depression)',1],['ST slope','ST slope (0-3)',[[2,'2 Flat'],[1,'1 Upsloping'],[3,'3 Downsloping'],[0,'0 Unknown']]]];
function predictForm(){
 $('#form').innerHTML=F.map(f=>`<div><label>${f[1]}</label>${Array.isArray(f[2])?`<select style="margin:0;width:100%" name="${f[0]}">${f[2].map(o=>`<option value="${o[0]}">${o[1]}</option>`).join('')}</select>`:`<input type="number" step="any" name="${f[0]}" value="${f[2]}">`}</div>`).join('');
 $('#go').onclick=async()=>{const d={};new FormData($('#form')).forEach((v,k)=>d[k]=v);
  const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}),j=await r.json();
  if(j.error){$('#out').innerHTML=`<div class="res">${j.error}</div>`;return}
  $('#out').innerHTML=Object.entries(j.probabilities).map(([n,v])=>`<div class="res ${v>=.5?'hi':'lo'}">${n}<b>${pct(v)}</b>${v>=.5?'Heart disease likely':'Low risk'}</div>`).join('')};
}
function findings(o,e,m){
 const R=[...m.results].sort((a,b)=>b.accuracy-a.accuracy),c=e.corr,i=m.importance,L=R[R.length-1];
 const pos=c.labels.map((l,k)=>[l,c.values[k]]).sort((a,b)=>b[1]-a[1]);
 $('#find').innerHTML=[`After cleaning, ${o.rows_clean} unique patients remain and ${pct(o.positive)} of them have heart disease, so the classes are reasonably balanced.`,
  `${R[0].name} scored highest at ${pct(R[0].accuracy)} accuracy, while ${L.name} scored lowest at ${pct(L.accuracy)}. The top models are within a few points of each other on this test set.`,
  `Strongest positive correlations with heart disease: ${pos[0][0]} (${pos[0][1]}) and ${pos[1][0]} (${pos[1][1]}). Strongest negative: ${pos[pos.length-1][0]} (${pos[pos.length-1][1]}).`,
  `Random Forest relies most on ${i[0].feature}, ${i[1].feature} and ${i[2].feature}.`,
  `Removing ${o.duplicates} duplicate rows lowers some scores, but it removes train/test leakage and gives a fairer estimate of real performance.`,
  `Cholesterol values of 0 are data errors, not real measurements, so they were imputed rather than used as-is.`].map(t=>`<li>${t}</li>`).join('');
}
const secs=[...document.querySelectorAll('main section')];
addEventListener('scroll',()=>{let cur=secs[0].id;secs.forEach(s=>{if(s.getBoundingClientRect().top<160)cur=s.id});
 document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('on',a.hash=='#'+cur))});
init();

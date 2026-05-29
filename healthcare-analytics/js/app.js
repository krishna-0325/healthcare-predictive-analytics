/* =================================================
   Healthcare Predictive Analytics — app.js
   Calls the real Flask API (api.py) for predictions.
   Falls back to local weighted model if API is offline.
   ================================================= */

'use strict';

const API_BASE = 'http://localhost:5000';

// ─── Tab Navigation ──────────────────────────────────────────────────────────

const navItems = document.querySelectorAll('.nav-item');
const panels   = document.querySelectorAll('.panel');

navItems.forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    navItems.forEach(n => n.classList.remove('active'));
    panels.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-' + tab).classList.add('active');
    if (tab === 'insights') { setTimeout(drawInsightCharts, 80); loadModelInfo(); }
  });
});

// ─── Slider Binding ──────────────────────────────────────────────────────────

function bindSlider(id, outId, decimals = 0) {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener('input', () => {
    const v = parseFloat(el.value);
    document.getElementById(outId).textContent =
      decimals > 0 ? v.toFixed(decimals) : Math.round(v);
  });
}

bindSlider('glucose','glucose-v'); bindSlider('bmi','bmi-v',1);
bindSlider('age-d','age-d-v');     bindSlider('bp','bp-v');
bindSlider('insulin','insulin-v'); bindSlider('preg','preg-v');
bindSlider('dpf','dpf-v',2);       bindSlider('skin','skin-v');
bindSlider('age-h','age-h-v');     bindSlider('rest-bp','rest-bp-v');
bindSlider('chol','chol-v');       bindSlider('thal-hr','thal-hr-v');
bindSlider('old-peak','old-peak-v',1);

// ─── Risk Helpers ─────────────────────────────────────────────────────────────

function riskInfo(score) {
  if (score < 30) return { label:'Low Risk',      cls:'low',  color:'#639922' };
  if (score < 60) return { label:'Moderate Risk', cls:'mod',  color:'#BA7517' };
  return               { label:'High Risk',      cls:'high', color:'#A32D2D' };
}

function renderResult(prefix, score, features, modelName) {
  const info = riskInfo(score);
  document.getElementById(prefix+'-placeholder').style.display    = 'none';
  document.getElementById(prefix+'-result-content').style.display = 'block';
  document.getElementById(prefix+'-score-num').textContent = score + '%';
  const fill = document.getElementById(prefix+'-meter-fill');
  fill.style.width = score + '%'; fill.style.background = info.color;
  const badge = document.getElementById(prefix+'-risk-badge');
  badge.textContent = info.label; badge.className = 'risk-badge ' + info.cls;

  const featList = document.getElementById(prefix+'-feat-list');
  featList.innerHTML = '';
  features.forEach(f => {
    const pct   = f.pct !== undefined ? f.pct : Math.round((f.importance||0)*100);
    const barW  = Math.min(pct * 3.5, 100);
    const color = pct > 20 ? '#A32D2D' : pct > 12 ? '#BA7517' : '#639922';
    const item  = document.createElement('div');
    item.className = 'feat-item';
    item.innerHTML = `<span class="feat-name">${f.name}</span>
      <div class="feat-bar-bg"><div class="feat-bar" style="width:${barW}%;background:${color}"></div></div>
      <span class="feat-pct">${pct}%</span>`;
    featList.appendChild(item);
  });

  const note = document.querySelector(`#${prefix}-result-content .model-note`);
  if (note && modelName) note.innerHTML = `<i class="ti ti-info-circle"></i> ${modelName} · 10-fold CV · StandardScaler normalization`;
}

function setButtonLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  btn.disabled = loading;
  btn.innerHTML = loading
    ? '<i class="ti ti-loader-2" style="animation:spin 1s linear infinite"></i> Running Model…'
    : '<i class="ti ti-brain"></i> Run Prediction';
}

function showApiStatus(prefix, isLive) {
  const rc = document.getElementById(prefix+'-result-content');
  let el = rc.querySelector('.api-status');
  if (!el) { el = document.createElement('div'); el.className='api-status'; rc.querySelector('.model-note').after(el); }
  el.innerHTML = isLive
    ? `<span style="color:#3B6D11;font-size:11px"><i class="ti ti-circle-check"></i> Live — Flask API connected (real model)</span>`
    : `<span style="color:#BA7517;font-size:11px"><i class="ti ti-circle-dashed"></i> Offline mode — run api.py for real predictions</span>`;
}

// ─── API Health ───────────────────────────────────────────────────────────────

async function isApiAlive() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    const d   = await res.json();
    return d.status === 'ok' && d.models_loaded;
  } catch { return false; }
}

// ─── Diabetes Prediction ──────────────────────────────────────────────────────

async function predictDiabetes() {
  setButtonLoading('predict-diabetes-btn', true);
  const payload = {
    Pregnancies: parseFloat(document.getElementById('preg').value),
    Glucose: parseFloat(document.getElementById('glucose').value),
    BloodPressure: parseFloat(document.getElementById('bp').value),
    SkinThickness: parseFloat(document.getElementById('skin').value),
    Insulin: parseFloat(document.getElementById('insulin').value),
    BMI: parseFloat(document.getElementById('bmi').value),
    DiabetesPedigreeFunction: parseFloat(document.getElementById('dpf').value),
    Age: parseFloat(document.getElementById('age-d').value),
  };

  const alive = await isApiAlive();
  if (alive) {
    try {
      const res  = await fetch(`${API_BASE}/api/predict/diabetes`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      const data = await res.json();
      renderResult('diabetes', data.score, data.top_features, data.model);
      showApiStatus('diabetes', true);
    } catch { runDiabetesFallback(payload); showApiStatus('diabetes', false); }
  } else { runDiabetesFallback(payload); showApiStatus('diabetes', false); }

  setButtonLoading('predict-diabetes-btn', false);
}

function runDiabetesFallback(p) {
  const g=Math.min((p.Glucose-50)/150,1), b=Math.min((p.BMI-15)/40,1),
        a=Math.min((p.Age-18)/72,1), bp=Math.min((p.BloodPressure-40)/90,1),
        i=(p.Insulin<30||p.Insulin>250)?0.65:Math.abs(p.Insulin-100)/200,
        pr=Math.min(p.Pregnancies/10,1), d=Math.min(p.DiabetesPedigreeFunction/2.5,1);
  const score=Math.max(2,Math.min(97,Math.round((g*.32+b*.22+a*.15+bp*.10+i*.09+pr*.07+d*.12-.04)*100)));
  renderResult('diabetes', score,
    [{name:'Glucose',pct:32,importance:g},{name:'BMI',pct:22,importance:b},
     {name:'Age',pct:15,importance:a},{name:'Blood Pressure',pct:10,importance:bp},
     {name:'Pedigree Fn.',pct:12,importance:d}].sort((a,b)=>b.importance-a.importance),
    'Local weighted model (API offline)');
}

// ─── Heart Prediction ─────────────────────────────────────────────────────────

async function predictHeart() {
  setButtonLoading('predict-heart-btn', true);
  const payload = {
    age:parseFloat(document.getElementById('age-h').value),
    sex:parseInt(document.getElementById('sex-h').value),
    cp:parseInt(document.getElementById('cp').value),
    trestbps:parseFloat(document.getElementById('rest-bp').value),
    chol:parseFloat(document.getElementById('chol').value),
    fbs:parseInt(document.getElementById('fbs').value),
    restecg:0, thalach:parseFloat(document.getElementById('thal-hr').value),
    exang:parseInt(document.getElementById('exang').value),
    oldpeak:parseFloat(document.getElementById('old-peak').value),
    slope:1, ca:0, thal:2,
  };

  const alive = await isApiAlive();
  if (alive) {
    try {
      const res  = await fetch(`${API_BASE}/api/predict/heart`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      const data = await res.json();
      renderResult('heart', data.score, data.top_features, data.model);
      showApiStatus('heart', true);
    } catch { runHeartFallback(payload); showApiStatus('heart', false); }
  } else { runHeartFallback(payload); showApiStatus('heart', false); }

  setButtonLoading('predict-heart-btn', false);
}

function runHeartFallback(p) {
  const a=Math.min((p.age-25)/55,1), bp=Math.min((p.trestbps-80)/120,1),
        c=Math.min((p.chol-100)/500,1), hr=1-Math.min((p.thalach-60)/150,1),
        op=Math.min(p.oldpeak/6,1), cp=[0.55,0.30,0.25,0.85][p.cp]??0.4, ex=p.exang*.75;
  const score=Math.max(2,Math.min(97,Math.round((a*.18+bp*.15+c*.14+hr*.18+op*.12+cp*.13+ex*.10-.06+(p.sex?0.12:0)+(p.fbs?0.06:0))*100)));
  renderResult('heart', score,
    [{name:'Max Heart Rate',pct:18,importance:hr},{name:'Age',pct:18,importance:a},
     {name:'Chest Pain',pct:13,importance:cp},{name:'Cholesterol',pct:14,importance:c},
     {name:'ST Depression',pct:12,importance:op}].sort((a,b)=>b.importance-a.importance),
    'Local weighted model (API offline)');
}

// ─── Button Listeners ─────────────────────────────────────────────────────────
document.getElementById('predict-diabetes-btn').addEventListener('click', predictDiabetes);
document.getElementById('predict-heart-btn').addEventListener('click', predictHeart);

// ─── Model Info (Insights panel) ─────────────────────────────────────────────
async function loadModelInfo() {
  try {
    const res  = await fetch(`${API_BASE}/api/model-info`, { signal:AbortSignal.timeout(2000) });
    const data = await res.json();
    const dm = data.diabetes?.metrics, hm = data.heart?.metrics;
    const cards = document.querySelectorAll('.metric-card .metric-val');
    if (dm && cards[0]) cards[0].textContent = dm.accuracy + '%';
    if (hm && cards[1]) cards[1].textContent = hm.auc_roc;
    // Update table
    const rows = document.querySelectorAll('.metrics-table tbody tr');
    const setCells = (row, m) => {
      if (!row || !m) return;
      const c = row.querySelectorAll('td');
      if(c[1]) c[1].textContent = m.accuracy+'%';
      if(c[2]) c[2].textContent = m.precision+'%';
      if(c[3]) c[3].textContent = m.recall+'%';
      if(c[4]) c[4].textContent = m.f1+'%';
      if(c[5]) c[5].textContent = m.auc_roc;
    };
    setCells(rows[0], dm); setCells(rows[1], hm);
  } catch { /* API offline, static values shown */ }
}

// ─── Insight Charts ───────────────────────────────────────────────────────────
let chartsDrawn = false;

async function drawInsightCharts() {
  if (chartsDrawn) return; chartsDrawn = true;
  let dL=['Glucose','BMI','Age','Diab. Pedigree','Blood Pressure','Insulin'], dD=[32,22,15,12,10,9];
  let hL=['Max Heart Rate','Age','Blood Pressure','Cholesterol','Chest Pain','ST Depression','Exer. Angina'], hD=[18,18,15,14,13,12,10];

  try {
    const res  = await fetch(`${API_BASE}/api/model-info`, { signal:AbortSignal.timeout(2000) });
    const info = await res.json();
    if (info.diabetes?.importances?.length) {
      const items=info.diabetes.importances.slice(0,6); dL=items.map(f=>f.name); dD=items.map(f=>f.pct);
    }
    if (info.heart?.importances?.length) {
      const items=info.heart.importances.slice(0,7); hL=items.map(f=>f.name); hD=items.map(f=>f.pct);
    }
  } catch {}

  new Chart(document.getElementById('diabChart'), {
    type:'bar', data:{ labels:dL, datasets:[{ label:'Importance (%)', data:dD,
      backgroundColor:['#185FA5','#185FA5','#378ADD','#378ADD','#B5D4F4','#B5D4F4'], borderRadius:5, borderSkipped:false }]},
    options:{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false}, tooltip:{callbacks:{label:ctx=>`  ${ctx.raw.toFixed(1)}% importance`}} },
      scales:{ x:{grid:{color:'rgba(0,0,0,0.04)'},ticks:{callback:v=>v+'%',font:{size:11}}}, y:{grid:{display:false},ticks:{font:{size:12}}} } }
  });

  new Chart(document.getElementById('heartChart'), {
    type:'bar', data:{ labels:hL, datasets:[{ label:'Importance (%)', data:hD,
      backgroundColor:['#993556','#993556','#D4537E','#D4537E','#ED93B1','#ED93B1','#F4C0D1'], borderRadius:5, borderSkipped:false }]},
    options:{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false}, tooltip:{callbacks:{label:ctx=>`  ${ctx.raw.toFixed(1)}% importance`}} },
      scales:{ x:{grid:{color:'rgba(0,0,0,0.04)'},ticks:{callback:v=>v+'%',font:{size:11}}}, y:{grid:{display:false},ticks:{font:{size:12}}} } }
  });
}

// Inject spinner + status styles
const s=document.createElement('style');
s.textContent=`@keyframes spin{to{transform:rotate(360deg)}} .api-status{margin-top:8px;display:flex;align-items:center;gap:5px;}`;
document.head.appendChild(s);

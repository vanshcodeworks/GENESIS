// filepath: e:\DNA-SEQ(CH22)\demo\app.js
// Enhanced GENESIS Demo App
(function(){
  // Auto-detect API base if not provided (same origin)
  const detectedBase = (window.API_BASE && window.API_BASE.trim()) || (location.origin);
  window.API_BASE = detectedBase.replace(/\/+$/,'');

  const seqEl = document.getElementById('seq');
  const seqView = document.getElementById('seqView');
  const statusEl = document.getElementById('status');
  const resEl = document.getElementById('result');
  const posEl = document.getElementById('pos');
  const baseEl = document.getElementById('base');
  const spinner = document.getElementById('spinner');
  const confidenceBar = document.getElementById('confidenceBar');
  const confidenceText = document.getElementById('confidenceText');

  // Example sequences for each gene type (fallback; will be overridden by examples.json if present)
  const examples = {
    CHEK2: 'ATGCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT',
    TBX1:  'CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG',
    COMT:  'TACGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG',
    Other: null
  };
  let curated = null;

  async function loadCurated() {
    try {
      const rsp = await fetch('./examples.json', { cache: 'no-cache' });
      if (!rsp.ok) return;
      const data = await rsp.json();
      curated = data;
    } catch (_) { /* ignore */ }
  }

  function pickCurated(type) {
    if (!curated || !curated[type]) return null;
    const arr = Array.isArray(curated[type]) ? curated[type] : [curated[type]];
    if (!arr.length) return null;
    return arr[Math.floor(Math.random() * arr.length)];
  }

  function cleanSeq(s){ return s.toUpperCase().replace(/[^ACGT]/g,'A'); }

  function ensureLen(s, n=200){
    s = (s || '').toUpperCase().replace(/[^ACGT]/g,'A');
    if (s.length > n) return s.slice(0, n);
    if (s.length < n) return s + 'A'.repeat(n - s.length);
    return s;
  }

  function updateStats(s) {
    const lenEl = document.getElementById('seqLength');
    const compEl = document.getElementById('seqComposition');
    if (!lenEl || !compEl) return;
    lenEl.textContent = `Length: ${s.length}`;
    if(s.length > 0) {
      const counts = {A:0, C:0, G:0, T:0};
      for(let c of s) if(counts.hasOwnProperty(c)) counts[c]++;
      const total = s.length;
      const comp = Object.entries(counts).map(([k,v]) => `${k}:${((v/total)*100).toFixed(1)}%`).join(' ');
      compEl.textContent = comp;
    }
  }

  function renderSeq(s){
    if (!seqView) return;
    const p = parseInt((posEl && posEl.value) || '1',10);
    let out = '';
    for(let i=0;i<s.length;i++){
      const ch = s[i];
      if(i+1 === p) out += `[${ch}]`;
      else out += ch;
      if((i+1) % 50 === 0) out += '\n';
    }
    seqView.textContent = out;
  }

  function showProbs(obj){
    resEl.innerHTML = '';
    const probs = obj.probabilities || {};
    const values = Object.values(probs);
    const maxProb = values.length ? Math.max(...values) : 0;

    Object.entries(probs).forEach(([k,v])=>{
      const div = document.createElement('div');
      div.className = 'prob';
      if(v === maxProb && v > 0.5) div.classList.add('high');

      const label = document.createElement('div');
      label.style.fontWeight = 'bold';
      label.style.marginBottom = '5px';
      label.textContent = k;

      const percentage = document.createElement('div');
      percentage.style.fontSize = '1.2em';
      percentage.style.color = v === maxProb ? '#16a34a' : '#64748b';
      percentage.textContent = `${(v*100).toFixed(2)}%`;

      div.appendChild(label);
      div.appendChild(percentage);
      resEl.appendChild(div);
    });

    const confidence = maxProb * 100;
    if (confidenceBar) confidenceBar.style.width = `${confidence}%`;
    if (confidenceText) confidenceText.textContent = `${confidence.toFixed(1)}% confidence`;
  }

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg;
    const apiInfo = document.getElementById('apiInfo');
    if (apiInfo) apiInfo.textContent = `API: ${window.API_BASE}`;
  }

  async function predict(){
    let seq = ensureLen(seqEl.value.trim(), 200);
    if(seq.length !== 200){
      alert('Sequence must be exactly 200bp');
      return;
    }
    seqEl.value = seq;
    updateStats(seq);
    renderSeq(seq);

    setStatus('Analyzing sequence...');
    if (spinner) spinner.style.display = 'inline-block';

    try{
      const rsp = await fetch(`${window.API_BASE}/predict`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({sequence: seq})
      });
      if(!rsp.ok){
        const text = await rsp.text();
        throw new Error(text || `HTTP ${rsp.status}`);
      }
      const data = await rsp.json();
      if (statusEl) statusEl.textContent = `Predicted class: ${data.prediction_label}`;
      showProbs(data);
    }catch(e){
      setStatus('Error contacting API');
      resEl.innerHTML = `<div class="prob" style="border-color:#dc2626;">
        Connection failed. Ensure the API is running at ${window.API_BASE}.
        <br/><small>${e.message}</small>
      </div>`;
    } finally {
      if (spinner) spinner.style.display = 'none';
    }
  }

  function loadExample(type) {
    let seq = pickCurated(type);
    if(!seq) seq = examples[type];
    if(!seq || type === 'Other') seq = randomSeq(200);
    seq = ensureLen(seq, 200);
    seqEl.value = seq;
    updateStats(seq);
    renderSeq(seq);
    if (statusEl) statusEl.textContent = `Loaded ${type} example sequence`;
  }

  // Event listeners
  const fillRandomBtn = document.getElementById('fillRandom');
  if (fillRandomBtn) fillRandomBtn.onclick = ()=>{
    const seq = randomSeq(200);
    seqEl.value = seq;
    updateStats(seq);
    renderSeq(seq);
  };

  const cleanBtn = document.getElementById('clean');
  if (cleanBtn) cleanBtn.onclick = ()=>{
    const cleaned = ensureLen(cleanSeq(seqEl.value), 200);
    seqEl.value = cleaned;
    updateStats(cleaned);
    renderSeq(cleaned);
  };

  const copyBtn = document.getElementById('copySeq');
  if (copyBtn) copyBtn.onclick = ()=>{
    navigator.clipboard.writeText(seqEl.value || '');
    setStatus('Sequence copied to clipboard');
  };

  const applyVarBtn = document.getElementById('applyVar');
  if (applyVarBtn) applyVarBtn.onclick = ()=>{
    let s = ensureLen((seqEl.value || '').trim(), 200);
    const p = Math.max(1, Math.min(200, parseInt((posEl && posEl.value) || '1',10)));
    const b = (baseEl && baseEl.value) || 'A';
    s = s.substring(0, p-1) + b + s.substring(p);
    seqEl.value = s;
    updateStats(s);
    renderSeq(s);
    setStatus(`Applied ${b} at position ${p}`);
  };

  const predictBtn = document.getElementById('predict');
  if (predictBtn) predictBtn.onclick = predict;

  document.querySelectorAll('.example-btn').forEach(btn => {
    btn.onclick = () => loadExample(btn.dataset.type);
  });

  document.querySelectorAll('.gene-region').forEach(region => {
    region.onclick = () => {
      const gene = region.dataset.gene;
      loadExample(gene);
      region.style.transform = 'scale(1.05)';
      setTimeout(() => region.style.transform = '', 200);
    };
  });

  const track = document.querySelector('.chr-track');
  const zoomInBtn = document.getElementById('zoomIn');
  if (zoomInBtn) zoomInBtn.onclick = ()=>{
    zoomLevel = Math.min(3, zoomLevel + 0.5);
    const zl = document.getElementById('zoomLevel');
    if (zl) zl.textContent = `${zoomLevel}x`;
    if (track) track.style.transform = `scaleX(${zoomLevel})`;
  };
  const zoomOutBtn = document.getElementById('zoomOut');
  if (zoomOutBtn) zoomOutBtn.onclick = ()=>{
    zoomLevel = Math.max(0.5, zoomLevel - 0.5);
    const zl = document.getElementById('zoomLevel');
    if (zl) zl.textContent = `${zoomLevel}x`;
    if (track) track.style.transform = `scaleX(${zoomLevel})`;
  };

  if (seqEl) seqEl.oninput = () => {
    const s = ensureLen(seqEl.value, 200);
    updateStats(s);
    renderSeq(s);
  };

  if (posEl) posEl.oninput = () => renderSeq(ensureLen(seqEl.value, 200));

  // Initialize
  (async () => {
    await loadCurated();
    const initialSeq = randomSeq(200);
    if (seqEl) {
      seqEl.value = initialSeq;
      updateStats(initialSeq);
      renderSeq(initialSeq);
    }
    setStatus('Ready');
  })();
})();

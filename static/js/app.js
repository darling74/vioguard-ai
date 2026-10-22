/* VIOGUARD AI front-end: upload -> real progress -> result. No framework, no innerHTML with server data. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const views = { upload: $('viewUpload'), progress: $('viewProgress'), result: $('viewResult') };
  const state = { file: null, objectUrl: null, maxBytes: 200 * 1024 * 1024, exts: ['mp4', 'avi', 'mov'], modelOk: true, busy: false, poll: null };

  /* ---------- helpers ---------- */
  const show = (name) => Object.entries(views).forEach(([k, v]) => { v.hidden = k !== name; });
  const fmtBytes = (b) => (b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(b / 1024)) + ' KB');
  const fmtTime = (s) => { s = Math.max(0, s); const m = Math.floor(s / 60); return m ? `${m}m ${(s % 60).toFixed(0)}s` : `${s.toFixed(1)}s`; };
  const pct = (x, d = 1) => (x * 100).toFixed(d) + '%';
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  function showError(msg) { const e = $('uploadError'); e.textContent = msg; e.hidden = !msg; }
  function setObjectUrl(file) { if (state.objectUrl) URL.revokeObjectURL(state.objectUrl); state.objectUrl = file ? URL.createObjectURL(file) : null; return state.objectUrl; }

  /* ---------- navigation ---------- */
  const toggle = $('navToggle'), links = $('navLinks');
  toggle.addEventListener('click', () => { const open = links.classList.toggle('open'); toggle.setAttribute('aria-expanded', String(open)); });
  links.addEventListener('click', (e) => { if (e.target.tagName === 'A') { links.classList.remove('open'); toggle.setAttribute('aria-expanded', 'false'); } });

  /* ---------- model health ---------- */
  async function loadHealth() {
    const pill = $('modelPill'), text = $('modelPillText'), notice = $('modelNotice');
    try {
      const r = await fetch('/api/health', { cache: 'no-store' });
      const h = await r.json();
      const m = h.model, s = h.settings;
      state.maxBytes = s.max_upload_mb * 1048576; state.exts = s.allowed_extensions; $('maxMb').textContent = s.max_upload_mb;
      const ready = m.state === 'ready' || m.state === 'not_loaded';
      state.modelOk = m.available;
      pill.dataset.state = m.state === 'ready' ? 'ready' : (m.available ? 'wait' : 'bad');
      text.textContent = m.state === 'ready' ? `Model ready (${m.device})` : ready || m.state === 'loading' ? 'Model loading...' : 'Model unavailable';
      notice.hidden = m.available;
      if (!m.available) {
        notice.textContent = '';
        notice.append(el('strong', '', 'Model not ready. '), document.createTextNode(m.message + ' '));
      }
      if (m.state === 'loading' || m.state === 'not_loaded') setTimeout(loadHealth, 2500);
    } catch (e) {
      pill.dataset.state = 'bad'; text.textContent = 'Server unreachable';
    }
  }

  /* ---------- file selection ---------- */
  const dz = $('dropzone'), input = $('fileInput');
  function validate(file) {
    const ext = (file.name.split('.').pop() || '').toLowerCase();
    if (!state.exts.includes(ext)) return `Unsupported file type ".${ext}". Please use ${state.exts.map((x) => x.toUpperCase()).join(', ')}.`;
    if (file.size === 0) return 'This file is empty.';
    if (file.size > state.maxBytes) return `File is ${fmtBytes(file.size)}; the limit is ${fmtBytes(state.maxBytes)}.`;
    return '';
  }
  function chooseFile(file) {
    showError('');
    if (!file) return;
    const problem = validate(file);
    if (problem) { showError(problem); return; }
    state.file = file;
    const url = setObjectUrl(file), v = $('previewVideo');
    v.src = url;
    $('fileName').textContent = file.name;
    $('fileInfo').textContent = fmtBytes(file.size);
    v.onloadedmetadata = () => { $('fileInfo').textContent = `${fmtBytes(file.size)} · ${fmtTime(v.duration)} · ${v.videoWidth}×${v.videoHeight}`; };
    v.onerror = () => { $('fileInfo').textContent = `${fmtBytes(file.size)} · preview unavailable in this browser (analysis may still work)`; };
    dz.hidden = true; $('selected').hidden = false;
  }
  function clearFile() { state.file = null; setObjectUrl(null); input.value = ''; $('previewVideo').removeAttribute('src'); $('selected').hidden = true; dz.hidden = false; showError(''); }

  $('browseBtn').addEventListener('click', (e) => { e.stopPropagation(); input.click(); });
  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => chooseFile(input.files[0]));
  ['dragenter', 'dragover'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', (e) => { const f = e.dataTransfer.files; if (f.length > 1) showError('Please drop one video at a time.'); if (f.length) chooseFile(f[0]); });
  $('clearBtn').addEventListener('click', clearFile);

  /* ---------- progress ---------- */
  function setProgress(title, detail, percent, step) {
    $('progressTitle').textContent = title; $('progressDetail').textContent = detail;
    const p = Math.max(0, Math.min(100, percent));
    $('progressFill').style.width = p + '%'; $('progressPct').textContent = p.toFixed(0) + '%';
    $('progressBar').setAttribute('aria-valuenow', p.toFixed(0));
    const order = ['upload', 'decoding', 'inference', 'done'], idx = order.indexOf(step);
    document.querySelectorAll('#progressSteps li').forEach((li) => {
      const i = order.indexOf(li.dataset.step);
      li.classList.toggle('done', i < idx); li.classList.toggle('active', i === idx);
    });
  }

  function fail(message) {
    state.busy = false; clearInterval(state.poll); show('upload'); showError(message); $('analyzeBtn').disabled = false;
  }

  /* ---------- analysis ---------- */
  function startAnalysis() {
    if (!state.file || state.busy) return;
    if (!state.modelOk) { showError('The model is not available yet. See the notice above.'); return; }
    state.busy = true; showError(''); show('progress'); setProgress('Uploading video…', state.file.name, 0, 'upload');
    const fd = new FormData(); fd.append('video', state.file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/analyze');
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) setProgress('Uploading video…', `${fmtBytes(e.loaded)} of ${fmtBytes(e.total)}`, (e.loaded / e.total) * 15, 'upload'); };
    xhr.onerror = () => fail('Network error while uploading. Is the server still running?');
    xhr.onload = () => {
      let body = {}; try { body = JSON.parse(xhr.responseText); } catch (_) { /* non-JSON */ }
      if (xhr.status !== 202 || !body.job_id) { fail((body.error && body.error.message) || `Upload failed (HTTP ${xhr.status}).`); return; }
      poll(body.job_id);
    };
    xhr.send(fd);
  }

  const STAGE = {
    queued: ['Waiting in queue…', 'Another video is being analysed', 'decoding'],
    probing: ['Validating video…', 'Checking that frames decode', 'decoding'],
    decoding: ['Extracting frames…', 'Decoding and resizing the sampled frames', 'decoding'],
    inference: ['Analysing with X3D-M…', '', 'inference'],
    done: ['Finishing…', '', 'done'],
  };
  function poll(jobId) {
    let failures = 0;
    state.poll = setInterval(async () => {
      try {
        const r = await fetch(`/api/jobs/${jobId}`, { cache: 'no-store' });
        const j = await r.json();
        if (!r.ok) throw new Error((j.error && j.error.message) || 'Job lookup failed');
        failures = 0;
        if (j.state === 'error') { fail(j.error.message); return; }
        if (j.state === 'done') { clearInterval(state.poll); setProgress('Complete', '', 100, 'done'); state.busy = false; setTimeout(() => renderResult(j.result), 250); return; }
        const p = j.progress, s = STAGE[p.stage] || STAGE.queued;
        const detail = p.stage === 'inference' ? `Segment ${p.segments_done} of ${p.segments_total}` : s[1];
        setProgress(s[0], detail, Math.max(15, p.percent), s[2]);
      } catch (err) { if (++failures > 5) fail('Lost connection to the server: ' + err.message); }
    }, 600);
  }

  /* ---------- warning sound ---------- */
 function playWarningTone() {
  const warningSound = new Audio("/static/sounds/warning.mp3");

  warningSound.currentTime = 0;

  return warningSound.play();
}

  /* ---------- result ---------- */
  function renderResult(r) {
    const violent = r.label === 'violence';
    const verdict = $('verdict');
    verdict.className = 'verdict ' + (violent ? 'violence' : 'safe');
    $('verdictIcon').textContent = violent ? '!' : '✓';
    $('verdictLabel').textContent = r.verdict;
    $('verdictSub').textContent = violent ? 'Possible violent activity was detected in this video.' : 'No violent activity was detected in this video.';
    const soundBtn = $('soundBtn');
    soundBtn.hidden = !violent;
    if (violent) {
      // Browser autoplay rules may block sound after long-running analysis. If so,
      // the visible button lets the user play the warning with one click.
      playWarningTone().catch(() => { soundBtn.hidden = false; });
    }
    $('scoreValue').textContent = pct(r.violence_probability);
    const gf = $('gaugeFill'); gf.className = 'gauge-fill ' + (violent ? 'violence' : 'safe'); gf.style.width = '0%';
    requestAnimationFrame(() => requestAnimationFrame(() => { gf.style.width = pct(Math.max(r.violence_probability, 0.01), 2); }));
    $('gaugeThr').style.left = pct(r.threshold, 2); $('gaugeThrLabel').textContent = `threshold ${pct(r.threshold, 0)}`;
    $('explanation').textContent = r.explanation;
    $('statConf').textContent = pct(r.confidence);
    $('statCert').textContent = r.certainty;
    $('statTime').textContent = fmtTime(r.processing_time_sec);
    $('statDur').textContent = fmtTime(r.video.duration_sec);
    $('statSeg').textContent = r.segment_count;
    $('statFrames').textContent = r.frames_analyzed;
    $('statRes').textContent = `${r.video.width}×${r.video.height} @ ${r.video.fps.toFixed(1)}`;
    $('statAgg').textContent = r.aggregation;

    const rv = $('resultVideo'); rv.src = state.objectUrl || '';
    const tl = $('timeline'); tl.replaceChildren();
    r.segments.forEach((s) => {
      const b = el('button', 'tl' + (s.violence_probability >= r.threshold ? ' hot' : ''));
      b.type = 'button'; b.style.height = Math.max(8, s.violence_probability * 100) + '%';
      b.title = `${s.start_sec}s–${s.end_sec}s: ${pct(s.violence_probability)} violence`;
      b.setAttribute('aria-label', b.title); b.setAttribute('role', 'listitem');
      b.addEventListener('click', () => { rv.currentTime = s.start_sec; rv.play().catch(() => {}); });
      tl.append(b);
    });
    const body = $('segBody'); body.replaceChildren();
    r.segments.forEach((s) => {
      const tr = el('tr');
      const tag = el('span', 'tag ' + (s.label === 'violence' ? 'v' : 'n'), s.label === 'violence' ? 'Violence' : 'Non-violence');
      const last = el('td'); last.append(tag);
      tr.append(el('td', '', String(s.index + 1)), el('td', '', `${s.start_sec}s – ${s.end_sec}s`), el('td', '', pct(s.violence_probability, 2)), last);
      body.append(tr);
    });
    show('result'); views.result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  $('soundBtn').addEventListener('click', () => {
    playWarningTone().catch(() => { showError('Warning sound could not be played in this browser.'); });
  });
  $('analyzeBtn').addEventListener('click', startAnalysis);
  $('againBtn').addEventListener('click', () => { $('resultVideo').removeAttribute('src'); clearFile(); show('upload'); $('analyze').scrollIntoView({ behavior: 'smooth' }); });
  window.addEventListener('beforeunload', () => setObjectUrl(null));
  loadHealth();
})();

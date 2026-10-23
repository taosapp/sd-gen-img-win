/* genimg Web 界面前端（无框架、无 CDN） */

const $ = (id) => document.getElementById(id);

const state = {
  catalog: null,
  mode: 'txt2img',
  initFile: null,
  jobId: null,
  timer: null,
  busy: false,
  ratio: 'auto',
  ratio: 'auto',
  pixels: 512,
  sbRatio: '9:16',
  sbTimer: null,
};

/* ---------------------------------------------------------------- 目录信息 */
async function loadCatalog() {
  const res = await fetch('/api/catalog');
  const data = await res.json();
  state.catalog = data;

  const hw = data.hardware;
  $('hw').innerHTML =
    `<b>${hw.name}</b> · ${hw.cc} · ${hw.vram_gb} GB · 驱动 ${hw.driver}<br>` +
    `torch ${hw.torch} / CUDA ${hw.cuda} · fp8 ${hw.fp8 ? '支持' : '不支持'}`;

  // 模型
  const modelSel = $('model');
  modelSel.innerHTML = '';
  data.models.forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m.key;
    opt.textContent = `${m.label}${m.cached ? '' : '（未下载）'}`;
    opt.title = `${m.repo} · 原生 ${m.native}px\n${m.note}`;
    modelSel.appendChild(opt);
  });

  // 精度：不可用的不列出（硬件不支持 + 该模型实测否决），由 syncPrecisionOptions 按模型重建

  // 默认值：优先用上次选择，否则用服务端默认
  const last = JSON.parse(localStorage.getItem('genimg.last') || '{}');
  modelSel.value = last.model || data.defaults.model;

  syncModel();
  setRatio(state.ratio || 'auto');
  setPixels(state.pixels || 512);

  await loadHistory();
  refreshAdvice();
}

function currentModel() {
  return state.catalog.models.find((m) => m.key === $('model').value);
}

/* 只列出真正能跑的精度：硬件不支持（fp8）和该模型实测否决（SDXL 的 fp32/tf32）都不出现 */
function syncPrecisionOptions(m) {
  const sel = $('precision');
  const prev = sel.value || state.precision;
  const blocked = (m && m.blocked_precisions) || [];
  const usable = state.catalog.precisions.filter((p) => p.ok && !blocked.includes(p.key));

  sel.innerHTML = '';
  usable.forEach((p) => {
    const opt = document.createElement('option');
    opt.value = p.key;
    opt.textContent = p.label;
    opt.title = p.note;
    sel.appendChild(opt);
  });

  const values = usable.map((p) => p.key);
  const last = JSON.parse(localStorage.getItem('genimg.last') || '{}');
  const fallback = (state.catalog && state.catalog.defaults && state.catalog.defaults.precision) || '';
  let want = values[0] || '';
  if (values.includes(fallback)) want = fallback;
  if (values.includes(last.precision)) want = last.precision;
  if (values.includes(prev)) want = prev;
  sel.value = want;
  sel.disabled = values.length === 0;
  state.precision = want;
}

function syncModel() {
  const m = currentModel();
  if (!m) return;

  syncPrecisionOptions(m);

  // 画质挡位按模型过滤
  const gearSel = $('gear');
  const prev = gearSel.value;
  gearSel.innerHTML = '';
  state.catalog.gears
    .filter((g) => m.gears.includes(g.key))
    .forEach((g) => {
      const opt = document.createElement('option');
      opt.value = g.key;
      opt.textContent = `${g.label}（${g.steps} 步 / CFG ${g.guidance}）`;
      opt.title = g.note;
      gearSel.appendChild(opt);
    });
  const valid = [...gearSel.options].map((o) => o.value);
  gearSel.value = valid.includes(prev) ? prev : (m.default_gear || valid[0]);

  $('statNote').textContent = m.note;

  syncGear();
}

function syncGear() {
  const key = $('gear').value;
  const g = state.catalog.gears.find((x) => x.key === key);
  if (!g) return;
  $('steps').placeholder = `自动（${g.steps}）`;
  $('guidance').placeholder = `自动（${g.guidance}）`;
  $('gearHint').textContent = g.note;
}

/* ---------------------------------------------------------------- 交互 */
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('#tabs button').forEach((b) => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  const sb = mode === 'storyboard';
  $('normalPanel').hidden = sb;
  $('sbPanel').hidden = !sb;
  $('results').classList.toggle('storyboard', sb);
  $('drop').hidden = mode !== 'img2img';
  $('strengthWrap').hidden = mode !== 'img2img';
  $('btnRecipe').hidden = mode !== 'img2img';
  if (!sb) scheduleAdvice();
}

/* ---------------------------------------------------------------- 故事板 */
const SB_COUNT = 10;
const SB_DEFAULT = 3;

function sbRows() {
  return [...document.querySelectorAll('#sbInputs .sb-item')];
}

function updateSbAdd() {
  const btn = $('sbAdd');
  if (!btn) return;
  const n = sbRows().length;
  const full = n >= SB_COUNT;
  btn.disabled = full;
  btn.textContent = full ? `已达上限 ${SB_COUNT} 个` : `＋ 添加镜头（${n}/${SB_COUNT}）`;
}

function appendSbRow() {
  const box = $('sbInputs');
  if (!box || sbRows().length >= SB_COUNT) return;
  const n = sbRows().length + 1;
  const row = document.createElement('div');
  row.className = 'sb-item';
  const idx = document.createElement('div');
  idx.className = 'idx';
  idx.textContent = String(n);
  const ta = document.createElement('textarea');
  ta.rows = 2;
  ta.placeholder = `镜头 ${n} 的提示词（英文）`;
  row.appendChild(idx);
  row.appendChild(ta);
  box.appendChild(row);
  updateSbAdd();
}

function buildSbInputs() {
  const box = $('sbInputs');
  if (!box) return;
  box.innerHTML = '';
  for (let i = 0; i < SB_DEFAULT; i += 1) appendSbRow();
}

function setSbRatio(ratio) {
  state.sbRatio = ratio;
  document.querySelectorAll('#sbRatios button').forEach((b) => {
    b.classList.toggle('active', b.dataset.ratio === ratio);
  });
}

async function runStoryboard() {
  const prompts = [...document.querySelectorAll('#sbInputs textarea')]
    .map((t) => t.value.trim())
    .filter(Boolean);
  if (!prompts.length) {
    alert('至少填写一个镜头提示词');
    return;
  }
  if (state.busy) return;

  state.busy = true;
  $('sbGo').disabled = true;
  $('results').innerHTML = '';
  $('metrics').innerHTML = '';
  $('barWrap').hidden = false;
  $('bar').style.width = '2%';
  setBadge('提交中', 'run');
  statusHtml(`<div class="muted">已提交 ${prompts.length} 个镜头，排队生成…</div>`);

  const fd = new FormData();
  fd.append('prompts', JSON.stringify(prompts));
  fd.append('ratio', state.sbRatio || '9:16');
  fd.append('gear', $('sbGear') ? $('sbGear').value : 'lightning');
  try {
    const res = await fetch('/api/storyboard', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '提交失败');
    $('sbNote').textContent = `批次 ${data.batch_id} · ${data.width}×${data.height} · 共 ${data.total} 张`;
    pollStoryboard(data.batch_id, data.total, data.width, data.height);
  } catch (err) {
    state.busy = false;
    $('sbGo').disabled = false;
    setBadge('失败', 'err');
    statusHtml(`<div class="error-box">${escapeHtml(err.message)}</div>`);
  }
}

function pollStoryboard(batchId, total, width, height) {
  const tick = async () => {
    try {
      const res = await fetch(`/api/storyboard/${batchId}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '查询失败');
      renderStoryboard(data);
      const finished = data.done + data.failed;
      $('bar').style.width = `${Math.round((finished / data.total) * 100)}%`;
      setBadge(`${finished}/${data.total}`, data.failed ? 'err' : 'run');
      statusHtml(
        `<div class="kv"><div>画幅 <b>${width}×${height}</b></div>` +
          `<div>已完成 <b>${finished}/${data.total}</b></div>` +
          `<div>失败 <b>${data.failed}</b></div></div>`
      );
      if (finished >= data.total) {
        clearInterval(state.sbTimer);
        setBadge('完成', data.failed ? 'err' : 'ok');
        state.busy = false;
        $('sbGo').disabled = false;
        loadHistory();
      }
    } catch (err) {
      /* 单次轮询失败不终止，下一轮继续 */
    }
  };
  tick();
  state.sbTimer = setInterval(tick, 800);
}

function renderStoryboard(data) {
  const box = $('results');
  box.innerHTML = '';
  data.items.forEach((it) => {
    const fig = document.createElement('figure');
    if (it.url) {
      const img = document.createElement('img');
      img.src = it.url;
      img.alt = `镜头 ${it.shot}`;
      img.onclick = () => window.open(it.url, '_blank');
      fig.appendChild(img);
    } else {
      const ph = document.createElement('div');
      ph.className = 'sb-shot';
      ph.textContent =
        it.status === 'error' ? '失败' : it.status === 'running' ? '生成中…' : '排队中';
      fig.appendChild(ph);
    }
    const cap = document.createElement('figcaption');
    cap.textContent = `镜头 ${it.shot}${it.status === 'error' ? ' · 失败' : ''}`;
    fig.appendChild(cap);
    box.appendChild(fig);
  });
}

/* 与后端 size_for_ratio 同算法的前端版：只用于提示“实际会出多大” */
function bucketFor(rw, rh, base) {
  let best = null;
  for (let w = 256; w <= 1536; w += 64) {
    const h = Math.round((w * rh) / rw / 64) * 64;
    if (h < 256 || h > 1536) continue;
    const ratioErr = Math.abs(Math.log(w / h / (rw / rh)));
    let areaErr = Math.abs(Math.log((w * h) / (base * base)));
    if (w * h > base * base) areaErr *= 2.5;
    const score = ratioErr * 20 + areaErr;
    if (!best || score < best.score) best = { score, w, h };
  }
  return best ? [best.w, best.h] : [base, base];
}

function refreshSizeHint() {
  const hint = $('sizeHint');
  if (!hint) return;
  if (state.ratio === 'auto') {
    hint.innerHTML = state.initFile
      ? '自动：跟随参考图比例（等比装入，不拉伸）'
      : '自动：图生图跟随参考图；文生图读提示词里的 <b>3:4</b>，都没有则用模型原生比例';
    return;
  }
  const [rw, rh] = state.ratio.split(':').map(Number);
  const [w, h] = bucketFor(rw, rh, state.pixels);
  hint.innerHTML = `将输出 <b>${w}×${h}</b>（${state.ratio} · ${state.pixels} 档，已对齐到 64 倍数）`;
}

function setRatio(ratio) {
  state.ratio = ratio;
  document.querySelectorAll('#ratios button').forEach((b) => {
    b.classList.toggle('active', b.dataset.ratio === ratio);
  });
  refreshSizeHint();
}

function setPixels(px) {
  state.pixels = Number(px) || 512;
  document.querySelectorAll('#pixels button').forEach((b) => {
    b.classList.toggle('active', Number(b.dataset.px) === state.pixels);
  });
  refreshSizeHint();
}

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    alert('请选择图片文件');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    alert('图片超过 20MB');
    return;
  }
  state.initFile = file;
  const url = URL.createObjectURL(file);
  $('dropInner').innerHTML = `<img src="${url}" alt="初始图"><div class="hint">${file.name} · 点击可更换</div>`;
  // 读一下真实宽高，把「自动＝跟随参考图」具体化到界面
  const probe = new Image();
  probe.onload = () => {
    if (state.ratio === 'auto' && $('sizeHint')) {
      $('sizeHint').innerHTML =
        `自动：将跟随参考图比例 <b>${probe.naturalWidth}×${probe.naturalHeight}</b>` +
        `（等比装入，不拉伸）`;
    }
  };
  probe.src = url;
}

function setBadge(text, cls) {
  const b = $('badge');
  if (!b) return;
  b.textContent = text;
  b.className = `badge ${cls}`;
}

function statusHtml(html) {
  const st = $('status');
  st.innerHTML = `<div><span class="badge idle" id="badge">空闲</span></div>` + html;
}

/* ---------------------------------------------------------------- 生成 */
async function submit() {
  if (state.busy) return;
  const prompt = $('prompt').value.trim();
  if (!prompt) {
    alert('请填写提示词');
    return;
  }
  if (state.mode === 'img2img' && !state.initFile) {
    alert('图生图需要先上传一张图片');
    return;
  }

  const fd = new FormData();
  fd.append('mode', state.mode);
  fd.append('prompt', prompt);
  fd.append('negative', $('negative').value || '');
  fd.append('model', $('model').value);
  fd.append('precision', $('precision').value);
  fd.append('gear', $('gear').value);
  fd.append('ratio', state.ratio || 'auto');
  fd.append('pixels', state.pixels || 512);
  fd.append('steps', $('steps').value);
  fd.append('guidance', $('guidance').value);
  fd.append('seed', $('seed').value);
  fd.append('batch', $('batch').value);
  fd.append('strength', $('strength').value);
  if (state.mode === 'img2img' && state.initFile) fd.append('image', state.initFile);

  localStorage.setItem('genimg.last', JSON.stringify({
    model: $('model').value,
    precision: $('precision').value,
    ratio: state.ratio,
    pixels: state.pixels,
  }));

  state.busy = true;
  $('go').disabled = true;
  $('go').textContent = '生成中…';
  $('results').innerHTML = '';
  $('metrics').innerHTML = '';
  $('barWrap').hidden = false;
  $('bar').style.width = '2%';
  statusHtml('<div class="muted">任务已提交，等待 GPU…</div>');

  try {
    const res = await fetch('/api/generate', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '提交失败');
    state.jobId = data.id;
    if (data.warning) {
      $('status').insertAdjacentHTML('beforeend', `<div class="warn-box">${escapeHtml(data.warning)}</div>`);
    }
    poll();
  } catch (err) {
    fail(err.message);
  }
}

function poll() {
  clearInterval(state.timer);
  const tick = async () => {
    try {
      const res = await fetch(`/api/jobs/${state.jobId}`);
      if (!res.ok) throw new Error('任务状态获取失败');
      const job = await res.json();

      if (job.status === 'queued') {
        setBadge('排队中', 'idle');
        $('bar').style.width = '3%';
        statusHtml('<div class="muted">前面还有任务在跑，稍候…</div>');
        return;
      }
      if (job.status === 'running') {
        setBadge('生成中', 'run');
        const pct = job.step_total ? Math.round((job.step_now / job.step_total) * 100) : 8;
        $('bar').style.width = `${Math.max(4, pct)}%`;
        statusHtml(
          `<div class="kv">
             <div>进度 <b>${job.step_now}/${job.step_total || '?'}</b> 步</div>
             <div>已用 <b>${job.elapsed.toFixed(1)}</b> s</div>
             <div>模型 <b>${job.params.model}</b></div>
             <div>精度 <b>${job.params.precision}</b></div>
           </div>`
        );
        return;
      }
      if (job.status === 'done') {
        clearInterval(state.timer);
        done(job);
        return;
      }
      if (job.status === 'error') {
        clearInterval(state.timer);
        fail(job.error);
        return;
      }
    } catch (err) {
      clearInterval(state.timer);
      fail(err.message);
    }
  };
  tick();
  state.timer = setInterval(tick, 700);
}

function done(job) {
  setBadge('完成', 'ok');
  $('bar').style.width = '100%';
  const m = job.meta || {};
  const st = m.stage_seconds || {};
  const vram = m.vram || {};

  statusHtml(
    `<div class="kv">
       <div>推理总计 <b>${(st.infer_total || 0).toFixed(2)}</b> s</div>
       <div>去噪 <b>${(st.denoise || 0).toFixed(2)}</b> s</div>
       <div>编解码 <b>${(st.encode_decode || 0).toFixed(2)}</b> s</div>
       <div>权重加载 <b>${(m.load_seconds || 0).toFixed(1)}</b> s</div>
       <div>显存峰值 <b>${(vram.peak_allocated_gb || 0).toFixed(2)}</b> GB</div>
       <div>放置 <b>${m.placement || '-'}</b></div>
     </div>`
  );

  const ok = (st.infer_total || 0) <= 30;
  $('metrics').innerHTML =
    `<div class="kv">
       <div>速度验收 <b class="${ok ? 'verdict-ok' : 'verdict-bad'}">${ok ? '满足' : '超出'} 30 秒预算</b></div>
       <div>步数 / CFG <b>${m.steps} / ${m.guidance}</b></div>
       <div>种子 <b>${m.seed === null ? '随机' : m.seed}</b></div>
       <div>调度器 <b>${m.scheduler}</b></div>
     </div>`;

  $('results').innerHTML = '';
  (job.images || []).forEach((src, i) => {
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = src;
    img.alt = `结果 ${i + 1}`;
    img.onclick = () => window.open(src, '_blank');
    const cap = document.createElement('figcaption');
    cap.textContent = `${m.width}×${m.height} · ${m.model_key} / ${m.precision_gear} / ${m.quality_gear}` +
      (m.seed !== null ? ` · seed ${Number(m.seed) + i}` : '');
    fig.appendChild(img);
    fig.appendChild(cap);
    $('results').appendChild(fig);
  });

  if (job.warning) {
    $('status').insertAdjacentHTML('beforeend', `<div class="warn-box">${escapeHtml(job.warning)}</div>`);
  }

  // 中文提示词被自动翻译时，把实际用的英文露出来（透明可查）
  if (m.prompt_translated) {
    const tr = m.translation || {};
    const unknown = (tr.unknown || []).length
      ? `；未翻译片段：${escapeHtml(tr.unknown.join('、'))}`
      : '';
    $('status').insertAdjacentHTML(
      'beforeend',
      `<div class="muted">中文已转英文（${escapeHtml(tr.method || 'dict')}）：${escapeHtml(m.prompt)}${unknown}</div>`
    );
  }

  state.busy = false;
  $('go').disabled = false;
  $('go').textContent = '生成';
  loadHistory();
}

function fail(message) {
  clearInterval(state.timer);
  setBadge('失败', 'err');
  $('bar').style.width = '0';
  statusHtml(`<div class="error-box">${escapeHtml(String(message))}</div>`);
  state.busy = false;
  $('go').disabled = false;
  $('go').textContent = '生成';
}

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const data = await res.json();
    const box = $('history');
    if (!data.items || data.items.length === 0) {
      box.innerHTML = '<span class="muted">还没有记录</span>';
      return;
    }
    box.innerHTML = '';
    data.items.forEach((job) => {
      (job.images || []).slice(0, 1).forEach((src) => {
        const img = document.createElement('img');
        img.src = src;
        img.title = `${job.prompt.slice(0, 60)}\n${job.params.model} / ${job.params.precision} / ${job.params.gear}`;
        img.onclick = () => window.open(src, '_blank');
        box.appendChild(img);
      });
    });
  } catch (_) { /* 历史失败不影响主流程 */ }
}

async function loadPreload() {
  const el = $('preloadNote');
  if (!el) return;
  try {
    const res = await fetch('/api/health');
    const h = await res.json();
    const p = h.preload || {};
    if (p.status === 'loading' || p.status === 'pending') {
      el.textContent = `模型预热中…（${p.model || ''} / ${p.precision || ''}），就绪后首次生成无需等待`;
      setTimeout(loadPreload, 1200);
    } else if (p.status === 'ready') {
      el.innerHTML = `<span style="color:var(--accent-2)">● 模型已就绪</span> ${p.model} / ${p.precision} / ${p.gear}（预热 ${p.seconds}s）`;
    } else if (p.status === 'error') {
      el.textContent = `预热未成功：${p.error}（不影响使用，首次生成时会重新加载）`;
    } else {
      el.textContent = '';
    }
  } catch (_) { /* 状态查不到不影响主流程 */ }
}

/* ---------------------------------------------------------------- 生成前体检 */
let adviceTimer = null;

function scheduleAdvice() {
  clearTimeout(adviceTimer);
  adviceTimer = setTimeout(refreshAdvice, 350);
}

async function refreshAdvice() {
  const box = $('adviceBox');
  if (!box) return;
  const params = new URLSearchParams({
    mode: state.mode,
    prompt: $('prompt').value,
    model: $('model').value,
    gear: $('gear').value,
    strength: $('strength').value,
    steps: $('steps').value,
    guidance: $('guidance').value,
  });
  try {
    const res = await fetch('/api/advice?' + params.toString());
    if (!res.ok) { box.innerHTML = ''; return; }
    const data = await res.json();
    box.innerHTML = (data.advice || []).map((t) => `<div class="warn-box">${escapeHtml(t)}</div>`).join('');
  } catch (_) {
    box.innerHTML = '';
  }
}

function applyRecipe() {
  $('model').value = 'sd15';
  syncModel();
  $('gear').value = 'balanced';
  syncGear();
  $('steps').value = '';
  $('guidance').value = '';
  $('strength').value = '0.5';
  $('strengthVal').textContent = '0.50';
  // 负向输入已在界面上禁用，这里不再回填默认值
  refreshAdvice();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ---------------------------------------------------------------- 绑定 */
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('#tabs button').forEach((b) => {
    b.onclick = () => setMode(b.dataset.mode);
  });

  $('model').onchange = () => { syncModel(); scheduleAdvice(); };
  $('gear').onchange = () => { syncGear(); scheduleAdvice(); };
  $('precision').onchange = () => {
    const p = state.catalog.precisions.find((x) => x.key === $('precision').value);
    $('statNote').textContent = p ? p.note : '';
  };
  $('prompt').oninput = scheduleAdvice;
  $('steps').oninput = scheduleAdvice;
  $('guidance').oninput = scheduleAdvice;
  $('btnRecipe').onclick = applyRecipe;

  buildSbInputs();
  document.querySelectorAll('#sbRatios button').forEach((b) => {
    b.onclick = () => setSbRatio(b.dataset.ratio);
  });
  $('sbGo').onclick = runStoryboard;
  $('sbAdd').onclick = appendSbRow;
  setSbRatio(state.sbRatio);

  document.querySelectorAll('#ratios button').forEach((b) => {
    b.onclick = () => setRatio(b.dataset.ratio);
  });
  document.querySelectorAll('#pixels button').forEach((b) => {
    b.onclick = () => setPixels(b.dataset.px);
  });

  $('drop').onclick = () => $('file').click();
  $('file').onchange = (e) => setFile(e.target.files[0]);
  ['dragenter', 'dragover'].forEach((ev) => $('drop').addEventListener(ev, (e) => {
    e.preventDefault();
    $('drop').classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((ev) => $('drop').addEventListener(ev, (e) => {
    e.preventDefault();
    $('drop').classList.remove('over');
  }));
  $('drop').addEventListener('drop', (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  $('strength').oninput = () => {
    $('strengthVal').textContent = Number($('strength').value).toFixed(2);
    scheduleAdvice();
  };

  $('go').onclick = submit;
  $('btnCatalog').onclick = loadCatalog;
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') submit();
  });

  // 支持 ?mode=storyboard / ?mode=img2img 直接打开对应模式
  const deep = new URLSearchParams(location.search).get('mode');
  setMode(deep === 'storyboard' || deep === 'img2img' ? deep : 'txt2img');
  loadCatalog();
  loadPreload();});

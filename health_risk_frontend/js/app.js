/**
 * js/app.js — UI Layer (Extended with Vitals & Labs Panels)
 *
 * KEY FIX: All getElementById calls are now inside DOMContentLoaded.
 * Calling getElementById at the top level (before DOM is parsed) returns
 * null — any later method call on null throws a TypeError that silently
 * stops the entire script, so no event listeners ever attach.
 */

// =============================================
// MEASUREMENT PANEL DEFINITIONS
// =============================================
const MEASUREMENT_PANELS = {
  cardiac: {
    label: 'Cardiac Measurements', icon: '❤️',
    vitals: [
      { id: 'v-systolic',   label: 'Systolic BP',    unit: 'mmHg',   min: 60,  max: 250, step: 1,   hint: 'Normal: 90–120' },
      { id: 'v-diastolic',  label: 'Diastolic BP',   unit: 'mmHg',   min: 40,  max: 150, step: 1,   hint: 'Normal: 60–80' },
      { id: 'v-heart-rate', label: 'Heart Rate',     unit: 'bpm',    min: 20,  max: 300, step: 1,   hint: 'Normal: 60–100' },
      { id: 'v-spo2',       label: 'SpO₂',           unit: '%',      min: 50,  max: 100, step: 0.1, hint: 'Normal: ≥ 96%' },
    ],
    labs: [
      { id: 'l-total-chol', label: 'Total Cholesterol', unit: 'mmol/L', min: 0, max: 15, step: 0.1, hint: 'Normal: < 5.0' },
      { id: 'l-ldl',        label: 'LDL Cholesterol',   unit: 'mmol/L', min: 0, max: 12, step: 0.1, hint: 'Optimal: < 2.6' },
      { id: 'l-hdl',        label: 'HDL Cholesterol',   unit: 'mmol/L', min: 0, max: 5,  step: 0.1, hint: 'Protective: ≥ 1.0' },
      { id: 'l-trig',       label: 'Triglycerides',     unit: 'mmol/L', min: 0, max: 20, step: 0.1, hint: 'Normal: < 1.7' },
    ]
  },
  metabolic: {
    label: 'Metabolic Measurements', icon: '🩸',
    vitals: [
      { id: 'v-weight', label: 'Body Weight', unit: 'kg', min: 10, max: 400, step: 0.1, hint: 'Used to calculate BMI' },
      { id: 'v-height', label: 'Height',      unit: 'cm', min: 50, max: 250, step: 0.1, hint: 'Used to calculate BMI' },
      { id: 'v-temp',   label: 'Temperature', unit: '°C', min: 30, max: 45,  step: 0.1, hint: 'Normal: 36.1–37.2°C' },
    ],
    labs: [
      { id: 'l-glucose',    label: 'Fasting Glucose',    unit: 'mmol/L', min: 0, max: 35, step: 0.1, hint: 'Normal: 3.9–5.5 | Diabetic: ≥ 7.0' },
      { id: 'l-hba1c',      label: 'HbA1c',              unit: '%',      min: 0, max: 20, step: 0.1, hint: 'Normal: < 5.7 | Diabetic: ≥ 6.5' },
      { id: 'l-total-chol', label: 'Total Cholesterol',  unit: 'mmol/L', min: 0, max: 15, step: 0.1, hint: 'Normal: < 5.0' },
      { id: 'l-trig',       label: 'Triglycerides',      unit: 'mmol/L', min: 0, max: 20, step: 0.1, hint: 'Normal: < 1.7' },
    ]
  },
  respiratory: {
    label: 'Respiratory Measurements', icon: '🫁',
    vitals: [
      { id: 'v-spo2',       label: 'SpO₂',             unit: '%',      min: 50, max: 100, step: 0.1, hint: 'Normal: ≥ 96% | Danger: < 90%' },
      { id: 'v-rr',         label: 'Respiratory Rate', unit: 'br/min', min: 5,  max: 60,  step: 1,   hint: 'Normal: 12–20' },
      { id: 'v-heart-rate', label: 'Heart Rate',       unit: 'bpm',    min: 20, max: 300, step: 1,   hint: 'Normal: 60–100' },
      { id: 'v-temp',       label: 'Temperature',      unit: '°C',     min: 30, max: 45,  step: 0.1, hint: 'Normal: 36.1–37.2°C' },
    ],
    labs: []
  },
  general: {
    label: 'General Health Measurements', icon: '🩺',
    vitals: [
      { id: 'v-systolic',   label: 'Systolic BP',    unit: 'mmHg', min: 60,  max: 250, step: 1,   hint: 'Normal: 90–120' },
      { id: 'v-diastolic',  label: 'Diastolic BP',   unit: 'mmHg', min: 40,  max: 150, step: 1,   hint: 'Normal: 60–80' },
      { id: 'v-heart-rate', label: 'Heart Rate',     unit: 'bpm',  min: 20,  max: 300, step: 1,   hint: 'Normal: 60–100' },
      { id: 'v-spo2',       label: 'SpO₂',           unit: '%',    min: 50,  max: 100, step: 0.1, hint: 'Normal: ≥ 96%' },
      { id: 'v-temp',       label: 'Temperature',    unit: '°C',   min: 30,  max: 45,  step: 0.1, hint: 'Normal: 36.1–37.2°C' },
      { id: 'v-weight',     label: 'Weight',         unit: 'kg',   min: 10,  max: 400, step: 0.1, hint: 'Optional — used for BMI' },
      { id: 'v-height',     label: 'Height',         unit: 'cm',   min: 50,  max: 250, step: 0.1, hint: 'Optional — used for BMI' },
    ],
    labs: [
      { id: 'l-glucose',    label: 'Fasting Glucose',   unit: 'mmol/L', min: 0, max: 35, step: 0.1, hint: 'Normal: 3.9–5.5' },
      { id: 'l-total-chol', label: 'Total Cholesterol', unit: 'mmol/L', min: 0, max: 15, step: 0.1, hint: 'Normal: < 5.0' },
    ]
  }
};

// =============================================
// DOM REFS — populated inside DOMContentLoaded
// =============================================
let inputSection, loadingSection, resultsSection, errorSection, analyzeBtn;

// =============================================
// PAGE LOAD — wire up everything here
// =============================================
document.addEventListener('DOMContentLoaded', async () => {

  // Grab DOM refs now that the page is fully parsed
  inputSection   = document.getElementById('input-section');
  loadingSection = document.getElementById('loading-section');
  resultsSection = document.getElementById('results-section');
  errorSection   = document.getElementById('error-section');
  analyzeBtn     = document.getElementById('analyze-btn');

  // ---- Chip toggle ----
  document.getElementById('chip-grid').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    chip.classList.toggle('selected');
  });

  // ---- Character counter ----
  const textarea  = document.getElementById('symptom-text');
  const charCount = document.getElementById('char-count');
  textarea.addEventListener('input', () => { charCount.textContent = textarea.value.length; });

  // ---- Focus area buttons ----
  // Use click on each button directly — more reliable than delegation
  // when inner <span> elements could intercept the event target.
  document.querySelectorAll('.focus-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const wasActive = btn.classList.contains('active');
      // Deactivate all
      document.querySelectorAll('.focus-btn').forEach(b => b.classList.remove('active'));
      if (wasActive) {
        // Clicking the active button again collapses the panel
        renderMeasurementPanel(null);
      } else {
        btn.classList.add('active');
        renderMeasurementPanel(btn.dataset.focus);
      }
    });
  });

  // ---- Analyze button ----
  analyzeBtn.addEventListener('click', handleAnalyze);

  // ---- Health check ----
  await refreshHealthStatus();
});

// =============================================
// MEASUREMENT PANEL RENDERER
// =============================================
function renderMeasurementPanel(focusKey) {
  const container = document.getElementById('measurement-panel');
  if (!focusKey || !MEASUREMENT_PANELS[focusKey]) {
    container.innerHTML = '';
    return;
  }

  const panel = MEASUREMENT_PANELS[focusKey];

  let html = `<div class="meas-panel">
    <div class="meas-panel-header">
      <span class="meas-icon">${panel.icon}</span>
      <span class="meas-title">${panel.label}</span>
      <span class="meas-hint-global">All fields optional — enter only what you have</span>
    </div>`;

  if (panel.vitals.length > 0) {
    html += `<div class="meas-section-label">Vital Signs</div><div class="meas-grid">`;
    panel.vitals.forEach(f => { html += buildFieldHTML(f); });
    html += `</div>`;
  }

  if (panel.labs.length > 0) {
    html += `<div class="meas-section-label">Lab Values</div><div class="meas-grid">`;
    panel.labs.forEach(f => { html += buildFieldHTML(f); });
    html += `</div>`;
  }

  html += `</div>`;
  container.innerHTML = html;

  // Attach live validation to every input
  container.querySelectorAll('.meas-input').forEach(input => {
    input.addEventListener('input', () => validateMeasInput(input));
  });
}

function buildFieldHTML(field) {
  return `
    <div class="meas-field">
      <label class="meas-label" for="${field.id}">${field.label}</label>
      <div class="meas-input-wrap">
        <input
          type="number"
          id="${field.id}"
          class="meas-input"
          placeholder="—"
          min="${field.min}"
          max="${field.max}"
          step="${field.step}"
          data-min="${field.min}"
          data-max="${field.max}"
          autocomplete="off"
        >
        <span class="meas-unit">${field.unit}</span>
      </div>
      <div class="meas-hint">${field.hint}</div>
      <div class="meas-field-warn" id="warn-${field.id}"></div>
    </div>`;
}

function validateMeasInput(input) {
  const warnEl = document.getElementById('warn-' + input.id);
  if (!warnEl) return;
  const val = parseFloat(input.value);
  if (isNaN(val) || input.value === '') {
    warnEl.textContent = '';
    input.classList.remove('input-warn', 'input-ok');
    return;
  }
  const min = parseFloat(input.dataset.min);
  const max = parseFloat(input.dataset.max);
  if (val < min || val > max) {
    warnEl.textContent = `⚠ Out of range (${min}–${max})`;
    input.classList.add('input-warn');
    input.classList.remove('input-ok');
  } else {
    warnEl.textContent = '';
    input.classList.remove('input-warn');
    input.classList.add('input-ok');
  }
}

// =============================================
// HEALTH STATUS
// =============================================
async function refreshHealthStatus() {
  const statusEl = document.getElementById('api-status');
  const textEl   = statusEl.querySelector('.status-text');
  statusEl.className = 'api-status';
  textEl.textContent = 'Checking...';
  const result = await checkHealth(getBaseUrl());
  if (result.ok) {
    statusEl.classList.add('connected');
    textEl.textContent = 'Backend connected';
  } else {
    statusEl.classList.add('error');
    textEl.textContent = 'Backend offline';
  }
}

// =============================================
// STEP BAR
// =============================================
function setActiveStep(n) {
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById(`step-${i}`);
    el.classList.remove('active', 'done');
    const circle = el.querySelector('.step-circle');
    if (i < n)      { el.classList.add('done');   circle.textContent = '✓'; }
    else if (i === n){ el.classList.add('active'); circle.textContent = i; }
    else             { circle.textContent = i; }
  }
}

// =============================================
// PIPELINE LOADER
// =============================================
const PIPELINE_MESSAGES = [
  { step: 'pl-nlp', msg: 'Running NLP — extracting symptom keywords...' },
  { step: 'pl-ml',  msg: 'Interpreting vitals & labs — building feature vector...' },
  { step: 'pl-ai',  msg: 'Claude AI reasoning — generating explanation...' },
];

let pipelineTimer = null, pipelineIndex = 0;

function startPipelineAnimation() {
  pipelineIndex = 0;
  resetPipeline();
  stepPipeline();
  pipelineTimer = setInterval(stepPipeline, 2500);
}

function stepPipeline() {
  if (pipelineIndex >= PIPELINE_MESSAGES.length) return;
  if (pipelineIndex > 0) {
    const prevId = PIPELINE_MESSAGES[pipelineIndex - 1].step;
    const ps = document.querySelector(`#${prevId} .pl-status`);
    if (ps) { ps.className = 'pl-status done'; ps.textContent = 'done ✓'; }
  }
  const { step, msg } = PIPELINE_MESSAGES[pipelineIndex];
  const statusEl = document.querySelector(`#${step} .pl-status`);
  if (statusEl) { statusEl.className = 'pl-status running'; statusEl.textContent = 'running...'; }
  document.getElementById('loading-msg').textContent = msg;
  pipelineIndex++;
}

function stopPipelineAnimation() {
  clearInterval(pipelineTimer);
  ['pl-nlp', 'pl-ml', 'pl-ai'].forEach(id => {
    const el = document.querySelector(`#${id} .pl-status`);
    if (el) { el.className = 'pl-status done'; el.textContent = 'done ✓'; }
  });
}

function resetPipeline() {
  ['pl-nlp', 'pl-ml', 'pl-ai'].forEach(id => {
    const el = document.querySelector(`#${id} .pl-status`);
    if (el) { el.className = 'pl-status idle'; el.textContent = 'waiting'; }
  });
}

// =============================================
// SECTION VISIBILITY
// =============================================
function showSection(name) {
  [inputSection, loadingSection, resultsSection, errorSection].forEach(s => s.classList.add('hidden'));
  if (name === 'input')   inputSection.classList.remove('hidden');
  if (name === 'loading') loadingSection.classList.remove('hidden');
  if (name === 'results') resultsSection.classList.remove('hidden');
  if (name === 'error')   errorSection.classList.remove('hidden');
}

// =============================================
// MAIN HANDLER
// =============================================
async function handleAnalyze() {
  const payload = buildPayload();
  if (!payload.symptom_text && payload.selected_chips.length === 0) {
    document.getElementById('symptom-text').style.borderColor = 'var(--color-high)';
    document.getElementById('symptom-text').focus();
    return;
  }
  document.getElementById('symptom-text').style.borderColor = '';
  analyzeBtn.disabled = true;
  showSection('loading');
  setActiveStep(2);
  startPipelineAnimation();

  const startTime = Date.now();
  const result = await analyzeSymptoms(payload, getBaseUrl());
  const elapsed = Date.now() - startTime;
  stopPipelineAnimation();

  if (!result.ok) {
    showError(result.error, result.detail);
    analyzeBtn.disabled = false;
    return;
  }

  renderResults(result.data, elapsed);
  showSection('results');
  setActiveStep(4);
  analyzeBtn.disabled = false;
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// =============================================
// RESET
// =============================================
function handleReset() {
  document.querySelectorAll('.chip.selected').forEach(c => c.classList.remove('selected'));
  document.getElementById('symptom-text').value = '';
  document.getElementById('char-count').textContent = '0';
  // Clear measurement inputs and collapse panel
  document.querySelectorAll('.meas-input').forEach(i => { i.value = ''; });
  document.querySelectorAll('.focus-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('measurement-panel').innerHTML = '';
  setActiveStep(1);
  showSection('input');
  window.scrollTo({ top: 0, behavior: 'smooth' });
  refreshHealthStatus();
}

// =============================================
// RENDER RESULTS
// =============================================
function renderResults(data, clientElapsedMs) {
  renderVitalsCard(data.vitals_summary);
  renderNLP(data.nlp);
  renderRisk(data.risk);
  renderExplanation(data.risk, data, clientElapsedMs);
  renderSuggestions(data.suggestions);
}

function renderVitalsCard(vitals_summary) {
  const card = document.getElementById('vitals-result-card');
  if (!card) return;
  if (!vitals_summary || !vitals_summary.flags || vitals_summary.flags.length === 0) {
    card.classList.add('hidden');
    return;
  }
  card.classList.remove('hidden');
  const flagsEl = document.getElementById('vitals-flags');
  flagsEl.innerHTML = vitals_summary.flags.map(flag => {
    const isHigh = /crisis|severe|hypoxaemia|failure|stage [234]/i.test(flag);
    const isMed  = /stage 1|elevated|borderline|tachycardia|tachypnoea|impaired|pre.diab/i.test(flag);
    const cls    = isHigh ? 'flag-high' : isMed ? 'flag-med' : 'flag-ok';
    return `<div class="vitals-flag ${cls}">${escapeHtml(flag)}</div>`;
  }).join('');
  document.getElementById('vitals-summary-text').textContent = vitals_summary.summary || '';
}

function renderNLP(nlp) {
  const kwContainer = document.getElementById('nlp-keywords');
  if (nlp.detected_keywords && nlp.detected_keywords.length > 0) {
    kwContainer.innerHTML = nlp.detected_keywords
      .map(kw => `<span class="kw-tag">${escapeHtml(kw)}</span>`).join('');
  } else {
    kwContainer.innerHTML = '<span class="kw-tag" style="background:var(--color-bg);color:var(--color-text-3)">No specific keywords detected</span>';
  }
  document.getElementById('nlp-summary').textContent = nlp.symptom_summary || '';
  const sevContainer = document.getElementById('nlp-severity');
  if (nlp.severity_indicators && nlp.severity_indicators.length > 0) {
    sevContainer.innerHTML = '⚠ Severity indicators: ' +
      nlp.severity_indicators.map(s => `<span class="sev-tag">${escapeHtml(s)}</span>`).join(' ');
  } else {
    sevContainer.innerHTML = '';
  }
  const hlEl = document.getElementById('nlp-highlighted');
  if (nlp.parsed_highlights) {
    hlEl.innerHTML = nlp.parsed_highlights;
    hlEl.style.display = '';
  } else {
    hlEl.style.display = 'none';
  }
}

function renderRisk(risk) {
  const level = risk.level || 'medium';
  const score = Math.min(100, Math.max(0, risk.score || 0));
  document.getElementById('risk-badge').innerHTML =
    `<span class="risk-badge ${level}">${level.toUpperCase()} RISK</span>`;
  const scoreEl = document.getElementById('score-comparison');
  if (risk.ml_score !== undefined && risk.ai_score !== undefined) {
    scoreEl.innerHTML = `ML model: <strong>${Math.round(risk.ml_score)}/100</strong><br>AI refined: <strong>${Math.round(risk.ai_score)}/100</strong>`;
  } else if (risk.ml_score !== undefined) {
    scoreEl.innerHTML = `ML score: <strong>${Math.round(risk.ml_score)}/100</strong>`;
  } else {
    scoreEl.innerHTML = `Score: <strong>${score}/100</strong>`;
  }
  const fill = document.getElementById('risk-fill');
  fill.className = `risk-fill ${level}`;
  setTimeout(() => { fill.style.width = score + '%'; }, 50);
  document.getElementById('primary-concern').textContent = risk.primary_concern || '';
  const stepsEl = document.getElementById('reasoning-steps');
  if (risk.reasoning_steps && risk.reasoning_steps.length > 0) {
    stepsEl.innerHTML =
      '<p style="font-size:12px;color:var(--color-text-3);font-weight:500;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">How the model reasoned:</p>' +
      risk.reasoning_steps.map((s, i) =>
        `<div class="reasoning-step"><span class="step-num">${i + 1}</span><span>${escapeHtml(s)}</span></div>`
      ).join('');
  } else {
    stepsEl.innerHTML = '';
  }
}

function renderExplanation(risk, data, clientElapsedMs) {
  document.getElementById('explanation-text').textContent = risk.explanation || '';
  const metaEl = document.getElementById('processing-meta');
  const chips = [];
  if (data.session_id)          chips.push(`Session: ${data.session_id}`);
  if (risk.confidence)          chips.push(`Confidence: ${risk.confidence}`);
  if (data.processing_time_ms)  chips.push(`Server: ${data.processing_time_ms}ms`);
  if (clientElapsedMs)          chips.push(`Total: ${clientElapsedMs}ms`);
  metaEl.innerHTML = chips.map(c => `<span class="meta-chip">${escapeHtml(c)}</span>`).join('');
}

function renderSuggestions(suggestions) {
  const el = document.getElementById('suggestions-list');
  if (!suggestions || suggestions.length === 0) {
    el.innerHTML = '<p style="color:var(--color-text-3);font-size:14px">No specific suggestions available.</p>';
    return;
  }
  el.innerHTML = suggestions.map(s =>
    `<div class="suggestion-item">
      <div class="suggest-icon-wrap ${escapeHtml(s.priority || 'general')}">${s.icon || '🩺'}</div>
      <div class="suggest-content">
        <div class="suggest-priority ${escapeHtml(s.priority || 'general')}">${escapeHtml(s.priority || 'general')}</div>
        <div class="suggest-title">${escapeHtml(s.title || '')}</div>
        <div class="suggest-detail">${escapeHtml(s.detail || '')}</div>
      </div>
    </div>`
  ).join('');
}

// =============================================
// ERROR STATE
// =============================================
function showError(message, detail) {
  document.getElementById('error-title').textContent = message || 'Something went wrong';
  document.getElementById('error-detail').textContent = detail || 'Please check the backend server and try again.';
  showSection('error');
  setActiveStep(1);
}

// =============================================
// UTILITY
// =============================================
function escapeHtml(str) {
  if (typeof str !== 'string') return String(str || '');
  return str
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}
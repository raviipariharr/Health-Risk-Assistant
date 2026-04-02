/**
 * STEP 3: js/app.js — The UI Layer
 * ===================================
 * This file ONLY touches the DOM (HTML elements).
 * It calls api.js for data, then renders it.
 *
 * Responsibilities:
 *  - State management (what step is the user on?)
 *  - Event handlers (button clicks, chip toggles)
 *  - Showing/hiding sections (input → loading → results)
 *  - Rendering API response data into HTML
 *  - Animating the step bar and pipeline loader
 *
 * Key DOM pattern used throughout:
 *   el.classList.add('hidden')    → hide an element
 *   el.classList.remove('hidden') → show an element
 *   el.textContent = '...'        → set text safely (no XSS risk)
 *   el.innerHTML = '...'          → set HTML (only use with trusted data)
 */


// =============================================
// SECTION REFERENCES
// Grab references once at the top — faster than
// calling getElementById() repeatedly in functions
// =============================================
const inputSection   = document.getElementById('input-section');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const errorSection   = document.getElementById('error-section');
const analyzeBtn     = document.getElementById('analyze-btn');


// =============================================
// ON PAGE LOAD
// Check if the backend is reachable and update
// the status dot in the header
// =============================================
document.addEventListener('DOMContentLoaded', async () => {
  // Character counter for textarea
  const textarea  = document.getElementById('symptom-text');
  const charCount = document.getElementById('char-count');
  textarea.addEventListener('input', () => {
    charCount.textContent = textarea.value.length;
  });

  // Check backend health
  await refreshHealthStatus();
});


/**
 * refreshHealthStatus()
 * ----------------------
 * Calls the /health endpoint and updates the status pill in the header.
 * Called on load and after errors.
 */
async function refreshHealthStatus() {
  const statusEl = document.getElementById('api-status');
  const textEl   = statusEl.querySelector('.status-text');

  statusEl.className = 'api-status';  // reset classes
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

/**
 * setActiveStep(n)
 * -----------------
 * Updates the visual step progress bar.
 * Steps < n are marked "done" (green check)
 * Step n is "active" (filled teal)
 * Steps > n are inactive (grey)
 */
function setActiveStep(n) {
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById(`step-${i}`);
    el.classList.remove('active', 'done');
    const circle = el.querySelector('.step-circle');

    if (i < n) {
      el.classList.add('done');
      circle.textContent = '✓';
    } else if (i === n) {
      el.classList.add('active');
      circle.textContent = i;
    } else {
      circle.textContent = i;
    }
  }
}


// =============================================
// PIPELINE LOADER ANIMATION
// Shows which backend step is running
// =============================================

const PIPELINE_MESSAGES = [
  { step: 'pl-nlp',  status: 'running', msg: 'Running NLP — extracting symptom keywords...' },
  { step: 'pl-ml',   status: 'running', msg: 'Running ML scorer — computing risk weights...' },
  { step: 'pl-ai',   status: 'running', msg: 'Claude AI reasoning — generating explanation...' },
];

let pipelineTimer = null;
let pipelineIndex = 0;

function startPipelineAnimation() {
  pipelineIndex = 0;
  resetPipeline();
  stepPipeline();
  // Advance every 2.5s to simulate pipeline stages
  pipelineTimer = setInterval(stepPipeline, 2500);
}

function stepPipeline() {
  if (pipelineIndex >= PIPELINE_MESSAGES.length) return;

  // Mark previous step as done
  if (pipelineIndex > 0) {
    const prevId = PIPELINE_MESSAGES[pipelineIndex - 1].step;
    const prevStatus = document.querySelector(`#${prevId} .pl-status`);
    if (prevStatus) {
      prevStatus.className = 'pl-status done';
      prevStatus.textContent = 'done ✓';
    }
  }

  const { step, status, msg } = PIPELINE_MESSAGES[pipelineIndex];
  const statusEl = document.querySelector(`#${step} .pl-status`);
  if (statusEl) {
    statusEl.className = `pl-status ${status}`;
    statusEl.textContent = 'running...';
  }

  document.getElementById('loading-msg').textContent = msg;
  pipelineIndex++;
}

function stopPipelineAnimation() {
  clearInterval(pipelineTimer);
  // Mark all steps done
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
// Only one section visible at a time
// =============================================

function showSection(name) {
  inputSection.classList.add('hidden');
  loadingSection.classList.add('hidden');
  resultsSection.classList.add('hidden');
  errorSection.classList.add('hidden');

  if (name === 'input')   inputSection.classList.remove('hidden');
  if (name === 'loading') loadingSection.classList.remove('hidden');
  if (name === 'results') resultsSection.classList.remove('hidden');
  if (name === 'error')   errorSection.classList.remove('hidden');
}


// =============================================
// MAIN HANDLER: Analyze button click
// =============================================

async function handleAnalyze() {
  // --- Validate form ---
  const payload = buildPayload();

  if (!payload.symptom_text && payload.selected_chips.length === 0) {
    // Show inline validation message
    document.getElementById('symptom-text').style.borderColor = 'var(--color-high)';
    document.getElementById('symptom-text').placeholder = 'Please describe your symptoms or select from the list above...';
    document.getElementById('symptom-text').focus();
    return;
  }

  // Reset textarea border
  document.getElementById('symptom-text').style.borderColor = '';

  // --- Switch to loading state ---
  analyzeBtn.disabled = true;
  showSection('loading');
  setActiveStep(2);
  startPipelineAnimation();

  // --- Call the API (api.js) ---
  const startTime = Date.now();
  const result = await analyzeSymptoms(payload, getBaseUrl());
  const elapsed = Date.now() - startTime;

  stopPipelineAnimation();

  // --- Handle result ---
  if (!result.ok) {
    showError(result.error, result.detail);
    analyzeBtn.disabled = false;
    return;
  }

  // --- Render results ---
  renderResults(result.data, elapsed);
  showSection('results');
  setActiveStep(4);
  analyzeBtn.disabled = false;

  // Scroll to results
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}


// =============================================
// RESET: Back to input form
// =============================================

function handleReset() {
  // Clear chips
  document.querySelectorAll('.chip.selected').forEach(c => c.classList.remove('selected'));
  // Clear text
  document.getElementById('symptom-text').value = '';
  document.getElementById('char-count').textContent = '0';
  // Reset step bar
  setActiveStep(1);
  showSection('input');
  window.scrollTo({ top: 0, behavior: 'smooth' });
  // Re-check backend status
  refreshHealthStatus();
}


// =============================================
// RENDER RESULTS
// Takes the AnalysisResponse from the API and
// populates each result card section by section
// =============================================

function renderResults(data, clientElapsedMs) {
  renderNLP(data.nlp);
  renderRisk(data.risk);
  renderExplanation(data.risk, data, clientElapsedMs);
  renderSuggestions(data.suggestions);
}


/**
 * renderNLP(nlp)
 * ---------------
 * Populates the NLP Analysis card (Step 2 result)
 * nlp = { detected_keywords, symptom_summary, severity_indicators,
 *          parsed_highlights, duration_context }
 */
function renderNLP(nlp) {
  // Keywords as coloured tags
  const kwContainer = document.getElementById('nlp-keywords');
  if (nlp.detected_keywords && nlp.detected_keywords.length > 0) {
    kwContainer.innerHTML = nlp.detected_keywords
      .map(kw => `<span class="kw-tag">${escapeHtml(kw)}</span>`)
      .join('');
  } else {
    kwContainer.innerHTML = '<span class="kw-tag" style="background:var(--color-bg);color:var(--color-text-3)">No specific keywords detected</span>';
  }

  // Summary sentence
  document.getElementById('nlp-summary').textContent = nlp.symptom_summary || '';

  // Severity indicators (if any)
  const sevContainer = document.getElementById('nlp-severity');
  if (nlp.severity_indicators && nlp.severity_indicators.length > 0) {
    sevContainer.innerHTML = '⚠ Severity indicators: ' +
      nlp.severity_indicators.map(s => `<span class="sev-tag">${escapeHtml(s)}</span>`).join(' ');
  } else {
    sevContainer.innerHTML = '';
  }

  // Highlighted original text
  // parsed_highlights comes from the backend with <span class="kw"> already in it
  // We trust this HTML because it came from our own backend, not user input
  const hlEl = document.getElementById('nlp-highlighted');
  if (nlp.parsed_highlights) {
    hlEl.innerHTML = nlp.parsed_highlights;
    hlEl.style.display = '';
  } else {
    hlEl.style.display = 'none';
  }
}


/**
 * renderRisk(risk)
 * -----------------
 * Populates the Risk Prediction card (Step 3 result)
 * risk = { level, score, primary_concern, reasoning_steps,
 *           ml_score, ai_score, confidence }
 */
function renderRisk(risk) {
  const level = risk.level || 'medium';
  const score = Math.min(100, Math.max(0, risk.score || 0));

  // Risk badge (Low / Medium / High)
  document.getElementById('risk-badge').innerHTML =
    `<span class="risk-badge ${level}">${level.toUpperCase()} RISK</span>`;

  // ML vs AI score comparison
  const scoreEl = document.getElementById('score-comparison');
  if (risk.ml_score !== undefined && risk.ai_score !== undefined) {
    scoreEl.innerHTML =
      `ML model: <strong>${Math.round(risk.ml_score)}/100</strong><br>` +
      `AI refined: <strong>${Math.round(risk.ai_score)}/100</strong>`;
  } else if (risk.ml_score !== undefined) {
    scoreEl.innerHTML = `ML score: <strong>${Math.round(risk.ml_score)}/100</strong>`;
  } else {
    scoreEl.innerHTML = `Score: <strong>${score}/100</strong>`;
  }

  // Animated risk meter bar
  const fill = document.getElementById('risk-fill');
  fill.className = `risk-fill ${level}`;
  // Small delay so CSS transition fires after element is visible
  setTimeout(() => { fill.style.width = score + '%'; }, 50);

  // Primary concern callout
  document.getElementById('primary-concern').textContent = risk.primary_concern || '';

  // Reasoning steps (explainability)
  const stepsEl = document.getElementById('reasoning-steps');
  if (risk.reasoning_steps && risk.reasoning_steps.length > 0) {
    stepsEl.innerHTML = '<p style="font-size:12px;color:var(--color-text-3);font-weight:500;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">How the model reasoned:</p>' +
      risk.reasoning_steps.map((s, i) =>
        `<div class="reasoning-step">
          <span class="step-num">${i + 1}</span>
          <span>${escapeHtml(s)}</span>
        </div>`
      ).join('');
  } else {
    stepsEl.innerHTML = '';
  }
}


/**
 * renderExplanation(risk, data, elapsed)
 * ----------------------------------------
 * Populates the AI Explanation card (Step 4)
 */
function renderExplanation(risk, data, clientElapsedMs) {
  document.getElementById('explanation-text').textContent = risk.explanation || '';

  // Processing metadata chips
  const metaEl = document.getElementById('processing-meta');
  const chips = [];

  if (data.session_id)  chips.push(`Session: ${data.session_id}`);
  if (risk.confidence)  chips.push(`Confidence: ${risk.confidence}`);
  if (data.processing_time_ms) chips.push(`Server: ${data.processing_time_ms}ms`);
  if (clientElapsedMs)  chips.push(`Total: ${clientElapsedMs}ms`);

  metaEl.innerHTML = chips
    .map(c => `<span class="meta-chip">${escapeHtml(c)}</span>`)
    .join('');
}


/**
 * renderSuggestions(suggestions)
 * --------------------------------
 * Renders the ordered list of recommendations
 * suggestions = [{ priority, icon, title, detail }, ...]
 */
function renderSuggestions(suggestions) {
  const el = document.getElementById('suggestions-list');

  if (!suggestions || suggestions.length === 0) {
    el.innerHTML = '<p style="color:var(--color-text-3);font-size:14px">No specific suggestions available.</p>';
    return;
  }

  el.innerHTML = suggestions.map(s =>
    `<div class="suggestion-item">
      <div class="suggest-icon-wrap ${escapeHtml(s.priority || 'general')}">
        ${s.icon || '🩺'}
      </div>
      <div class="suggest-content">
        <div class="suggest-priority ${escapeHtml(s.priority || 'general')}">
          ${escapeHtml(s.priority || 'general')}
        </div>
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
  document.getElementById('error-detail').textContent =
    detail || 'Please check the backend server and try again.';
  showSection('error');
  setActiveStep(1);
}


// =============================================
// UTILITY
// =============================================

/**
 * escapeHtml(str)
 * ----------------
 * Converts < > & " ' to safe HTML entities.
 * ALWAYS use this when inserting user-provided text into innerHTML.
 * Prevents XSS (Cross-Site Scripting) attacks.
 *
 * Example:
 *   escapeHtml('<script>alert(1)</script>')
 *   → '&lt;script&gt;alert(1)&lt;/script&gt;'
 */
function escapeHtml(str) {
  if (typeof str !== 'string') return String(str || '');
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
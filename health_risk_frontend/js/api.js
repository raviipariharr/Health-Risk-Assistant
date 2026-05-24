/**
 * js/api.js — Data Layer (Extended with Vitals & Labs)
 * ======================================================
 * buildPayload() now reads vitals/lab numeric inputs and
 * includes them in the POST body as structured objects.
 */

async function checkHealth(baseUrl) {
  try {
    const response = await fetch(`${baseUrl}/api/v1/health`, { method: 'GET' });
    if (!response.ok) return { ok: false, error: `Server responded with ${response.status}` };
    const data = await response.json();
    return { ok: true, data };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

async function analyzeSymptoms(payload, baseUrl) {
  try {
    const response = await fetch(`${baseUrl}/api/v1/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let errorDetail = `HTTP ${response.status}`;
      try {
        const errData = await response.json();
        errorDetail = errData.detail || errData.error || errorDetail;
        if (Array.isArray(errorDetail)) {
          errorDetail = errorDetail.map(e => `${e.loc?.join('.')}: ${e.msg}`).join(', ');
        }
      } catch (_) {}
      return { ok: false, status: response.status, error: errorDetail };
    }

    const data = await response.json();
    return { ok: true, data };
  } catch (err) {
    return { ok: false, status: 0, error: 'Could not reach the backend. Is the server running?', detail: err.message };
  }
}

/**
 * parseOptionalInt / parseOptionalFloat
 * Read an input field and return the numeric value or null.
 */
function parseOptionalInt(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  const v = el.value.trim();
  if (!v) return null;
  const n = parseInt(v, 10);
  return isNaN(n) ? null : n;
}

function parseOptionalFloat(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  const v = el.value.trim();
  if (!v) return null;
  const n = parseFloat(v);
  return isNaN(n) ? null : n;
}

/**
 * buildVitals() — reads the vitals panel inputs.
 * Returns null if all fields are empty (no vitals panel active).
 */
function buildVitals() {
  const vitals = {
    systolic_bp:         parseOptionalInt('v-systolic'),
    diastolic_bp:        parseOptionalInt('v-diastolic'),
    heart_rate:          parseOptionalInt('v-heart-rate'),
    spo2:                parseOptionalFloat('v-spo2'),
    respiratory_rate:    parseOptionalInt('v-rr'),
    temperature_celsius: parseOptionalFloat('v-temp'),
    weight_kg:           parseOptionalFloat('v-weight'),
    height_cm:           parseOptionalFloat('v-height'),
  };
  // Only include if at least one value is entered
  const hasAny = Object.values(vitals).some(v => v !== null);
  return hasAny ? vitals : null;
}

/**
 * buildLabs() — reads the labs panel inputs.
 */
function buildLabs() {
  const labs = {
    total_cholesterol: parseOptionalFloat('l-total-chol'),
    ldl_cholesterol:   parseOptionalFloat('l-ldl'),
    hdl_cholesterol:   parseOptionalFloat('l-hdl'),
    triglycerides:     parseOptionalFloat('l-trig'),
    fasting_glucose:   parseOptionalFloat('l-glucose'),
    hba1c:             parseOptionalFloat('l-hba1c'),
    creatinine:        parseOptionalFloat('l-creatinine'),
    egfr:              parseOptionalFloat('l-egfr'),
  };
  const hasAny = Object.values(labs).some(v => v !== null);
  return hasAny ? labs : null;
}

function buildPayload() {
  const chips = [...document.querySelectorAll('.chip.selected')].map(el => el.dataset.symptom);
  const text  = document.getElementById('symptom-text').value.trim();
  const ageRaw = document.getElementById('age').value;
  const age = ageRaw ? parseInt(ageRaw, 10) : null;

  // Get active focus area
  const focusEl = document.querySelector('.focus-btn.active');
  const focusArea = focusEl ? focusEl.dataset.focus : null;

  return {
    symptom_text:   text,
    selected_chips: chips,
    age:            age,
    sex:            document.getElementById('sex').value,
    duration:       document.getElementById('duration').value,
    vitals:         buildVitals(),
    labs:           buildLabs(),
    focus_area:     focusArea,
  };
}

function getBaseUrl() {
  const input = document.getElementById('api-base-url');
  return (input?.value || 'http://localhost:8000').replace(/\/$/, '');
}
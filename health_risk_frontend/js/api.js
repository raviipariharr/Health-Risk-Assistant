/**
 * STEP 3: js/api.js — The Data Layer
 * =====================================
 * This file ONLY talks to the backend.
 * It knows nothing about the DOM (no getElementById here).
 *
 * Why separate api.js from app.js?
 * → Single Responsibility Principle.
 *   api.js  = "get me data from the server"
 *   app.js  = "show that data to the user"
 *
 * If you swap the backend URL, change the auth method,
 * or add retries — you only touch THIS file.
 * The UI code stays untouched.
 *
 * Pattern used: async/await with fetch()
 * → fetch() is the browser's built-in HTTP client
 * → async/await makes async code read like sync code
 * → try/catch handles network errors cleanly
 */


/**
 * checkHealth()
 * ---------------
 * Calls GET /api/v1/health to verify the backend is reachable.
 * Returns { ok: true } or { ok: false, error: "..." }
 *
 * Called on page load so the status dot in the header updates.
 */
async function checkHealth(baseUrl) {
  try {
    // fetch() returns a Promise — await pauses until it resolves
    const response = await fetch(`${baseUrl}/api/v1/health`, {
      method: 'GET',
      // signal: AbortSignal.timeout(3000)  ← timeout after 3s (modern browsers)
    });

    if (!response.ok) {
      return { ok: false, error: `Server responded with ${response.status}` };
    }

    const data = await response.json();
    // data = { status: "healthy", timestamp: "...", version: "..." }
    return { ok: true, data };

  } catch (err) {
    // fetch() throws if the server is unreachable (connection refused, DNS fail, etc.)
    return { ok: false, error: err.message };
  }
}


/**
 * analyzeSymptoms(payload, baseUrl)
 * -----------------------------------
 * The main API call. Sends symptom data to POST /api/v1/analyze.
 *
 * payload shape (must match SymptomRequest in models.py):
 * {
 *   symptom_text:    string,
 *   selected_chips:  string[],
 *   age:             number | null,
 *   sex:             "male" | "female" | "other",
 *   duration:        "hours" | "days" | "week" | "weeks" | "months"
 * }
 *
 * Returns { ok: true, data: AnalysisResponse }
 *      or { ok: false, status: 400/500, error: "..." }
 */
async function analyzeSymptoms(payload, baseUrl) {
  try {
    const response = await fetch(`${baseUrl}/api/v1/analyze`, {
      method: 'POST',

      // Headers tell the server what format we're sending
      headers: {
        'Content-Type': 'application/json',
        // Uncomment if you add auth to the backend:
        // 'Authorization': `Bearer ${token}`,
      },

      // JSON.stringify() converts the JS object to a JSON string
      // The server's Pydantic model parses it back into Python
      body: JSON.stringify(payload),
    });

    // response.ok is true for 2xx status codes (200, 201, etc.)
    if (!response.ok) {
      let errorDetail = `HTTP ${response.status}`;
      try {
        // FastAPI returns validation errors as JSON — try to read them
        const errData = await response.json();
        errorDetail = errData.detail || errData.error || errorDetail;
        // Pydantic 422 errors come as an array of field errors
        if (Array.isArray(errorDetail)) {
          errorDetail = errorDetail.map(e => `${e.loc?.join('.')}: ${e.msg}`).join(', ');
        }
      } catch (_) { /* ignore parse error */ }

      return { ok: false, status: response.status, error: errorDetail };
    }

    // Parse the JSON response body
    // This becomes our AnalysisResponse object from models.py
    const data = await response.json();
    return { ok: true, data };

  } catch (err) {
    // Network error (backend not running, CORS blocked, etc.)
    return {
      ok: false,
      status: 0,
      error: 'Could not reach the backend. Is the server running?',
      detail: err.message
    };
  }
}


/**
 * buildPayload()
 * ---------------
 * Reads the current form state and builds the API payload object.
 * Keeps form-reading logic here so app.js doesn't need to know
 * about field IDs or data coercion.
 */
function buildPayload() {
  // Get selected chip labels
  const chips = [...document.querySelectorAll('.chip.selected')]
    .map(el => el.dataset.symptom);

  // Get free text
  const text = document.getElementById('symptom-text').value.trim();

  // Get age (convert string → number, or null if empty)
  const ageRaw = document.getElementById('age').value;
  const age = ageRaw ? parseInt(ageRaw, 10) : null;

  return {
    symptom_text:   text,
    selected_chips: chips,
    age:            age,
    sex:            document.getElementById('sex').value,
    duration:       document.getElementById('duration').value,
  };
}


/**
 * getBaseUrl()
 * -------------
 * Reads the backend URL from the settings input.
 * Strips trailing slash for consistency.
 */
function getBaseUrl() {
  const input = document.getElementById('api-base-url');
  return (input?.value || 'http://localhost:8000').replace(/\/$/, '');
}
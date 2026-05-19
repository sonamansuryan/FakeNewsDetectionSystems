// popup.js — TruthLens
// Reads pipeline result from chrome.storage.local and renders the UI.

// ── Helpers ──────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

function showState(name) {
  document.querySelectorAll(".state").forEach((s) => s.classList.remove("active"));
  const el = $(`state-${name}`);
  if (el) el.classList.add("active");
}

/**
 * Map API final_label to display data.
 */
function verdictMeta(label = "") {
  const l = label.toUpperCase();
  const map = {
    SUPPORTED:           { cls: "verdict-verified",    icon: "✅", text: "VERIFIED",    sub: "Supported by credible evidence" },
    PARTIALLY_SUPPORTED: { cls: "verdict-partial",     icon: "🟡", text: "PARTIAL",     sub: "Partially supported — some details unverified" },
    UNSUPPORTED:         { cls: "verdict-unsupported", icon: "❓", text: "UNVERIFIABLE", sub: "No strong evidence to confirm or deny" },
    MISLEADING_FRAMING:  { cls: "verdict-misleading",  icon: "⚠️", text: "MISLEADING",  sub: "Framing creates a false impression" },
    CONTRADICTED:        { cls: "verdict-fake",        icon: "❌", text: "FALSE / REFUTED", sub: "Actively refuted by available evidence" },
    FALSE_FABRICATED:    { cls: "verdict-fabricated",  icon: "🚫", text: "FABRICATED",  sub: "Documented reality is the direct opposite" },
  };
  return map[l] || { cls: "verdict-unsupported", icon: "❓", text: label || "UNKNOWN", sub: "" };
}

/**
 * Map manipulation_risk string to display config.
 */
function riskMeta(risk = "") {
  const r = risk.toUpperCase();
  if (r === "HIGH")   return { cls: "risk-high",   pct: 90, color: "var(--risk-high)" };
  if (r === "MEDIUM") return { cls: "risk-medium", pct: 55, color: "var(--risk-medium)" };
  return                     { cls: "risk-low",    pct: 18, color: "var(--risk-low)" };
}

/**
 * Extract a clean label + URL from a source entry.
 * Sources may be strings (URLs) or objects {url, title}.
 */
function parseSource(src) {
  if (typeof src === "string") {
    try {
      const u = new URL(src);
      return { url: src, domain: u.hostname, label: src };
    } catch {
      return { url: "#", domain: src, label: src };
    }
  }
  if (src && typeof src === "object") {
    const url = src.url || src.href || "#";
    let domain = "";
    try { domain = new URL(url).hostname; } catch { domain = url; }
    const label = src.title || src.label || domain || url;
    return { url, domain, label };
  }
  return { url: "#", domain: String(src), label: String(src) };
}

// ── Loading step animation ────────────────────────────────────────────────────

let _stepTimer = null;

function startLoadingSteps() {
  const steps = ["step-roberta", "step-rag", "step-mistral"];
  let i = 0;

  steps.forEach((id) => $(`${id}`)?.classList.remove("active"));
  $("step-roberta")?.classList.add("active");

  _stepTimer = setInterval(() => {
    steps.forEach((id) => $(id)?.classList.remove("active"));
    i = (i + 1) % steps.length;
    $(steps[i])?.classList.add("active");
  }, 1800);
}

function stopLoadingSteps() {
  clearInterval(_stepTimer);
  ["step-roberta", "step-rag", "step-mistral"].forEach((id) =>
    $(id)?.classList.remove("active")
  );
}

// ── Render result ─────────────────────────────────────────────────────────────

function renderResult(data) {
  stopLoadingSteps();

  // Claim
  $("result-claim").textContent = data.claim || "";

  // Verdict
  const label    = data.final_verdict?.label || data.final_verdict?.final_label || "";
  const meta     = verdictMeta(label);
  const banner   = $("verdict-banner");
  const glow     = $("verdict-glow");

  // Remove old verdict classes
  banner.className = `verdict-banner ${meta.cls}`;
  glow.style.background = `radial-gradient(ellipse at center, var(--v, var(--accent)) 0%, transparent 70%)`;

  $("verdict-icon").textContent  = meta.icon;
  $("verdict-label").textContent = meta.text;
  $("verdict-sub").textContent   = meta.sub;

  // RoBERTa signal
  const roLabel = data.roberta?.label || "";
  const roConf  = Math.round((data.roberta?.confidence || 0) * 100);
  const roEl    = $("roberta-label");
  roEl.textContent  = roLabel;
  roEl.style.color  = roLabel.toUpperCase() === "REAL" ? "var(--c-verified)" : "var(--c-fake)";
  $("roberta-pct").textContent  = `${roConf}%`;
  const roBar = $("roberta-bar");
  roBar.style.width      = `${roConf}%`;
  roBar.style.background = roLabel.toUpperCase() === "REAL" ? "var(--c-verified)" : "var(--c-fake)";

  // Verdict confidence (RoBERTa as proxy)
  $("verdict-conf").textContent = `RoBERTa ${roConf}%`;

  // Manipulation risk
  const manip   = data.manipulation_analysis || {};
  const risk    = manip.manipulation_risk || "LOW";
  const rMeta   = riskMeta(risk);
  const rBadge  = $("risk-badge");
  rBadge.textContent   = risk;
  rBadge.style.color   = rMeta.color;
  rBadge.style.borderColor = rMeta.color;

  $("risk-reason").textContent = manip.risk_reason || manip.manipulation_reason || "";

  const rFill = $("risk-fill");
  rFill.style.width      = `${rMeta.pct}%`;
  rFill.style.background = rMeta.color;

  // AI Reasoning
  $("ai-reasoning").textContent =
    data.final_verdict?.explanation || "No explanation provided.";

  // Counter-narrative
  $("counter-narrative").textContent =
    data.counter_narrative || "No counter-narrative generated.";

  // Sources
  const sources    = data.sources || [];
  const sourcesList = $("sources-list");
  const sourcesBlock = $("sources-block");

  sourcesList.innerHTML = "";

  if (sources.length === 0) {
    sourcesBlock.style.display = "none";
  } else {
    sourcesBlock.style.display = "block";
    sources.forEach((s) => {
      const { url, domain, label: sLabel } = parseSource(s);
      const li = document.createElement("li");
      li.innerHTML = `
        <a href="${url}" target="_blank" rel="noopener" title="${url}">
          <span class="source-domain">[${domain}]</span>
          <span>${sLabel !== url ? sLabel : domain}</span>
        </a>`;
      sourcesList.appendChild(li);
    });
  }

  showState("result");
}

// ── Event handlers ────────────────────────────────────────────────────────────

$("btn-new")?.addEventListener("click", () => {
  chrome.storage.local.set({ status: null, result: null, error: null, claim: null });
  showState("idle");
});

$("btn-copy")?.addEventListener("click", async () => {
  const { result } = await chrome.storage.local.get("result");
  if (!result) return;

  const label = result.final_verdict?.label || "";
  const meta  = verdictMeta(label);
  const manip = result.manipulation_analysis || {};

  const report = [
    `FACT-CHECK REPORT`,
    `─────────────────`,
    `Claim: ${result.claim}`,
    ``,
    `VERDICT: ${meta.text}`,
    `${meta.sub}`,
    ``,
    `Manipulation Risk: ${manip.manipulation_risk || "N/A"}`,
    `${manip.risk_reason || ""}`,
    ``,
    `AI Reasoning:`,
    result.final_verdict?.explanation || "",
    ``,
    `Counter-Narrative:`,
    result.counter_narrative || "",
    ``,
    `Sources:`,
    ...(result.sources || []).map((s) => {
      const { url } = parseSource(s);
      return `• ${url}`;
    }),
    ``,
    `Analysed by TruthLens (Triple-Signal AI)`,
  ].join("\n");

  try {
    await navigator.clipboard.writeText(report);
    const btn = $("btn-copy");
    btn.textContent = "Copied ✓";
    setTimeout(() => (btn.textContent = "Copy report"), 2000);
  } catch {
    // clipboard unavailable in extension context; silent fail
  }
});

$("btn-retry")?.addEventListener("click", async () => {
  const { claim } = await chrome.storage.local.get("claim");
  if (claim) {
    await chrome.storage.local.set({ status: "loading", result: null, error: null });
    showState("loading");
    startLoadingSteps();
    $("loading-claim").textContent = claim.length > 60 ? claim.slice(0, 60) + "…" : claim;
  }
});

// ── Listen for background → popup messages ────────────────────────────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === "RESULT_READY") {
    loadFromStorage();
  }
});

// ── Boot: read current state from storage ─────────────────────────────────────

async function loadFromStorage() {
  const { status, result, error, claim } =
    await chrome.storage.local.get(["status", "result", "error", "claim"]);

  if (status === "loading") {
    showState("loading");
    startLoadingSteps();
    if (claim) {
      $("loading-claim").textContent =
        claim.length > 60 ? claim.slice(0, 60) + "…" : claim;
    }
    return;
  }

  if (status === "error") {
    $("error-text").textContent = error || "Unknown error.";
    showState("error");
    return;
  }

  if (status === "done" && result) {
    renderResult(result);
    return;
  }

  // Default: idle
  showState("idle");
}

// Kick off
loadFromStorage();
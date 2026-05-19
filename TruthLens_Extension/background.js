// background.js — TruthLens Service Worker
// Handles context menu creation and API communication

const API_URL = "http://localhost:5000/verify";

// ── Create context menu on install ──────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "truthlens-verify",
    title: "🔍 Verify with TruthLens AI",
    contexts: ["selection"],
  });
});

// ── Context menu click handler ───────────────────────────────────────────────
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "truthlens-verify") return;

  const selectedText = (info.selectionText || "").trim();
  if (!selectedText) return;

  // Save the claim text and set loading state
  await chrome.storage.local.set({
    claim:  selectedText,
    status: "loading",
    result: null,
    error:  null,
  });

  // Open the popup
  await chrome.action.openPopup().catch(() => {
    // openPopup() may fail in some contexts; fallback: open as window
    chrome.windows.create({
      url:    chrome.runtime.getURL("popup.html"),
      type:   "popup",
      width:  520,
      height: 720,
    });
  });

  // Call the Flask API
  try {
    const response = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ claim: selectedText }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    await chrome.storage.local.set({
      status: "done",
      result: data,
      error:  null,
    });
  } catch (err) {
    await chrome.storage.local.set({
      status: "error",
      result: null,
      error:  err.message || "Could not reach the API. Is the server running?",
    });
  }

  // Notify popup that data is ready
  chrome.runtime.sendMessage({ type: "RESULT_READY" }).catch(() => {
    // popup might not be open yet; it will poll storage on load
  });
});
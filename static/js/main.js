/* ═══════════════════════════════════════════════════════════════════════════
   RAG Chat — Client-Side Logic
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  activeDocument: null,
  engineReady: false,
  typing: false,
  lastSources: [],
  statusInterval: null,
  processingProgress: 0,
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const enginePill    = $("engine-status-pill");
const engineText    = $("engine-status-text");
const uploadZone    = $("upload-zone");
const fileInput     = $("file-input");
const browseBtn     = $("browse-btn");
const procBar       = $("processing-bar");
const procText      = $("processing-text");
const progFill      = $("progress-fill");
const docList       = $("doc-list");
const chatMessages  = $("chat-messages");
const chatInput     = $("chat-input");
const sendBtn       = $("send-btn");
const docNameDisp   = $("doc-name-display");
const sourcesPanel  = $("sources-panel");
const sourcesToggle = $("sources-toggle");
const closeSources  = $("close-sources");
const sourcesContent= $("sources-content");
const welcomeScreen = $("welcome-screen");
const toastContainer= $("toast-container");

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(message, type = "info", duration = 4000) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<div class="toast-dot"></div><span>${message}</span>`;
  toastContainer.appendChild(el);
  setTimeout(() => {
    el.classList.add("removing");
    setTimeout(() => el.remove(), 300);
  }, duration);
}

// ── Engine status polling ─────────────────────────────────────────────────────
function setEnginePill(state) {
  enginePill.className = `status-pill status-${state}`;
}

let processingWasActive = false;

async function pollStatus() {
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();
    const eng  = data.engine;
    const proc = data.processing;

    // Engine pill
    engineText.textContent = eng.state === "ready"  ? "Engine Ready"
                           : eng.state === "error"  ? "Engine Error"
                           : "Initializing…";
    setEnginePill(eng.state === "ready" ? "ready" : eng.state === "error" ? "error" : "loading");

    state.engineReady = eng.state === "ready";
    updateInputState();

    // Processing bar
    const active = ["queued","processing","embedding"].includes(proc.state);
    if (active) {
      procBar.classList.remove("hidden");
      procText.textContent = proc.message;
      // Fake incremental progress
      state.processingProgress = Math.min(state.processingProgress + 4, 88);
      progFill.style.width = state.processingProgress + "%";
      processingWasActive = true;
    } else if (proc.state === "done" && processingWasActive) {
      procBar.classList.remove("hidden");
      procText.textContent = proc.message;
      progFill.style.width = "100%";
      processingWasActive = false;
      state.processingProgress = 0;
      showToast(proc.message, "success");
      await refreshDocumentList();
      setTimeout(() => procBar.classList.add("hidden"), 2500);
    } else if (proc.state === "error" && processingWasActive) {
      processingWasActive = false;
      state.processingProgress = 0;
      showToast("Processing failed: " + proc.message, "error", 6000);
      procBar.classList.add("hidden");
    }
  } catch (e) {
    // Network error — keep polling silently
  }
}

// ── Document list ─────────────────────────────────────────────────────────────
async function refreshDocumentList() {
  try {
    const res  = await fetch("/api/documents");
    const data = await res.json();
    const docs = data.documents || [];

    docList.innerHTML = "";
    if (docs.length === 0) {
      docList.innerHTML = `<li class="doc-empty">No documents yet — upload a PDF above.</li>`;
      return;
    }

    docs.forEach(name => {
      const li = document.createElement("li");
      li.className = `doc-item${name === state.activeDocument ? " active" : ""}`;
      li.dataset.name = name;
      li.innerHTML = `
        <span class="doc-item-icon">
          <svg viewBox="0 0 24 24" fill="none"><path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V8L14 2Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M14 2V8H20" stroke="currentColor" stroke-width="2"/></svg>
        </span>
        <span class="doc-item-name" title="${name}">${name}</span>`;
      li.addEventListener("click", () => selectDocument(name));
      docList.appendChild(li);
    });
  } catch (e) {
    console.error("Failed to refresh document list", e);
  }
}

function selectDocument(name) {
  if (state.activeDocument === name) return;
  state.activeDocument = name;

  // Update sidebar
  document.querySelectorAll(".doc-item").forEach(el => {
    el.classList.toggle("active", el.dataset.name === name);
  });

  // Update header
  docNameDisp.textContent = name;

  // Clear chat
  chatMessages.innerHTML = "";

  // Enable input if engine ready
  updateInputState();

  // Clear sources
  renderSources([]);
}

// ── Input gating ─────────────────────────────────────────────────────────────
function updateInputState() {
  const enabled = state.engineReady && !!state.activeDocument && !state.typing;
  chatInput.disabled = !enabled;
  sendBtn.disabled   = !enabled;
  chatInput.placeholder = !state.engineReady
    ? "Waiting for RAG engine…"
    : !state.activeDocument
    ? "Select a document from the sidebar first…"
    : "Ask a question about the selected document…";
}

// ── Auto-resize textarea ─────────────────────────────────────────────────────
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + "px";
});

// ── Send message ─────────────────────────────────────────────────────────────
chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    doSend();
  }
});
sendBtn.addEventListener("click", doSend);

async function doSend() {
  const query = chatInput.value.trim();
  if (!query || !state.activeDocument || !state.engineReady || state.typing) return;

  // Append user bubble
  appendMessage("user", query);
  chatInput.value = "";
  chatInput.style.height = "auto";

  // Show typing indicator
  const typingEl = appendTypingIndicator();
  state.typing = true;
  updateInputState();

  let fetchRes, fetchData;
  try {
    fetchRes = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, document: state.activeDocument }),
    });
    fetchData = await fetchRes.json();
  } catch (e) {
    typingEl.remove();
    const msg = `Network error — could not reach the server. (${e.message})`;
    appendMessage("assistant", `⚠️ ${msg}`);
    showToast(msg, "error", 6000);
    state.typing = false;
    updateInputState();
    return;
  }

  // Network succeeded — handle response outside the try/catch
  typingEl.remove();
  if (!fetchRes.ok) {
    const msg = fetchData.error || `Server error (${fetchRes.status})`;
    appendMessage("assistant", `⚠️ ${msg}`);
    showToast(msg, "error", 6000);
  } else {
    state.lastSources = fetchData.sources || [];
    appendMessage("assistant", fetchData.answer || "(no answer returned)", state.lastSources.length);
    renderSources(state.lastSources);
  }

  state.typing = false;
  updateInputState();
  scrollToBottom();
}


// ── Chat bubble helpers ───────────────────────────────────────────────────────
function appendMessage(role, content, sourceCount = 0) {
  // Remove welcome screen on first message
  if (welcomeScreen) welcomeScreen.remove();

  const div = document.createElement("div");
  div.className = `message ${role}`;

  const initials = role === "user" ? "You" : "AI";
  const safeContent = (content == null || content === "") ? "(empty response)" : String(content);
  const md = role === "assistant" ? marked.parse(safeContent) : escapeHtml(safeContent);

  let sourceBtn = "";
  if (role === "assistant" && sourceCount > 0) {
    sourceBtn = `<br/><button class="source-toggle-btn" onclick="toggleSourcesPanel()">📎 ${sourceCount} source${sourceCount > 1 ? "s" : ""}</button>`;
  }

  div.innerHTML = `
    <div class="avatar">${initials}</div>
    <div class="bubble">${md}${sourceBtn}</div>`;

  chatMessages.appendChild(div);
  scrollToBottom();
  return div;
}

function appendTypingIndicator() {
  const div = document.createElement("div");
  div.className = "message assistant";
  div.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble">
      <div class="typing-indicator"><span></span><span></span><span></span></div>
    </div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
  return div;
}

function escapeHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── Sources panel ─────────────────────────────────────────────────────────────
function renderSources(sources) {
  if (!sources || sources.length === 0) {
    sourcesContent.innerHTML = `<p class="sources-empty">Sources will appear here after each answer.</p>`;
    return;
  }
  sourcesContent.innerHTML = sources.map((s, i) => `
    <div class="source-card">
      <div class="source-page-badge">📄 Page ${s.page}</div>
      <p class="source-text" id="src-text-${i}">${escapeHtml(s.content)}</p>
      <button class="source-expand-btn" onclick="expandSource(${i})">Show more</button>
    </div>`).join("");
}

window.expandSource = function(i) {
  const el = $(`src-text-${i}`);
  const btn = el.nextElementSibling;
  if (el.style.webkitLineClamp === "unset") {
    el.style.webkitLineClamp = "6";
    el.style.display = "-webkit-box";
    btn.textContent = "Show more";
  } else {
    el.style.webkitLineClamp = "unset";
    el.style.display = "block";
    btn.textContent = "Show less";
  }
};

window.toggleSourcesPanel = function() {
  sourcesPanel.classList.toggle("hidden");
  sourcesToggle.setAttribute("aria-pressed",
    !sourcesPanel.classList.contains("hidden"));
};

sourcesToggle.addEventListener("click", toggleSourcesPanel);
closeSources.addEventListener("click", () => sourcesPanel.classList.add("hidden"));

// ── File upload ───────────────────────────────────────────────────────────────
browseBtn.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("click", e => {
  if (e.target === browseBtn || e.target === uploadZone || e.target.closest(".upload-icon")) {
    fileInput.click();
  }
});

uploadZone.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  handleFiles(Array.from(e.dataTransfer.files));
});
fileInput.addEventListener("change", () => {
  handleFiles(Array.from(fileInput.files));
  fileInput.value = "";
});

async function handleFiles(files) {
  const pdfs = files.filter(f => f.name.toLowerCase().endsWith(".pdf"));
  if (pdfs.length === 0) {
    showToast("Please upload PDF files only.", "warning");
    return;
  }

  const form = new FormData();
  pdfs.forEach(f => form.append("files", f));

  procBar.classList.remove("hidden");
  procText.textContent = `Uploading ${pdfs.length} file(s)…`;
  progFill.style.width = "8%";
  processingWasActive = true;
  state.processingProgress = 8;

  try {
    const res  = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      showToast(data.error || "Upload failed.", "error");
      procBar.classList.add("hidden");
      processingWasActive = false;
    } else {
      showToast(`Uploaded: ${pdfs.map(f=>f.name).join(", ")}`, "success");
    }
  } catch (e) {
    showToast("Upload network error.", "error");
    procBar.classList.add("hidden");
    processingWasActive = false;
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
(async function boot() {
  await pollStatus();
  await refreshDocumentList();
  state.statusInterval = setInterval(pollStatus, 2000);
})();

// VoxScribe front end. The browser does the heavy work so the server can stay tiny:
// decode audio -> 16 kHz mono -> ~2-minute WAV chunks -> POST /api/transcribe one by one.
// Transcripts and media live in this browser (IndexedDB). Nothing is stored on the server.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = sec => { sec = Math.max(0, Math.round(sec || 0)); const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), s = sec % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(s).padStart(2, "0"); };
const ago = t => { const d = (Date.now() / 1000 - t); if (d < 60) return "just now"; if (d < 3600) return Math.floor(d / 60) + " min ago";
  if (d < 86400) return Math.floor(d / 3600) + " h ago"; return new Date(t * 1000).toLocaleDateString(); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
function toast(msg, ms = 3200) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), ms); }

const LANGS = [["", "Auto-detect"], ["en", "English"], ["hi", "Hindi"], ["ur", "Urdu"], ["bn", "Bengali"], ["mr", "Marathi"], ["gu", "Gujarati"],
  ["pa", "Punjabi"], ["ta", "Tamil"], ["te", "Telugu"], ["kn", "Kannada"], ["ml", "Malayalam"], ["es", "Spanish"], ["fr", "French"],
  ["de", "German"], ["ar", "Arabic"], ["ja", "Japanese"], ["zh", "Chinese"]];
const LANG_NAME = Object.fromEntries(LANGS);
const MODEL_LABEL = { tiny: "Fastest (tiny)", base: "Fast (base)", small: "Balanced (small)", medium: "Accurate (medium)",
  "large-v3": "Most accurate (large-v3)", turbo: "Accurate + fast (turbo)",
  "whisper-large-v3": "Most accurate (Whisper large-v3)", "whisper-large-v3-turbo": "Faster (Whisper large-v3 turbo)" };
const ENGINE_LABEL = { groq: "Whisper · Groq cloud", local: "Whisper · this computer", none: "No speech engine" };
const STATUS = { queued: "Queued", decoding: "Reading audio", transcribing: "Transcribing", done: "Ready", error: "Failed", interrupted: "Paused" };

let CONFIG = { engine: "none", models: [], default_model: "", chunk_seconds: 110, ai_enabled: false, link_import: false, auth_required: false };

// ---------------- API (adds the app password if the server needs one) ----------------
const PW_KEY = "voxscribe_pw";
function pw() { try { return localStorage.getItem(PW_KEY) || ""; } catch { return ""; } }
async function api(path, opts = {}) {
  opts.headers = Object.assign({}, opts.headers, { "X-App-Password": pw() });
  const r = await fetch(path, opts);
  if (r.status === 401) { try { localStorage.removeItem(PW_KEY); } catch {} showLogin(); throw new Error("Password required"); }
  if (!r.ok) {
    let m = r.statusText; try { m = (await r.json()).detail || m; } catch {}
    const e = new Error(m); e.status = r.status; e.retryAfter = +r.headers.get("Retry-After") || 0; throw e;
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r;
}

// ---------------- IndexedDB storage ----------------
const db = (() => {
  let p;
  const open = () => p || (p = new Promise((res, rej) => {
    const req = indexedDB.open("voxscribe", 1);
    req.onupgradeneeded = () => { const d = req.result; d.createObjectStore("transcripts", { keyPath: "id" }); d.createObjectStore("media", { keyPath: "id" }); };
    req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error);
  }));
  const run = async (store, mode, fn) => { const d = await open(); return new Promise((res, rej) => {
    const tx = d.transaction(store, mode); const r = fn(tx.objectStore(store));
    tx.oncomplete = () => res(r && r.result); tx.onerror = () => rej(tx.error); tx.onabort = () => rej(tx.error); }); };
  return {
    get: (store, id) => run(store, "readonly", s => s.get(id)),
    put: (store, val) => run(store, "readwrite", s => s.put(val)),
    del: (store, id) => run(store, "readwrite", s => s.delete(id)),
    all: (store) => run(store, "readonly", s => s.getAll()),
  };
})();
const live = {};  // in-memory copies of records being transcribed (fast UI updates)
async function getRec(id) { return live[id] || db.get("transcripts", id); }
async function saveRec(rec) { await db.put("transcripts", rec); }

// ---------------- audio: decode, mono 16 kHz, chunk, WAV ----------------
const SR = 16000;
async function decodeToMono16k(blob) {
  if (blob.size > 700 * 1024 * 1024) throw new Error("File is too large for the browser to process (over 700 MB). Trim it or extract the audio first.");
  const buf = await blob.arrayBuffer();
  const ctx = new OfflineAudioContext(1, SR, SR);  // decodeAudioData resamples to the context rate
  let audio;
  try { audio = await ctx.decodeAudioData(buf); }
  catch { throw new Error("This browser can't read the audio in this file. Try MP3, M4A, WAV or MP4."); }
  const n = audio.length, ch = audio.numberOfChannels, out = new Float32Array(n);
  for (let c = 0; c < ch; c++) { const d = audio.getChannelData(c); for (let i = 0; i < n; i++) out[i] += d[i] / ch; }
  return out;
}

// End a chunk at the quietest moment in its last few seconds, so words aren't cut in half.
function quietCut(pcm, start, end) {
  const win = Math.floor(0.3 * SR), from = Math.max(start + 30 * SR, end - 12 * SR);
  let best = end, bestE = Infinity;
  for (let i = from; i + win <= end; i += Math.floor(win / 2)) {
    let e = 0; for (let j = i; j < i + win; j += 4) e += pcm[j] * pcm[j];
    if (e < bestE) { bestE = e; best = i + Math.floor(win / 2); }
  }
  return best;
}

function wav(pcm) {
  const b = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(b);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); w(8, "WAVE"); w(12, "fmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, pcm.length * 2, true);
  for (let i = 0, o = 44; i < pcm.length; i++, o += 2) { const s = Math.max(-1, Math.min(1, pcm[i])); v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
  return new Blob([b], { type: "audio/wav" });
}

// ---------------- job queue (one file at a time) ----------------
const queue = []; let running = false;
function enqueue(id) { if (!queue.includes(id)) queue.push(id); pump(); }
async function pump() {
  if (running) return; running = true;
  while (queue.length) { const id = queue.shift(); try { await runJob(id); } catch (e) { console.error(e); } }
  running = false;
}
window.addEventListener("beforeunload", e => { if (running) { e.preventDefault(); e.returnValue = ""; } });

async function runJob(id) {
  const rec = await db.get("transcripts", id); if (!rec) return;
  live[id] = rec;
  const set = async (f) => { Object.assign(rec, f); await saveRec(rec); notify(id); };
  try {
    const m = await db.get("media", id); if (!m) throw new Error("The media for this file is missing.");
    await set({ status: "decoding", error: "" });
    const pcm = await decodeToMono16k(m.blob);
    const total = pcm.length; await set({ duration: total / SR, status: "transcribing" });
    const step = CONFIG.chunk_seconds * SR;
    let pos = Math.round((rec.next_offset || 0) * SR);
    let lang = rec.language_requested || rec.language || "";
    while (pos < total) {
      let end = Math.min(pos + step, total);
      if (end < total) end = quietCut(pcm, pos, end);
      const fd = new FormData();
      fd.append("file", wav(pcm.subarray(pos, end)), "chunk.wav");
      fd.append("language", lang); fd.append("model", rec.model); fd.append("task", rec.translate ? "translate" : "transcribe");
      fd.append("detect", String(!lang));
      const out = await sendChunk(fd, rec, set);
      const off = pos / SR;
      if (!lang && out.language) { lang = out.language; rec.language = lang; }
      for (const s of out.segments) rec.segments.push({ start: s.start + off, end: s.end + off, text: s.text, speaker: "" });
      pos = end;
      await set({ next_offset: pos / SR, progress: pos / total });
    }
    await set({ status: "done", progress: 1, language: rec.language || lang || "?" });
    toast(`“${rec.title}” is ready`);
  } catch (e) {
    await set({ status: e.status === 401 ? "interrupted" : "error", error: e.message || String(e) });
  } finally { delete live[id]; notify(id); }
}

async function sendChunk(fd, rec, set) {
  for (let attempt = 0; ; attempt++) {
    try { return await api("/api/transcribe", { method: "POST", body: fd }); }
    catch (e) {
      const retriable = e.status === 429 || e.status >= 500 || !e.status;
      if (!retriable || attempt >= 6) throw e;
      const wait = e.status === 429 ? Math.max(e.retryAfter || 20, 5) : 3 * (attempt + 1);
      rec.note = e.status === 429 ? `Free-tier limit reached, waiting ${Math.round(wait)}s…` : "Connection problem, retrying…";
      notify(rec.id);
      await sleep(wait * 1000);
      rec.note = "";
    }
  }
}

const listeners = new Set();
function notify(id) { listeners.forEach(fn => fn(id)); }

async function createJob(blob, name, opts) {
  const id = uid();
  const rec = { id, created: Date.now() / 1000, title: name.replace(/\.[^.]+$/, "") || "Untitled", status: "queued", progress: 0, error: "",
    model: opts.model, language_requested: opts.language, translate: opts.translate, language: "", duration: 0,
    segments: [], summary: "", next_offset: 0, media_name: name, media_type: blob.type || "" };
  await db.put("media", { id, blob });
  await saveRec(rec);
  enqueue(id);
  return id;
}

// ---------------- router ----------------
let pollTimer = null;
async function route() {
  clearTimeout(pollTimer); listeners.clear();
  if (dirty) await flushSave();
  const h = location.hash || "#/";
  $$(".nav").forEach(n => n.classList.remove("active"));
  if (h.startsWith("#/t/")) return renderEditor(h.slice(4));
  if (h === "#/files") { $("#nav-files").classList.add("active"); return renderHome(true); }
  $("#nav-new").classList.add("active"); renderHome(false);
}
window.addEventListener("hashchange", route);

function showLogin() {
  $("#view").innerHTML = `<div class="card login"><h1>VoxScribe</h1><p class="sub">This app is protected. Enter the password.</p>
    <form id="loginf"><input type="password" id="pwin" class="url" placeholder="Password" autofocus>
    <button class="btn primary" style="margin-top:12px;width:100%;justify-content:center">Open</button></form><p class="sub" id="loginerr" style="color:var(--danger)"></p></div>`;
  $("#loginf").onsubmit = async e => {
    e.preventDefault(); try { localStorage.setItem(PW_KEY, $("#pwin").value); } catch {}
    try { await api("/api/login", { method: "POST" }); route(); } catch (err) { $("#loginerr").textContent = "Wrong password"; }
  };
}

// ---------------- home: new transcription + files ----------------
let chosenFile = null, recorder = null, recChunks = [], recBlob = null, recTimer = null, mode = "upload";

function renderHome(filesOnly) {
  chosenFile = null; recBlob = null;
  const noEngine = CONFIG.engine === "none";
  $("#view").innerHTML = `
    ${filesOnly ? `<h1>My files</h1><p class="sub">Your transcripts are saved in this browser on this device.</p>` : `
    <h1>New transcription</h1>
    <p class="sub">Turn audio or video into an editable transcript. Copy it, or export as Markdown, PDF, TXT or subtitles.</p>
    ${noEngine ? `<div class="card notice">No speech engine is set up on the server. Add <code>GROQ_API_KEY</code> (free at console.groq.com) and redeploy. See the README.</div>` : ""}
    <div class="card">
      <div class="tabs">
        <button class="tab active" data-mode="upload">Upload file</button>
        <button class="tab" data-mode="record">Record audio</button>
        ${CONFIG.link_import ? `<button class="tab" data-mode="link">From a link</button>` : ""}
      </div>
      <div class="pane" id="pane"></div>
      <div class="opts">
        <div class="opt"><label>Spoken language</label>
          <select id="lang">${LANGS.map(([v, n]) => `<option value="${v}">${n}</option>`).join("")}</select></div>
        <div class="opt"><label>Accuracy</label>
          <select id="model">${CONFIG.models.map(m => `<option value="${m}" ${m === CONFIG.default_model ? "selected" : ""}>${MODEL_LABEL[m] || m}</option>`).join("")}</select></div>
        <label class="toggle"><input type="checkbox" id="translate"> Translate to English</label>
        <div class="spacer"></div>
        <button class="btn primary" id="start" ${noEngine ? "disabled" : ""}>Transcribe
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button>
      </div>
    </div>`}
    <div class="section-title"><h2>${filesOnly ? "" : "Recent files"}</h2></div>
    <div class="card" id="files"><div class="empty">Loading…</div></div>`;

  if (!filesOnly) {
    $$(".tab").forEach(t => t.onclick = () => { $$(".tab").forEach(x => x.classList.toggle("active", x === t)); mode = t.dataset.mode; renderPane(); });
    mode = "upload"; renderPane();
    $("#lang").onchange = e => { if (e.target.value === "hi" && $("#model").value === "small") toast("Tip: 'Accurate (medium)' or 'turbo' gives much better Hindi."); };
    $("#start").onclick = start;
  }
  loadFiles();
  listeners.add(() => { clearTimeout(pollTimer); pollTimer = setTimeout(loadFiles, 300); });
}

function renderPane() {
  const p = $("#pane");
  if (mode === "upload") {
    p.innerHTML = `<div class="drop" id="drop">
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#0f766e" stroke-width="1.8" stroke-linecap="round"><path d="M12 16V4M7 9l5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>
        <strong>Drop a file here or click to browse</strong>
        <small>MP3, WAV, M4A, OGG, FLAC, MP4, MOV, WEBM</small>
        <div class="chosen" id="chosen">${chosenFile ? esc(chosenFile.name) : ""}</div>
      </div><input type="file" id="fileinput" hidden accept="audio/*,video/*">`;
    const drop = $("#drop"), inp = $("#fileinput");
    drop.onclick = () => inp.click();
    inp.onchange = () => { chosenFile = inp.files[0]; $("#chosen").textContent = chosenFile?.name || ""; };
    drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
    drop.ondragleave = () => drop.classList.remove("over");
    drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); chosenFile = e.dataTransfer.files[0]; $("#chosen").textContent = chosenFile?.name || ""; };
  } else if (mode === "record") {
    p.innerHTML = `<div class="rec">
        <button class="recbtn" id="recbtn" title="Start / stop recording"></button>
        <div class="timer" id="rectime">0:00</div>
        <div class="sub" id="rechint" style="margin:0">${recBlob ? "Recording ready. Press Transcribe." : "Click to start recording from your microphone"}</div>
        <audio id="recplay" controls ${recBlob ? "" : "hidden"}></audio>
      </div>`;
    if (recBlob) $("#recplay").src = URL.createObjectURL(recBlob);
    $("#recbtn").onclick = toggleRecord;
  } else {
    p.innerHTML = `<input type="text" class="url" id="url" placeholder="Paste a YouTube or other video link — https://…">
      <p class="sub" style="margin:8px 0 0;font-size:13px">Only use links to media you have the right to download.</p>`;
  }
}

async function toggleRecord() {
  const btn = $("#recbtn");
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recChunks = []; recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => e.data.size && recChunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop()); clearInterval(recTimer);
      recBlob = new Blob(recChunks, { type: recorder.mimeType || "audio/webm" });
      btn.classList.remove("on"); $("#rechint").textContent = "Recording ready. Press Transcribe.";
      const a = $("#recplay"); a.hidden = false; a.src = URL.createObjectURL(recBlob);
    };
    recorder.start(); btn.classList.add("on"); $("#rechint").textContent = "Recording… click to stop";
    const t0 = Date.now(); recTimer = setInterval(() => $("#rectime").textContent = fmt((Date.now() - t0) / 1000), 250);
  } catch (e) { toast("Microphone not available: " + e.message); }
}

async function start() {
  const opts = { language: $("#lang").value, model: $("#model").value, translate: $("#translate").checked };
  const btn = $("#start");
  let blob, name;
  if (mode === "upload") { if (!chosenFile) return toast("Choose a file first"); blob = chosenFile; name = chosenFile.name; }
  else if (mode === "record") {
    if (!recBlob) return toast("Record something first");
    blob = recBlob; name = `Recording ${new Date().toLocaleString().replace(/[/:]/g, "-")}.webm`;
  } else {
    const u = $("#url").value.trim(); if (!u) return toast("Paste a link first");
    btn.disabled = true; btn.textContent = "Downloading…";
    try {
      const r = await api("/api/fetch-link", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: u }) });
      blob = await r.blob();
      name = decodeURIComponent((r.headers.get("content-disposition") || "").match(/filename\*?=(?:UTF-8'')?"?([^";]+)/i)?.[1] || "Link import");
    } catch (e) { toast(e.message); btn.disabled = false; btn.textContent = "Transcribe"; return; }
  }
  const id = await createJob(blob, name, opts);
  location.hash = "#/t/" + id;
}

async function loadFiles() {
  const box = $("#files"); if (!box) return;
  const items = (await db.all("transcripts")).map(r => live[r.id] || r).sort((a, b) => b.created - a.created);
  if (!items.length) { box.innerHTML = `<div class="empty">No transcripts yet. Upload your first file above.</div>`; return; }
  box.innerHTML = `<table><thead><tr><th>Name</th><th>Status</th><th>Language</th><th>Duration</th><th>Created</th><th></th></tr></thead><tbody>
    ${items.map(m => {
      const busy = ["queued", "decoding", "transcribing"].includes(m.status);
      const cls = m.status === "done" ? "done" : m.status === "error" ? "error" : "busy";
      return `<tr data-id="${m.id}">
        <td><div class="fname"><span class="ficon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#0f766e" stroke-width="2.2" stroke-linecap="round"><path d="M4 10v4M8 6v12M12 3v18M16 7v10M20 10v4"/></svg></span>${esc(m.title)}</div></td>
        <td><span class="pill ${cls}">${STATUS[m.status] || m.status}${m.status === "transcribing" ? " " + Math.round(m.progress * 100) + "%" : ""}</span>
          ${busy ? `<div class="bar"><i style="width:${Math.round(m.progress * 100)}%"></i></div>` : ""}</td>
        <td>${esc(LANG_NAME[m.language] || m.language || "—")}${m.translate ? " → English" : ""}</td>
        <td>${m.duration ? fmt(m.duration) : "—"}</td>
        <td>${ago(m.created)}</td>
        <td style="text-align:right"><button class="btn ghost sm del" title="Delete">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/></svg></button></td></tr>`;
    }).join("")}</tbody></table>`;
  $$("tbody tr", box).forEach(tr => tr.onclick = async e => {
    if (e.target.closest(".del")) {
      const id = tr.dataset.id;
      if (live[id]) return toast("Wait until this file finishes, then delete it.");
      if (confirm("Delete this transcript and its media from this browser?")) {
        const qi = queue.indexOf(id); if (qi >= 0) queue.splice(qi, 1);
        await db.del("transcripts", id); await db.del("media", id); loadFiles();
      }
      return;
    }
    location.hash = "#/t/" + tr.dataset.id;
  });
}

// ---------------- editor ----------------
let T = null, dirty = false, saveTimer = null, mediaURL = null;

async function renderEditor(id) {
  const m = await getRec(id);
  if (!m) { $("#view").innerHTML = `<div class="empty">This transcript isn't in this browser.</div>`; return; }
  T = m;
  if (m.status !== "done") return renderProgress(m);
  const media = await db.get("media", id);
  if (mediaURL) URL.revokeObjectURL(mediaURL);
  mediaURL = media ? URL.createObjectURL(media.blob) : "";
  $("#view").innerHTML = `
    <div class="topbar">
      <a class="btn ghost sm" href="#/files">← Files</a>
      <input class="title-input" id="title" value="${esc(m.title)}">
      <span class="saved" id="saved">All changes saved</span>
      <button class="btn" id="copybtn" title="Copy the whole transcript to the clipboard">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>
        <span id="copylabel">Copy text</span></button>
      <div class="menu"><button class="btn primary" id="expbtn">Export
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M6 9l6 6 6-6"/></svg></button>
        <div class="menu-pop" id="exppop">
          <button data-fmt="md">Markdown <span class="ext">.md</span></button>
          <button data-fmt="pdf">PDF <span class="ext">.pdf</span></button>
          <button data-fmt="txt">Plain text <span class="ext">.txt</span></button>
          <button data-fmt="srt">Subtitles <span class="ext">.srt</span></button>
          <hr>
          <button data-copy="meta">Copy with timestamps</button>
          <hr>
          <label class="chk"><input type="checkbox" id="exp-ts" checked> Include timestamps</label>
          <label class="chk"><input type="checkbox" id="exp-sum" checked> Include AI summary</label>
        </div></div>
    </div>
    <div class="editor">
      <div class="card doc">
        <div class="findbar">
          <input type="text" id="find" placeholder="Find…">
          <input type="text" id="repl" placeholder="Replace with…">
          <button class="btn sm" id="replall">Replace all</button>
          <label class="toggle" style="padding:0;font-size:13px;color:var(--muted)"><input type="checkbox" id="follow" checked> Follow playback</label>
        </div>
        <div id="doc"></div>
      </div>
      <div class="side">
        <div class="card player">
          ${mediaURL ? `<video id="media" controls preload="metadata" src="${mediaURL}"></video>` : `<div class="sub">Media not available in this browser.</div><video id="media" hidden></video>`}
          <div class="player-row"><span>Click any sentence to jump there</span>
            <select id="speed" style="padding:3px 6px">${[0.75, 1, 1.25, 1.5, 2].map(s => `<option ${s === 1 ? "selected" : ""} value="${s}">${s}×</option>`).join("")}</select></div>
        </div>
        <div class="card"><div class="inner">
          <h3>AI tools</h3>
          <div class="ai-btns">
            <button class="btn" data-ai="summary">✦ Summarize</button>
            <button class="btn" data-ai="cleanup">✎ Fix punctuation &amp; spelling</button>
            <button class="btn" data-ai="speakers">☺ Detect speakers</button>
          </div>
          <div class="ai-note">${CONFIG.ai_enabled ? "Using " + esc(CONFIG.ai_model) + "." : "AI tools are off. Set an AI key on the server (see README)."}</div>
        </div></div>
        <div class="card" id="sumcard" ${m.summary ? "" : "hidden"}><div class="inner"><h3>Summary</h3><div class="summary" id="summary"></div></div></div>
        <div class="card"><div class="inner"><h3>Details</h3>
          <dl class="details">
            <dt>Language</dt><dd>${esc(LANG_NAME[m.language] || m.language)}${m.translate ? " → English" : ""}</dd>
            <dt>Duration</dt><dd>${fmt(m.duration)}</dd>
            <dt>Model</dt><dd>${esc(MODEL_LABEL[m.model] || m.model)}</dd>
            <dt>Words</dt><dd id="wc"></dd>
          </dl></div></div>
      </div>
    </div>`;

  renderDoc(); renderSummary();
  const mediaEl = $("#media");
  mediaEl.addEventListener("loadedmetadata", () => { if (!mediaEl.videoWidth) mediaEl.classList.add("audio-only"); });
  mediaEl.addEventListener("timeupdate", highlight);
  $("#speed").onchange = e => mediaEl.playbackRate = +e.target.value;
  $("#title").onchange = e => { T.title = e.target.value.trim() || T.title; queueSave(); };
  $("#expbtn").onclick = e => { e.stopPropagation(); $("#exppop").classList.toggle("open"); };
  document.onclick = e => { if (!e.target.closest(".menu")) $("#exppop")?.classList.remove("open"); };
  $$("#exppop button[data-fmt]").forEach(b => b.onclick = async () => {
    await flushSave(); $("#exppop").classList.remove("open");
    exportAs(b.dataset.fmt, $("#exp-ts").checked, $("#exp-sum").checked);
  });
  $("#copybtn").onclick = () => copyTranscript(false);
  $$("#exppop button[data-copy]").forEach(b => b.onclick = () => { $("#exppop").classList.remove("open"); copyTranscript(true); });
  $("#find").oninput = renderDoc;
  $("#replall").onclick = replaceAll;
  $$("[data-ai]").forEach(b => { b.disabled = !CONFIG.ai_enabled; b.onclick = () => runAI(b); });
}

function renderProgress(m) {
  const pct = Math.round((m.progress || 0) * 100);
  const isErr = m.status === "error" || m.status === "interrupted";
  const msg = { queued: "Waiting for the previous file to finish…", decoding: `Reading the audio in your browser… (about 1 minute per 30 minutes of audio)`,
    transcribing: m.note || "Transcribing…" }[m.status] || "";
  $("#view").innerHTML = `<div class="topbar"><a class="btn ghost sm" href="#/files">← Files</a></div>
    <div class="card progress-card">
      ${isErr ? `<h2 style="margin:0 0 6px">${m.status === "error" ? "Transcription stopped" : "Transcription paused"}</h2>
        <p class="sub" style="margin:0 0 6px">${esc(m.error || "The page was closed before it finished.")}</p>
        ${m.segments.length ? `<p class="sub" style="margin:0">${pct}% is done and saved. Resume to continue from there.</p>` : ""}
        <div style="display:flex;gap:8px;justify-content:center;margin-top:16px">
          <button class="btn primary" id="resume">Resume</button>
          ${m.segments.length ? `<button class="btn" id="keep">Keep what's done</button>` : ""}</div>` : `
      <div class="spinner"></div>
      <h2 style="margin:16px 0 4px">${esc(m.title)}</h2>
      <p class="sub" style="margin:0">${esc(msg)}</p>
      <div class="bigbar"><i style="width:${pct}%"></i></div><div class="sub" style="margin:0">${pct}%</div>
      <p class="sub" style="margin-top:14px;font-size:12px">Keep this tab open until it finishes. You can browse other pages of the app meanwhile.</p>`}
    </div>`;
  if (isErr) {
    $("#resume").onclick = () => { Object.assign(m, { status: "queued", error: "" }); enqueue(m.id); renderProgress(m); };
    const k = $("#keep"); if (k) k.onclick = async () => { m.status = "done"; await saveRec(m); renderEditor(m.id); };
  } else {
    listeners.clear();
    listeners.add(id => { if (id === m.id) getRec(m.id).then(renderProgressOrEditor); });
  }
}
function renderProgressOrEditor(m) { if (!m || location.hash !== "#/t/" + m.id) return; m.status === "done" ? renderEditor(m.id) : renderProgress(m); }

function paragraphs(segs) {
  const out = [];
  segs.forEach((s, i) => {
    const last = out[out.length - 1];
    if (!last || s.start - segs[i - 1].end > 2 || last.length >= 6 || (s.speaker || "") !== (segs[i - 1].speaker || "")) out.push([i]);
    else last.push(i);
  });
  return out;
}

const reEsc = q => q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
function highlightText(text, q) {
  if (!q) return esc(text);
  return text.split(new RegExp(`(${reEsc(q)})`, "gi")).map((p, i) => i % 2 ? `<mark>${esc(p)}</mark>` : esc(p)).join("");
}

function renderDoc() {
  const q = $("#find")?.value || "", segs = T.segments;
  $("#doc").innerHTML = segs.length ? paragraphs(segs).map(idx => {
    const first = segs[idx[0]];
    return `<div class="para">
      <div class="meta-col">
        <span class="spk ${first.speaker ? "" : "none"}" data-i="${idx[0]}" title="Click to rename speaker">${esc(first.speaker || "+ Speaker")}</span>
        <span class="ts" data-t="${first.start}">${fmt(first.start)}</span>
      </div>
      <div class="ptext">${idx.map(i => `<span class="seg" contenteditable="true" spellcheck="false" data-i="${i}">${highlightText(segs[i].text, q)}</span>`).join(" ")}</div>
    </div>`;
  }).join("") : `<div class="empty">No speech was detected in this file.</div>`;
  $("#wc").textContent = segs.reduce((n, s) => n + s.text.split(/\s+/).filter(Boolean).length, 0).toLocaleString();
  const media = $("#media");
  $$(".ts").forEach(el => el.onclick = () => { media.currentTime = +el.dataset.t; media.play(); });
  $$(".seg").forEach(el => {
    el.onclick = () => { media.currentTime = segs[+el.dataset.i].start + 0.01; };
    el.onkeydown = e => { if (e.key === "Enter") { e.preventDefault(); el.blur(); } };
    el.onblur = () => { const txt = el.innerText.replace(/\s+/g, " ").trim(); if (txt !== segs[+el.dataset.i].text) { segs[+el.dataset.i].text = txt; queueSave(); } };
  });
  $$(".spk").forEach(el => el.onclick = () => renameSpeaker(+el.dataset.i));
}

// ---- copy / export (all done in the browser) ----
function transcriptText(withMeta) {
  const segs = T.segments;
  return paragraphs(segs).map(idx => {
    const first = segs[idx[0]], text = idx.map(i => segs[i].text.trim()).join(" ");
    const label = withMeta ? [first.speaker, fmt(first.start)].filter(Boolean).join(" · ") : first.speaker;
    return (label ? `[${label}] ` : "") + text;
  }).join("\n\n");
}

async function copyTranscript(withMeta) {
  if (!T.segments.length) return toast("Nothing to copy");
  const txt = transcriptText(withMeta);
  try { await navigator.clipboard.writeText(txt); }
  catch {
    const ta = document.createElement("textarea"); ta.value = txt; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select(); const ok = document.execCommand("copy"); ta.remove();
    if (!ok) return toast("Couldn't copy. Select the text and press Cmd+C.");
  }
  const lab = $("#copylabel"); lab.textContent = "Copied ✓"; setTimeout(() => { if (lab) lab.textContent = "Copy text"; }, 1600);
  toast(`Copied ${txt.split(/\s+/).filter(Boolean).length.toLocaleString()} words${withMeta ? " with timestamps" : ""}`);
}

function metaLines() {
  return [["Source", T.media_name || T.title], ["Language", (LANG_NAME[T.language] || T.language) + (T.translate ? " → English" : "")],
    ["Duration", fmt(T.duration)], ["Generated", new Date().toLocaleString()]];
}

function download(name, content, type) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([content], { type })); a.download = name;
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function exportAs(fmtName, ts, withSum) {
  const segs = T.segments, base = (T.title || "transcript").replace(/[\\/:*?"<>|]/g, "_");
  const label = idx => [segs[idx[0]].speaker, ts ? fmt(segs[idx[0]].start) : ""].filter(Boolean).join(" · ");
  const paras = paragraphs(segs).map(idx => ({ label: label(idx), text: idx.map(i => segs[i].text.trim()).join(" ") }));
  const summary = withSum ? (T.summary || "").trim() : "";
  if (fmtName === "txt") {
    download(base + ".txt", [`Transcript: ${T.title}`, "", ...(summary ? ["SUMMARY", summary, "", "TRANSCRIPT"] : []),
      ...paras.map(p => (p.label ? `[${p.label}] ` : "") + p.text + "\n")].join("\n"), "text/plain;charset=utf-8");
  } else if (fmtName === "md") {
    const demoted = summary.replace(/^#{1,6}\s+/gm, "### ");
    download(base + ".md", [`# ${T.title}`, "", ...metaLines().map(([k, v]) => `- **${k}:** ${v}`), "", "---", "",
      ...(summary ? ["## Summary", "", demoted, "", "## Transcript", ""] : []),
      ...paras.map(p => (p.label ? `**[${p.label}]** ` : "") + p.text + "\n")].join("\n"), "text/markdown;charset=utf-8");
  } else if (fmtName === "srt") {
    const t = x => { const ms = Math.round(x * 1000), h = Math.floor(ms / 3600000), m = Math.floor(ms % 3600000 / 60000), s = Math.floor(ms % 60000 / 1000);
      return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(ms % 1000).padStart(3, "0")}`; };
    download(base + ".srt", segs.map((s, i) => `${i + 1}\n${t(s.start)} --> ${t(s.end)}\n${s.speaker ? s.speaker + ": " : ""}${s.text.trim()}\n`).join("\n"), "application/x-subrip");
  } else if (fmtName === "pdf") {
    printPdf(paras, summary);
  }
}

// PDF: the browser's own "Save as PDF" renders every script (Hindi included) perfectly.
function printPdf(paras, summary) {
  const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${esc(T.title)}</title><style>
    @page { margin: 18mm; } body { font: 11.5pt/1.6 -apple-system, "Segoe UI", Roboto, "Noto Sans", "Noto Sans Devanagari", sans-serif; color: #1c2026; }
    h1 { font-size: 20pt; margin: 0 0 6px; } .meta { color: #6b7280; font-size: 9pt; border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-bottom: 16px; }
    h2 { font-size: 13pt; margin: 18px 0 6px; } .lab { color: #0f766e; font-size: 9pt; font-weight: 600; margin-top: 12px; }
    p { margin: 2px 0 8px; } .sum { background: #f6f7f9; padding: 8px 14px; border-radius: 6px; } ul { margin: 4px 0; }
  </style></head><body><h1>${esc(T.title)}</h1><div class="meta">${metaLines().map(([k, v]) => `${k}: ${esc(v)}`).join(" &nbsp;·&nbsp; ")}</div>
  ${summary ? `<h2>Summary</h2><div class="sum">${mdToHtml(summary)}</div><h2>Transcript</h2>` : ""}
  ${paras.map(p => `${p.label ? `<div class="lab">${esc(p.label)}</div>` : ""}<p>${esc(p.text)}</p>`).join("")}</body></html>`;
  const f = document.createElement("iframe"); f.style.cssText = "position:fixed;right:0;bottom:0;width:0;height:0;border:0";
  document.body.appendChild(f); f.contentDocument.open(); f.contentDocument.write(html); f.contentDocument.close();
  setTimeout(() => { f.contentWindow.focus(); f.contentWindow.print(); toast("Choose “Save as PDF” in the print window"); setTimeout(() => f.remove(), 60000); }, 300);
}

function renameSpeaker(i) {
  const segs = T.segments, old = segs[i].speaker || "";
  const name = prompt(old ? `Rename "${old}" everywhere to:` : "Speaker name for this paragraph:", old || "Speaker 1");
  if (name === null) return;
  if (old) segs.forEach(s => { if (s.speaker === old) s.speaker = name.trim(); });
  else paragraphs(segs).find(p => p.includes(i)).forEach(j => segs[j].speaker = name.trim());
  renderDoc(); queueSave();
}

function replaceAll() {
  const q = $("#find").value, r = $("#repl").value;
  if (!q) return toast("Type something to find");
  const re = new RegExp(reEsc(q), "gi"); let n = 0;
  T.segments.forEach(s => { s.text = s.text.replace(re, () => (n++, r)); });
  toast(n ? `Replaced ${n} occurrence${n > 1 ? "s" : ""}` : "No matches");
  if (n) { $("#find").value = r; renderDoc(); queueSave(); }
}

let activeSeg = -1;
function highlight() {
  const t = $("#media").currentTime, segs = T.segments;
  const i = segs.findIndex(s => t >= s.start && t < s.end + 0.15);
  if (i === activeSeg) return;
  $(".seg.active")?.classList.remove("active"); activeSeg = i;
  if (i < 0) return;
  const el = $(`.seg[data-i="${i}"]`); if (!el) return;
  el.classList.add("active");
  if ($("#follow").checked && document.activeElement !== el) {
    const r = el.getBoundingClientRect();
    if (r.top < 90 || r.bottom > innerHeight - 40) el.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

function mdToHtml(md) {
  const lines = esc(md).split("\n"); let html = "", inList = false;
  for (let l of lines) {
    l = l.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
    if (/^\s*[-*•]\s+/.test(l)) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${l.replace(/^\s*[-*•]\s+/, "")}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    if (/^#+\s/.test(l)) html += `<h3>${l.replace(/^#+\s/, "")}</h3>`; else if (l.trim()) html += `<p>${l}</p>`;
  }
  return html + (inList ? "</ul>" : "");
}
function renderSummary() { if (!T.summary) return; $("#sumcard").hidden = false; $("#summary").innerHTML = mdToHtml(T.summary); }

async function runAI(btn) {
  const action = btn.dataset.ai, label = btn.textContent;
  $$("[data-ai]").forEach(b => b.disabled = true);
  try {
    await flushSave();
    if (action === "summary") {
      btn.textContent = "Working…";
      const r = await api("/api/ai", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action, segments: T.segments }) });
      T.summary = r.summary; renderSummary();
    } else {
      // Send in batches so each request stays short (serverless time limits).
      const B = 80, out = [], n = Math.ceil(T.segments.length / B); let prev = "";
      for (let k = 0; k < n; k++) {
        btn.textContent = `Working… ${k + 1}/${n}`;
        const r = await api("/api/ai", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action, segments: T.segments.slice(k * B, (k + 1) * B), prev }) });
        out.push(...r.segments); prev = r.segments[r.segments.length - 1]?.speaker || prev;
      }
      T.segments = out.map((s, i) => Object.assign({}, T.segments[i], { text: s.text, speaker: s.speaker || "" }));
      renderDoc();
    }
    await saveRec(T);
    toast({ summary: "Summary ready", cleanup: "Transcript cleaned up", speakers: "Speakers labeled — click a name to rename" }[action]);
  } catch (e) { toast(e.message); }
  btn.textContent = label;
  $$("[data-ai]").forEach(b => b.disabled = !CONFIG.ai_enabled);
}

function queueSave() { dirty = true; $("#saved").textContent = "Saving…"; clearTimeout(saveTimer); saveTimer = setTimeout(flushSave, 600); }
async function flushSave() {
  clearTimeout(saveTimer); if (!dirty || !T) return; dirty = false;
  try { await saveRec(T); const s = $("#saved"); if (s) s.textContent = "All changes saved"; }
  catch (e) { dirty = true; const s = $("#saved"); if (s) s.textContent = "Not saved: " + e.message; }
}

// ---- bring in transcripts saved by the older server-side version (local only) ----
async function importLegacy() {
  if (!CONFIG.legacy) return;
  try { if (localStorage.getItem("voxscribe_legacy_done")) return; } catch {}
  try {
    const items = await api("/api/legacy");
    for (const m of items) {
      if (await db.get("transcripts", "legacy-" + m.id)) continue;
      let blob = null; try { blob = await (await api(`/api/legacy/${m.id}/media`)).blob(); } catch {}
      const id = "legacy-" + m.id;
      if (blob) await db.put("media", { id, blob });
      await saveRec({ id, created: m.created, title: m.title, status: "done", progress: 1, error: "", model: m.model,
        language_requested: m.language_requested, translate: m.translate, language: m.language, duration: m.duration,
        segments: m.segments, summary: m.summary || "", next_offset: m.duration, media_name: m.media, media_type: blob?.type || "" });
    }
    try { localStorage.setItem("voxscribe_legacy_done", "1"); } catch {}
    if (items.length) { toast(`Imported ${items.length} earlier transcript${items.length > 1 ? "s" : ""}`); if (!location.hash.startsWith("#/t/")) route(); }
  } catch (e) { console.warn("legacy import failed", e); }
}

// ---------------- boot ----------------
(async () => {
  try { CONFIG = Object.assign(CONFIG, await (await fetch("/api/config")).json()); } catch {}
  $("#engine-status").innerHTML = `<span class="dot ${CONFIG.engine !== "none" ? "on" : ""}"></span>${ENGINE_LABEL[CONFIG.engine] || CONFIG.engine}`;
  if (CONFIG.ai_enabled) $("#ai-status").innerHTML = `<span class="dot on"></span>AI tools on`;
  if (CONFIG.auth_required && !pw()) return showLogin();
  // Anything that was mid-way when the page closed can be resumed.
  for (const r of await db.all("transcripts")) {
    if (["queued", "decoding", "transcribing"].includes(r.status)) { r.status = "interrupted"; r.error = "The page was closed before it finished."; await saveRec(r); }
  }
  route();
  importLegacy();
})();

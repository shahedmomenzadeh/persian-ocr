"use strict";

const SUPPORTED_EXTS = new Set([".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]);

const BLOCK_COLORS = {
  title: "#2563eb",
  text: "#0f766e",
  table: "#d97706",
  header: "#7c3aed",
  footer: "#475569",
  figure: "#db2777",
};

const STAGE_LABELS = [
  { key: "preprocess_sec", label: "پیش‌پردازش", color: "#64748b" },
  { key: "layout_sec", label: "تشخیص چیدمان", color: "#2563eb" },
  { key: "detect_sec", label: "تشخیص خطوط", color: "#0284c7" },
  { key: "table_sec", label: "ساختار جدول", color: "#d97706" },
  { key: "ctc_sec", label: "تشخیص CTC", color: "#0f766e" },
  { key: "vlm_sec", label: "تشخیص VLM", color: "#b45309" },
  { key: "postprocess_sec", label: "پس‌پردازش RTL", color: "#7c3aed" },
  { key: "spellcheck_sec", label: "اصلاح املایی", color: "#15803d" },
];

const state = {
  mode: "minimal",
  model: "bina-0.2-rizeh",
  spellcheck: "kenlm",
  minimal: {
    formats: ["html", "md", "txt", "json"],
    queue: [], // { id, file, name, size, status: queued|running|done|error, data?, error? }
    running: false,
    results: [], // { name, data }
    activeDocIndex: 0,
    activeFmt: "html",
  },
  debug: {
    runs: [],
    originalDoc: null,
    workingDoc: null,
    pageImages: {}, // pageNumberStr -> url
    outputUrls: {}, // fmt -> url
    htmlText: "",
    mdText: "",
    txtText: "",
    activePageIndex: 0,
    selectedBlockId: null,
    zoom: null, // null means Fit
    layers: { blocks: true, lines: false, cells: false, labels: true },
    activeTab: "editor",
    activeLiveFmt: "html",
    blockFilter: "all",
    spellRuleFilter: "all",
    autoPersianDigits: true,
    dirty: false,
  },
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function showToast(msg, kind = "info") {
  const wrap = $("#toast-container");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  wrap.appendChild(el);
  setTimeout(() => el.remove(), 3600);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 1024) return `${bytes || 0} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function getExt(name) {
  const idx = name.lastIndexOf(".");
  return idx >= 0 ? name.slice(idx).toLowerCase() : "";
}

function isSupportedFile(file) {
  return file && SUPPORTED_EXTS.has(getExt(file.name));
}

/* ==================== Persian Digits & Visual RTL Caret Helpers ==================== */

const PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹";
const DIGIT_ANY_RE = /[0-9\u0660-\u0669]/;
const ASCII_LETTER_RE = /[A-Za-z]/;
const BIDI_EN_RE = /[0-9\u06F0-\u06F9\u0660-\u0669]/;
const BIDI_LATIN_RE = /[A-Za-z\u00C0-\u024F]/;
const BIDI_NUM_SEP_RE = /[\/\-.,:\u066B\u066C]/;
const BIDI_LATIN_SEP_RE = /[\/\-.,:_@?&=#+%~]/;

function toPersianDigits(str) {
  return String(str ?? "")
    .replace(/[0-9]/g, (d) => PERSIAN_DIGITS[Number(d)])
    .replace(/[\u0660-\u0669]/g, (d) => PERSIAN_DIGITS[d.charCodeAt(0) - 0x0660]);
}

function hasAsciiLetters(token) {
  return ASCII_LETTER_RE.test(token || "");
}

function convertDigitsPreservingLatinTokens(str) {
  return String(str ?? "").replace(/\S+/g, (token) => (hasAsciiLetters(token) ? token : toPersianDigits(token)));
}

function isInsideLatinToken(fullText, start, end) {
  let left = start;
  while (left > 0 && !/\s/.test(fullText[left - 1])) left -= 1;
  let right = end;
  while (right < fullText.length && !/\s/.test(fullText[right])) right += 1;
  const surrounding = fullText.slice(left, start) + fullText.slice(end, right);
  return hasAsciiLetters(surrounding);
}

function convertInsertedTextForEditor(fullText, start, end, insertedText) {
  if (!insertedText || !DIGIT_ANY_RE.test(insertedText)) return insertedText;
  if (!/\s/.test(insertedText)) {
    if (hasAsciiLetters(insertedText) || isInsideLatinToken(fullText, start, end)) {
      return insertedText;
    }
    return toPersianDigits(insertedText);
  }
  return convertDigitsPreservingLatinTokens(insertedText);
}

function segmentLineBidiRuns(lineText) {
  const n = lineText.length;
  if (n === 0) return [];

  const types = new Array(n);
  const dirs = new Array(n).fill("rtl");

  for (let i = 0; i < n; i += 1) {
    const ch = lineText[i];
    if (BIDI_EN_RE.test(ch)) types[i] = "EN";
    else if (BIDI_LATIN_RE.test(ch)) types[i] = "L";
    else if (BIDI_NUM_SEP_RE.test(ch)) types[i] = "NUM_SEP";
    else if (BIDI_LATIN_SEP_RE.test(ch)) types[i] = "LATIN_SEP";
    else if (ch === " " || ch === "\t") types[i] = "WS";
    else types[i] = "R";

    if (types[i] === "EN" || types[i] === "L") {
      dirs[i] = "ltr";
    }
  }

  // Keep numeric separators inside numbers/dates (e.g. ۱۴۷۰/۰۲۰, 1403/07/20, ۱۲:۳۰) as LTR
  for (let i = 1; i < n - 1; i += 1) {
    if (
      types[i] === "NUM_SEP" &&
      (types[i - 1] === "EN" || types[i - 1] === "L") &&
      (types[i + 1] === "EN" || types[i + 1] === "L")
    ) {
      dirs[i] = "ltr";
    }
  }

  // Keep Latin/URL separators (e.g. Bina-0.2-Rizeh, https://example.com/a) inside Latin spans as LTR
  const isLatinSpanChar = (t) => t === "L" || t === "EN" || t === "NUM_SEP" || t === "LATIN_SEP";
  for (let i = 0; i < n; i += 1) {
    if (!isLatinSpanChar(types[i])) continue;
    let spanEnd = i;
    while (spanEnd + 1 < n && isLatinSpanChar(types[spanEnd + 1])) spanEnd += 1;

    let hasLatin = false;
    for (let k = i; k <= spanEnd; k += 1) {
      if (types[k] === "L") {
        hasLatin = true;
        break;
      }
    }

    if (hasLatin) {
      let innerStart = i;
      while (innerStart <= spanEnd && types[innerStart] !== "L" && types[innerStart] !== "EN") {
        innerStart += 1;
      }
      let innerEnd = spanEnd;
      while (innerEnd >= i && types[innerEnd] !== "L" && types[innerEnd] !== "EN") {
        innerEnd -= 1;
      }
      for (let k = innerStart; k <= innerEnd; k += 1) {
        dirs[k] = "ltr";
      }
    }
    i = spanEnd;
  }

  // Merge spaces/tabs strictly between two LTR spans that contain Latin letters (e.g. "hello world")
  for (let i = 0; i < n; i += 1) {
    if (types[i] !== "WS") continue;
    let wsEnd = i;
    while (wsEnd + 1 < n && types[wsEnd + 1] === "WS") wsEnd += 1;

    let leftHasLatin = false;
    for (let k = i - 1; k >= 0 && dirs[k] === "ltr"; k -= 1) {
      if (types[k] === "L") {
        leftHasLatin = true;
        break;
      }
    }
    let rightHasLatin = false;
    for (let k = wsEnd + 1; k < n && dirs[k] === "ltr"; k += 1) {
      if (types[k] === "L") {
        rightHasLatin = true;
        break;
      }
    }
    if (leftHasLatin && rightHasLatin) {
      for (let k = i; k <= wsEnd; k += 1) dirs[k] = "ltr";
    }
    i = wsEnd;
  }

  const runs = [];
  let start = 0;
  for (let i = 1; i <= n; i += 1) {
    if (i === n || dirs[i] !== dirs[start]) {
      runs.push({ start, end: i, dir: dirs[start] });
      start = i;
    }
  }
  return runs;
}

function buildVisualLineSlots(lineText, lineStart = 0) {
  if (!lineText || lineText.length === 0) return [lineStart];
  const runs = segmentLineBidiRuns(lineText);
  const slots = [];
  for (let i = 0; i < runs.length; i += 1) {
    const r = runs[i];
    if (r.dir === "ltr") {
      for (let p = r.end; p >= r.start; p -= 1) {
        slots.push(lineStart + p);
      }
    } else {
      const from = i > 0 ? r.start + 1 : r.start;
      const to = i < runs.length - 1 ? r.end - 1 : r.end;
      for (let p = from; p <= to; p += 1) {
        slots.push(lineStart + p);
      }
    }
  }
  return slots;
}

function getLineBounds(text, pos) {
  const nl = pos > 0 ? text.lastIndexOf("\n", pos - 1) : -1;
  const lineStart = nl === -1 ? 0 : nl + 1;
  const nextNl = text.indexOf("\n", pos);
  const lineEnd = nextNl === -1 ? text.length : nextNl;
  return { lineStart, lineEnd, lineText: text.slice(lineStart, lineEnd) };
}

function computeVisualArrowTarget(text, selStart, selEnd, key) {
  if (key !== "ArrowLeft" && key !== "ArrowRight") return null;
  if (selStart == null || selEnd == null) return null;

  if (selStart !== selEnd) {
    const b1 = getLineBounds(text, selStart);
    const b2 = getLineBounds(text, selEnd);
    if (b1.lineStart === b2.lineStart) {
      const slots = buildVisualLineSlots(b1.lineText, b1.lineStart);
      const i1 = slots.indexOf(selStart);
      const i2 = slots.indexOf(selEnd);
      if (i1 !== -1 && i2 !== -1) {
        return key === "ArrowLeft" ? slots[Math.max(i1, i2)] : slots[Math.min(i1, i2)];
      }
    }
    return key === "ArrowLeft" ? selEnd : selStart;
  }

  const pos = selStart;
  const { lineStart, lineEnd, lineText } = getLineBounds(text, pos);
  const slots = buildVisualLineSlots(lineText, lineStart);
  const idx = slots.indexOf(pos);
  if (idx === -1) return null;

  if (key === "ArrowLeft") {
    if (idx < slots.length - 1) return slots[idx + 1];
    if (lineEnd < text.length) {
      const next = getLineBounds(text, lineEnd + 1);
      return buildVisualLineSlots(next.lineText, next.lineStart)[0];
    }
    return pos;
  }

  if (idx > 0) return slots[idx - 1];
  if (lineStart > 0) {
    const prev = getLineBounds(text, lineStart - 1);
    const prevSlots = buildVisualLineSlots(prev.lineText, prev.lineStart);
    return prevSlots[prevSlots.length - 1];
  }
  return pos;
}

function attachPersianEditorBehavior(el, onValueChange) {
  let handledInBeforeInput = false;

  el.addEventListener("keydown", (e) => {
    if (
      (e.key === "ArrowLeft" || e.key === "ArrowRight") &&
      !e.shiftKey &&
      !e.ctrlKey &&
      !e.altKey &&
      !e.metaKey &&
      !e.isComposing
    ) {
      const target = computeVisualArrowTarget(el.value, el.selectionStart, el.selectionEnd, e.key);
      if (target !== null) {
        e.preventDefault();
        el.setSelectionRange(target, target);
      }
    }
  });

  el.addEventListener("beforeinput", (e) => {
    handledInBeforeInput = false;
    if (!state.debug.autoPersianDigits || e.isComposing) return;
    if (
      (e.inputType === "insertText" ||
        e.inputType === "insertFromPaste" ||
        e.inputType === "insertReplacementText") &&
      typeof e.data === "string" &&
      DIGIT_ANY_RE.test(e.data)
    ) {
      const start = el.selectionStart ?? el.value.length;
      const end = el.selectionEnd ?? el.value.length;
      const converted = convertInsertedTextForEditor(el.value, start, end, e.data);
      if (converted !== e.data) {
        e.preventDefault();
        let ok = false;
        try {
          ok = typeof document.execCommand === "function" && document.execCommand("insertText", false, converted);
        } catch {
          ok = false;
        }
        if (ok) {
          handledInBeforeInput = true;
        } else {
          el.setRangeText(converted, start, end, "end");
          onValueChange();
        }
      }
    }
  });

  el.addEventListener("input", (e) => {
    const alreadyHandled = handledInBeforeInput;
    handledInBeforeInput = false;

    if (!alreadyHandled && state.debug.autoPersianDigits && !e.isComposing && typeof el.selectionStart === "number") {
      const caret = el.selectionStart;
      if (typeof e.data === "string" && DIGIT_ANY_RE.test(e.data) && caret >= e.data.length) {
        const insStart = caret - e.data.length;
        if (el.value.slice(insStart, caret) === e.data) {
          const converted = convertInsertedTextForEditor(el.value, insStart, caret, e.data);
          if (converted !== e.data) {
            el.setRangeText(converted, insStart, caret, "end");
          }
        }
      } else if (caret > 0 && (!e.inputType || e.inputType.startsWith("insert"))) {
        const ch = el.value[caret - 1];
        if (DIGIT_ANY_RE.test(ch) && !isInsideLatinToken(el.value, caret - 1, caret)) {
          el.setRangeText(toPersianDigits(ch), caret - 1, caret, "end");
        }
      }
    }
    onValueChange();
  });
}

/* ==================== Health & Config ==================== */

async function refreshHealth() {
  const pill = $("#health-pill");
  const label = $("#health-text");
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    pill.classList.remove("err");
    pill.classList.add("ok");
    const dev = data.gpu_name ? "GPU" : String(data.device || "cpu").toUpperCase();
    const vram = Number(data.vram_peak_mb || 0) > 0 ? ` · ${Number(data.vram_peak_mb).toFixed(0)}MB` : "";
    label.textContent = `${dev}${vram} · ${data.active_model}`;
    if (data.active_model) {
      state.model = data.active_model;
      $("#select-model").value = data.active_model;
    }
    if (data.spellcheck_mode) {
      state.spellcheck = data.spellcheck_mode;
      $("#select-spellcheck").value = data.spellcheck_mode;
    }
  } catch (err) {
    pill.classList.remove("ok");
    pill.classList.add("err");
    label.textContent = "API Offline";
  }
}

async function syncServerConfig() {
  try {
    await fetch("/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.model,
        spellcheck_mode: state.spellcheck,
        out_formats: state.minimal.formats.length ? state.minimal.formats : ["html", "md", "txt", "json"],
      }),
    });
    await refreshHealth();
  } catch (err) {
    showToast(`خطا در تغییر تنظیمات: ${err.message}`, "error");
  }
}

function setMode(mode) {
  state.mode = mode;
  const isMin = mode === "minimal";
  $("#minimal-view").classList.toggle("hidden", !isMin);
  $("#debug-view").classList.toggle("hidden", isMin);
  $("#btn-mode-minimal").classList.toggle("active", isMin);
  $("#btn-mode-debug").classList.toggle("active", !isMin);
  $("#btn-mode-minimal").setAttribute("aria-selected", String(isMin));
  $("#btn-mode-debug").setAttribute("aria-selected", String(!isMin));
  if (!isMin) {
    loadRunHistory();
    renderDebugCanvas();
  }
}

/* ==================== Recursive Drag & Drop ==================== */

async function readAllEntries(dirReader) {
  const entries = [];
  while (true) {
    const batch = await new Promise((resolve, reject) => dirReader.readEntries(resolve, reject));
    if (!batch || batch.length === 0) break;
    entries.push(...batch);
  }
  return entries;
}

async function traverseEntry(entry, outFiles) {
  if (!entry) return;
  if (entry.isFile) {
    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
    if (isSupportedFile(file)) outFiles.push(file);
  } else if (entry.isDirectory) {
    const subEntries = await readAllEntries(entry.createReader());
    for (const child of subEntries) {
      await traverseEntry(child, outFiles);
    }
  }
}

async function extractDroppedFiles(dataTransfer) {
  const items = dataTransfer?.items;
  if (items && items.length && typeof items[0].webkitGetAsEntry === "function") {
    const collected = [];
    for (const item of Array.from(items)) {
      const entry = item.webkitGetAsEntry();
      if (entry) {
        await traverseEntry(entry, collected);
      } else {
        const f = item.getAsFile();
        if (f && isSupportedFile(f)) collected.push(f);
      }
    }
    return collected;
  }
  return Array.from(dataTransfer?.files || []).filter(isSupportedFile);
}

/* ==================== Minimal Mode ==================== */

function getMinimalSelectedFormats() {
  const checked = $$('input[name="min-format"]:checked').map((el) => el.value);
  return checked.length ? checked : ["html", "md", "txt", "json"];
}

function enqueueMinimalFiles(files) {
  const valid = files.filter(isSupportedFile);
  if (!valid.length) {
    showToast("هیچ فایل PDF یا تصویر معتبری یافت نشد.", "error");
    return;
  }
  for (const file of valid) {
    const rel = file.webkitRelativePath || file.name;
    state.minimal.queue.push({
      id: `${Date.now()}_${Math.random().toString(16).slice(2)}`,
      file,
      name: rel,
      size: file.size,
      status: "queued",
      data: null,
      error: null,
    });
  }
  renderMinimalQueue();
  showToast(`${valid.length} فایل به صف اضافه شد.`, "success");
}

function renderMinimalQueue() {
  const q = state.minimal.queue;
  const panel = $("#minimal-queue-panel");
  panel.classList.toggle("hidden", q.length === 0);
  if (!q.length) return;

  const doneCount = q.filter((x) => x.status === "done" || x.status === "error").length;
  const pct = Math.round((doneCount / q.length) * 100);
  $("#minimal-queue-summary").textContent = `${q.length} فایل در صف (${doneCount} تکمیل‌شده)`;
  $("#minimal-progress-bar").style.width = `${pct}%`;
  $("#btn-run-queue").disabled = state.minimal.running || q.every((x) => x.status === "done");
  $("#btn-clear-queue").disabled = state.minimal.running;

  const list = $("#minimal-queue-list");
  list.innerHTML = q
    .map((item) => {
      const statusMap = {
        queued: ["در صف", "badge-queued"],
        running: ["در حال پردازش...", "badge-running"],
        done: ["تکمیل شد", "badge-done"],
        error: ["خطا", "badge-error"],
      };
      const [label, cls] = statusMap[item.status] || statusMap.queued;
      return `
        <li class="queue-item" title="${escapeHtml(item.error || item.name)}">
          <span class="queue-item-name">${escapeHtml(item.name)}</span>
          <span class="muted" dir="ltr">${formatBytes(item.size)}</span>
          <span class="badge ${cls}">${label}</span>
        </li>
      `;
    })
    .join("");
}

async function runSingleOcr(file, formats) {
  const form = new FormData();
  form.append("file", file, file.name);
  const params = new URLSearchParams();
  params.set("model", state.model);
  params.set("spellcheck", state.spellcheck);
  params.set("save_outputs", "true");
  for (const fmt of formats) params.append("out_format", fmt);

  const res = await fetch(`/ocr?${params.toString()}`, { method: "POST", body: form });
  const payload = await res.json();
  if (!res.ok) {
    throw new Error(payload.detail || `HTTP ${res.status}`);
  }
  return payload;
}

async function processMinimalQueue() {
  if (state.minimal.running) return;
  const formats = getMinimalSelectedFormats();
  state.minimal.formats = formats;
  state.minimal.running = true;
  renderMinimalQueue();

  for (let i = 0; i < state.minimal.queue.length; i += 1) {
    const item = state.minimal.queue[i];
    if (item.status === "done") continue;
    item.status = "running";
    $("#minimal-progress-label").textContent = `در حال پردازش ${item.file.name}...`;
    renderMinimalQueue();
    try {
      const data = await runSingleOcr(item.file, formats);
      item.status = "done";
      item.data = data;
      state.minimal.results.push({ name: item.file.name, data });
      state.minimal.activeDocIndex = state.minimal.results.length - 1;
      renderMinimalResults();
    } catch (err) {
      item.status = "error";
      item.error = err.message;
      showToast(`خطا در ${item.file.name}: ${err.message}`, "error");
    }
    renderMinimalQueue();
  }

  state.minimal.running = false;
  $("#minimal-progress-label").textContent = "پردازش صف به پایان رسید.";
  renderMinimalQueue();
  refreshHealth();
}

function renderMinimalResults() {
  const results = state.minimal.results;
  const panel = $("#minimal-results-panel");
  panel.classList.toggle("hidden", results.length === 0);
  if (!results.length) return;

  const idx = Math.max(0, Math.min(state.minimal.activeDocIndex, results.length - 1));
  state.minimal.activeDocIndex = idx;
  const active = results[idx].data;

  $("#minimal-doc-tabs").innerHTML = results
    .map(
      (r, i) => `
      <button class="doc-tab ${i === idx ? "active" : ""}" data-doc-idx="${i}" type="button">
        <span dir="ltr">${escapeHtml(r.name)}</span>
      </button>`
    )
    .join("");

  $$("#minimal-doc-tabs .doc-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.minimal.activeDocIndex = Number(btn.dataset.docIdx);
      renderMinimalResults();
    });
  });

  const urls = active.output_urls || {};
  $("#minimal-download-links").innerHTML = ["html", "md", "txt", "json"]
    .filter((fmt) => urls[fmt])
    .map(
      (fmt) =>
        `<a class="btn btn-ghost btn-sm" href="${escapeHtml(urls[fmt])}" download>${fmt.toUpperCase()}</a>`
    )
    .join("");

  const fmt = state.minimal.activeFmt;
  $$("#minimal-preview-tabs .subtab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.fmt === fmt);
  });

  const iframe = $("#minimal-html-frame");
  const pre = $("#minimal-code-preview");
  if (fmt === "html") {
    iframe.classList.remove("hidden");
    pre.classList.add("hidden");
    renderHtmlPreview(iframe, active.html || synthesizeHtml(active.result));
  } else {
    iframe.classList.add("hidden");
    pre.classList.remove("hidden");
    pre.classList.toggle("rtl-preview", fmt === "txt" || fmt === "md");
    pre.textContent = getFormattedText(active, fmt);
  }
}

function getFormattedText(payload, fmt) {
  if (fmt === "html") return payload.html || synthesizeHtml(payload.result);
  if (fmt === "md") return payload.markdown || synthesizeMarkdown(payload.result);
  if (fmt === "txt") return payload.text || synthesizePlainText(payload.result);
  return JSON.stringify(payload.result || {}, null, 2);
}

function synthesizePlainText(doc) {
  if (!doc || !Array.isArray(doc.pages)) return "";
  return doc.pages
    .map((p) => (p.blocks || []).map((b) => b.text || "").filter(Boolean).join("\n\n"))
    .filter(Boolean)
    .join("\n\n");
}

function synthesizeMarkdown(doc) {
  if (!doc || !Array.isArray(doc.pages)) return "";
  return doc.pages
    .map((p) => {
      const header = `## صفحه ${p.page_number}`;
      const body = (p.blocks || [])
        .map((b) => {
          if (b.type === "title") return `### ${b.text || ""}`;
          return b.text || "";
        })
        .filter(Boolean)
        .join("\n\n");
      return `${header}\n\n${body}`;
    })
    .join("\n\n---\n\n");
}

function buildStaticTableHtml(cells) {
  if (!Array.isArray(cells) || cells.length === 0) return "";
  const rows = new Map();
  for (const c of cells) {
    if (!rows.has(c.row_start)) rows.set(c.row_start, []);
    rows.get(c.row_start).push(c);
  }
  const sortedRowKeys = Array.from(rows.keys()).sort((a, b) => a - b);
  const trs = sortedRowKeys
    .map((rKey) => {
      const rowCells = rows.get(rKey).sort((a, b) => a.col_start - b.col_start);
      const tds = rowCells
        .map((cell) => {
          const tag = cell.is_header ? "th" : "td";
          const rs = Math.max(1, (cell.row_end || cell.row_start + 1) - cell.row_start);
          const cs = Math.max(1, (cell.col_end || cell.col_start + 1) - cell.col_start);
          const spanAttrs = `${rs > 1 ? ` rowspan="${rs}"` : ""}${cs > 1 ? ` colspan="${cs}"` : ""}`;
          return `<${tag}${spanAttrs}>${escapeHtml(cell.text || "").replace(/\n/g, "<br>")}</${tag}>`;
        })
        .join("");
      return `    <tr>${tds}</tr>`;
    })
    .join("\n");
  return `<div class="table-block" dir="rtl">\n<table dir="rtl">\n  <tbody>\n${trs}\n  </tbody>\n</table>\n</div>`;
}

function synthesizeHtml(doc) {
  const body = doc && doc.pages
    ? doc.pages
        .map((p) => {
          const ordered = [...(p.blocks || [])].sort((a, b) => (a.reading_order ?? 0) - (b.reading_order ?? 0));
          const blocksHtml = ordered
            .map((b) => {
              if (Array.isArray(b.cells) && b.cells.length > 0) {
                return buildStaticTableHtml(b.cells);
              }
              if (b.html) return b.html;
              const escaped = escapeHtml(b.text || "").replace(/\n/g, "<br>");
              if (b.type === "title") return `<h2 dir="rtl">${escaped}</h2>`;
              if (b.type === "header") return `<header dir="rtl">${escaped}</header>`;
              if (b.type === "footer") return `<footer dir="rtl">${escaped}</footer>`;
              if (b.type === "figure") {
                return `<figure dir="rtl"><figcaption>${escapeHtml(b.text || "تصویر")}</figcaption></figure>`;
              }
              return `<p dir="rtl">${escaped}</p>`;
            })
            .join("\n");
          return `<section class="page" data-page="${escapeHtml(p.page_number)}" dir="rtl" style="margin-bottom:2rem;border-bottom:1px solid #ddd;padding-bottom:1rem">\n${blocksHtml}\n</section>`;
        })
        .join("\n")
    : "";
  return `<!DOCTYPE html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><style>body{font-family:Vazirmatn,Tahoma,sans-serif;line-height:1.8;padding:1rem 1.5rem}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:6px 8px;text-align:right}th{background:#f5f5f5;font-weight:600}header,footer{color:#666;font-size:0.9em}</style></head><body>${body}</body></html>`;
}

function scopeBodySelectorCss(cssText) {
  return String(cssText || "")
    .replace(/^\s*@import\b[^;]*;?/gim, "")
    .replace(/(^|[}\s,])body(?=\s*[{,:>~+])/gi, "$1.ocr-doc-body");
}

function sanitizePreviewNode(root) {
  if (!root) return;
  root.querySelectorAll("script, iframe, object, embed, applet, base, meta, link, form").forEach((n) => n.remove());
  const dangerousUriRe = /^\s*(?:javascript|vbscript):|^\s*data:text\/html/i;
  const urlAttrs = ["href", "src", "xlink:href", "action", "formaction"];
  root.querySelectorAll("*").forEach((el) => {
    for (const attr of Array.from(el.attributes)) {
      const name = attr.name.toLowerCase();
      if (name.startsWith("on")) {
        el.removeAttribute(attr.name);
        continue;
      }
      if (urlAttrs.includes(name) && dangerousUriRe.test(attr.value || "")) {
        el.removeAttribute(attr.name);
      }
    }
  });
}

function renderHtmlPreview(hostEl, rawHtml) {
  if (!hostEl) return;
  const shadow = hostEl.shadowRoot || hostEl.attachShadow({ mode: "open" });
  const parser = new DOMParser();
  const parsed = parser.parseFromString(String(rawHtml || ""), "text/html");

  const docCssChunks = [];
  parsed.querySelectorAll("style").forEach((st) => {
    docCssChunks.push(st.textContent || "");
    st.remove();
  });
  sanitizePreviewNode(parsed.body);

  const baseStyle = document.createElement("style");
  baseStyle.textContent =
    `:host { display: block; color: #1a1a1a; direction: rtl; text-align: right; font-family: Vazirmatn, Tahoma, Arial, sans-serif; line-height: 1.8; }\n` +
    `.ocr-doc-body { margin: 0 auto; max-width: 1020px; padding: 0; }`;

  const scopedDocCss = scopeBodySelectorCss(docCssChunks.join("\n"));
  const docStyle = document.createElement("style");
  docStyle.textContent = `${scopedDocCss}\n.ocr-doc-body { margin: 0 auto !important; padding: 0 !important; }`;

  const wrapper = document.createElement("div");
  wrapper.className = "ocr-doc-body";
  wrapper.setAttribute("lang", "fa");
  wrapper.setAttribute("dir", "rtl");
  const inner = parsed.body ? parsed.body.innerHTML : "";
  wrapper.innerHTML = inner.trim()
    ? inner
    : '<p style="color:#64748b;margin:0">محتوایی برای نمایش وجود ندارد.</p>';

  shadow.replaceChildren(baseStyle, docStyle, wrapper);
}

async function downloadAllZip() {
  const files = [];
  for (const r of state.minimal.results) {
    const urls = r.data.output_urls || {};
    for (const fmt of ["html", "md", "txt", "json"]) {
      if (urls[fmt]) files.push(urls[fmt]);
    }
  }
  if (!files.length) return;
  try {
    const res = await fetch("/outputs/zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "persian-ocr-outputs.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    showToast(`خطا در ساخت ZIP: ${err.message}`, "error");
  }
}

/* ==================== Debug Mode: Loading & State ==================== */

async function loadRunHistory() {
  try {
    const res = await fetch("/runs?limit=50");
    if (!res.ok) return;
    const data = await res.json();
    state.debug.runs = data.runs || [];
    const sel = $("#select-history-run");
    const prev = sel.value;
    sel.innerHTML =
      '<option value="">— انتخاب از تاریخچه اجراها —</option>' +
      state.debug.runs
        .map((run, idx) => {
          const mdl = run.metadata?.rec_model_variant || "";
          const name = run.source_filename || run.doc_id;
          return `<option value="${idx}">${escapeHtml(name)} · ${escapeHtml(mdl)} (${run.page_count} ص)</option>`;
        })
        .join("");
    if (prev) sel.value = prev;
  } catch {
    // Non-fatal
  }
}

function loadPayloadIntoDebug(payload) {
  const doc = structuredClone(payload.result);
  state.debug.originalDoc = structuredClone(payload.result);
  state.debug.workingDoc = doc;
  state.debug.pageImages = payload.page_images || {};
  state.debug.outputUrls = payload.output_urls || {};
  state.debug.htmlText = payload.html || synthesizeHtml(doc);
  state.debug.mdText = payload.markdown || synthesizeMarkdown(doc);
  state.debug.txtText = payload.text || synthesizePlainText(doc);
  state.debug.activePageIndex = 0;
  state.debug.selectedBlockId = null;
  state.debug.dirty = false;
  state.debug.spellRuleFilter = "all";

  $("#debug-empty-state").classList.add("hidden");
  $("#debug-workbench").classList.remove("hidden");
  $("#debug-kpi-strip").classList.remove("hidden");
  renderDebugAll();
}

async function loadHistoryRunByIndex(indexStr) {
  if (indexStr === "") return;
  const run = state.debug.runs[Number(indexStr)];
  if (!run) return;
  const urls = run.output_urls || {};
  const pageUrls = run.page_image_urls || {};
  if (!urls.json) {
    showToast("فایل JSON این اجرا ذخیره نشده است.", "error");
    return;
  }
  $("#debug-status-note").textContent = "در حال بارگذاری اجرا...";
  try {
    const [jsonRes, htmlText, mdText, txtText] = await Promise.all([
      fetch(urls.json).then((r) => {
        if (!r.ok) throw new Error(`JSON HTTP ${r.status}`);
        return r.json();
      }),
      urls.html ? fetch(urls.html).then((r) => (r.ok ? r.text() : "")) : Promise.resolve(""),
      urls.md ? fetch(urls.md).then((r) => (r.ok ? r.text() : "")) : Promise.resolve(""),
      urls.txt ? fetch(urls.txt).then((r) => (r.ok ? r.text() : "")) : Promise.resolve(""),
    ]);
    loadPayloadIntoDebug({
      result: jsonRes,
      output_urls: urls,
      page_images: pageUrls,
      html: htmlText,
      markdown: mdText,
      text: txtText,
    });
    $("#debug-status-note").textContent = `بارگذاری شد: ${run.source_filename}`;
  } catch (err) {
    $("#debug-status-note").textContent = "";
    showToast(`خطا در بارگذاری اجرا: ${err.message}`, "error");
  }
}

async function handleDebugFileUpload(file) {
  if (!file || !isSupportedFile(file)) {
    showToast("فرمت فایل پشتیبانی نمی‌شود.", "error");
    return;
  }
  $("#debug-status-note").textContent = `در حال اجرای OCR و استخراج اطلاعات دیباگ برای ${file.name}...`;
  try {
    const data = await runSingleOcr(file, ["html", "md", "txt", "json"]);
    loadPayloadIntoDebug(data);
    $("#debug-status-note").textContent = `تکمیل شد: ${file.name}`;
    loadRunHistory();
    refreshHealth();
    showToast("پردازش با موفقیت انجام شد.", "success");
  } catch (err) {
    $("#debug-status-note").textContent = "";
    showToast(`خطا در پردازش: ${err.message}`, "error");
  }
}

/* ==================== Debug Mode: Rendering ==================== */

function renderDebugAll() {
  renderDebugKpis();
  renderDebugCanvas();
  renderBlockEditor();
  renderTracesReport();
  renderSpellcheckReport();
  renderDebugLiveOutputs();
  updateDirtyStateUI();
}

function renderDebugKpis() {
  const doc = state.debug.workingDoc;
  if (!doc) return;
  const meta = doc.metadata || {};
  const dm = doc.metrics || {};
  const items = [
    { label: "مدل فعال", value: `<span dir="ltr">${escapeHtml(meta.rec_model_label || meta.rec_model_variant || "—")}</span>` },
    { label: "زمان کل / میانگین", value: `<span dir="ltr">${Number(dm.total_sec || doc.processing_time_sec || 0).toFixed(2)}s / ${Number(dm.mean_page_latency_sec || 0).toFixed(2)}s</span>` },
    { label: "اوج VRAM / RAM", value: `<span dir="ltr">${dm.peak_vram_mb != null ? `${dm.peak_vram_mb.toFixed(0)} MB` : "—"} / ${dm.peak_rss_ram_mb != null ? `${dm.peak_rss_ram_mb.toFixed(0)} MB` : "—"}</span>` },
    { label: "خروجی ساختاریافته", value: `${dm.total_blocks ?? 0} بلوک · ${dm.total_lines ?? 0} خط · ${dm.total_tables ?? 0} جدول` },
    { label: "مسیر ترکیبی (CTC / VLM)", value: `<span dir="ltr">CTC=${dm.total_ctc_blocks ?? 0} / VLM=${dm.total_vlm_blocks ?? 0}</span>` },
    { label: "اصلاح املایی", value: `${dm.total_corrections ?? 0} مورد (${escapeHtml(meta.spellcheck_mode || "off")})` },
  ];
  $("#debug-kpi-strip").innerHTML = items
    .map(
      (k) =>
        `<div class="kpi-card"><div class="kpi-label">${k.label}</div><div class="kpi-value">${k.value}</div></div>`
    )
    .join("");
}

function getActiveDebugPage() {
  const doc = state.debug.workingDoc;
  if (!doc || !doc.pages || !doc.pages.length) return null;
  const idx = Math.max(0, Math.min(state.debug.activePageIndex, doc.pages.length - 1));
  state.debug.activePageIndex = idx;
  return doc.pages[idx];
}

function renderDebugCanvas() {
  const page = getActiveDebugPage();
  const doc = state.debug.workingDoc;
  if (!page || !doc) return;

  $("#page-indicator").textContent = `صفحه ${page.page_number} از ${doc.page_count}`;
  $("#btn-prev-page").disabled = state.debug.activePageIndex <= 0;
  $("#btn-next-page").disabled = state.debug.activePageIndex >= doc.pages.length - 1;

  const viewport = $("#canvas-viewport");
  const stage = $("#canvas-stage");
  const img = $("#debug-page-img");
  const svg = $("#debug-svg-overlay");

  const pageW = Math.max(1, Number(page.width || 800));
  const pageH = Math.max(1, Number(page.height || 1000));
  let scale = state.debug.zoom;
  if (scale === null) {
    const availW = Math.max(320, viewport.clientWidth - 28);
    scale = Math.min(1.0, availW / pageW);
    $("#zoom-label").textContent = "Fit";
  } else {
    $("#zoom-label").textContent = `${Math.round(scale * 100)}%`;
  }

  stage.style.width = `${Math.round(pageW * scale)}px`;
  stage.style.height = `${Math.round(pageH * scale)}px`;

  const imgUrl = state.debug.pageImages[String(page.page_number)];
  if (imgUrl) {
    img.src = imgUrl;
    img.classList.remove("hidden");
  } else {
    img.removeAttribute("src");
    img.classList.add("hidden");
  }

  svg.setAttribute("viewBox", `0 0 ${pageW} ${pageH}`);
  const svgParts = [];

  for (const blk of page.blocks || []) {
    const color = BLOCK_COLORS[blk.type] || "#475569";
    const [bx1, by1, bx2, by2] = blk.bbox || [0, 0, 0, 0];
    const bw = Math.max(1, bx2 - bx1);
    const bh = Math.max(1, by2 - by1);
    const isSel = blk.block_id === state.debug.selectedBlockId;

    if (state.debug.layers.blocks) {
      svgParts.push(
        `<rect class="svg-box ${isSel ? "selected" : ""}" data-block-id="${escapeHtml(blk.block_id)}" ` +
          `x="${bx1}" y="${by1}" width="${bw}" height="${bh}" ` +
          `fill="${color}" fill-opacity="0.10" stroke="${color}" stroke-width="2" />`
      );
    }

    if (state.debug.layers.cells && Array.isArray(blk.cells)) {
      for (const cell of blk.cells) {
        if (!cell.bbox) continue;
        const [cx1, cy1, cx2, cy2] = cell.bbox;
        const cw = Math.max(1, cx2 - cx1);
        const ch = Math.max(1, cy2 - cy1);
        if (cw <= 2 || ch <= 2) continue;
        svgParts.push(
          `<rect x="${cx1}" y="${cy1}" width="${cw}" height="${ch}" ` +
            `fill="none" stroke="#ea580c" stroke-width="1" stroke-dasharray="4 3" />`
        );
      }
    }

    if (state.debug.layers.lines && Array.isArray(blk.lines)) {
      for (const ln of blk.lines) {
        if (Array.isArray(ln.polygon) && ln.polygon.length >= 3) {
          const pts = ln.polygon.map((pt) => `${pt[0]},${pt[1]}`).join(" ");
          svgParts.push(
            `<polygon points="${pts}" fill="#0284c7" fill-opacity="0.08" stroke="#0284c7" stroke-width="1" />`
          );
        } else if (ln.bbox) {
          const [lx1, ly1, lx2, ly2] = ln.bbox;
          svgParts.push(
            `<rect x="${lx1}" y="${ly1}" width="${Math.max(1, lx2 - lx1)}" height="${Math.max(1, ly2 - ly1)}" ` +
              `fill="#0284c7" fill-opacity="0.08" stroke="#0284c7" stroke-width="1" />`
          );
        }
      }
    }

    if (state.debug.layers.labels) {
      const recShort = blk.recognizer ? (blk.recognizer.includes("Koochik") || blk.recognizer.includes("VLM") ? "VLM" : "CTC") : "";
      const tagText = `#${blk.reading_order} ${blk.type}${recShort ? ` · ${recShort}` : ""}`;
      const pillW = Math.max(66, tagText.length * 7.2);
      const pillY = Math.max(0, by1 - 20);
      svgParts.push(
        `<g pointer-events="none">` +
          `<rect x="${bx1}" y="${pillY}" width="${pillW}" height="18" rx="4" fill="${color}" />` +
          `<text x="${bx1 + 5}" y="${pillY + 13}" fill="#ffffff" font-size="11" font-family="JetBrains Mono, monospace">${escapeHtml(tagText)}</text>` +
        `</g>`
      );
    }
  }

  svg.innerHTML = svgParts.join("");

  $$("#debug-svg-overlay .svg-box").forEach((rect) => {
    rect.addEventListener("click", () => {
      selectDebugBlock(rect.dataset.blockId, true);
    });
    rect.addEventListener("mousemove", (e) => {
      const blk = (page.blocks || []).find((b) => b.block_id === rect.dataset.blockId);
      if (!blk) return;
      const tip = $("#svg-tooltip");
      tip.classList.remove("hidden");
      tip.style.left = `${e.clientX + 14}px`;
      tip.style.top = `${e.clientY + 14}px`;
      tip.innerHTML =
        `<div><strong>${escapeHtml(blk.block_id)}</strong> (${escapeHtml(blk.type)})</div>` +
        `<div dir="ltr">rec: ${escapeHtml(blk.recognizer || "—")} · ${escapeHtml(blk.routing_reason || "—")}</div>` +
        `<div>اطمینان چیدمان: ${Number(blk.confidence || 0).toFixed(2)}</div>`;
    });
    rect.addEventListener("mouseleave", () => {
      $("#svg-tooltip").classList.add("hidden");
    });
  });
}

function selectDebugBlock(blockId, scrollToCard = false) {
  state.debug.selectedBlockId = blockId;
  $$("#debug-svg-overlay .svg-box").forEach((el) => {
    el.classList.toggle("selected", el.dataset.blockId === blockId);
  });
  $$("#block-editor-list .block-card").forEach((el) => {
    const match = el.dataset.blockId === blockId;
    el.classList.toggle("selected", match);
    if (match && scrollToCard) {
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  });
}

function buildEditableTableHtml(blk) {
  const cells = blk.cells || [];
  if (!cells.length) return "";
  const rows = new Map();
  for (let i = 0; i < cells.length; i += 1) {
    const c = cells[i];
    if (!rows.has(c.row_start)) rows.set(c.row_start, []);
    rows.get(c.row_start).push({ cell: c, idx: i });
  }
  const sortedRowKeys = Array.from(rows.keys()).sort((a, b) => a - b);
  const trs = sortedRowKeys
    .map((rKey) => {
      const items = rows.get(rKey).sort((a, b) => a.cell.col_start - b.cell.col_start);
      const tds = items
        .map(({ cell, idx }) => {
          const tag = cell.is_header ? "th" : "td";
          const rs = Math.max(1, (cell.row_end || cell.row_start + 1) - cell.row_start);
          const cs = Math.max(1, (cell.col_end || cell.col_start + 1) - cell.col_start);
          const spanAttrs = `${rs > 1 ? ` rowspan="${rs}"` : ""}${cs > 1 ? ` colspan="${cs}"` : ""}`;
          return `<${tag}${spanAttrs}><input type="text" dir="rtl" lang="fa" class="cell-input" data-block-id="${escapeHtml(blk.block_id)}" data-cell-idx="${idx}" value="${escapeHtml(cell.text || "")}"></${tag}>`;
        })
        .join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");
  return `<div class="editable-table-wrap"><table class="editable-table"><tbody>${trs}</tbody></table></div>`;
}

function renderBlockEditor() {
  const page = getActiveDebugPage();
  const list = $("#block-editor-list");
  if (!page) {
    list.innerHTML = "";
    return;
  }

  const urls = state.debug.outputUrls || {};
  $("#debug-download-bar").innerHTML =
    `<span class="muted">دانلود فایل‌های ذخیره‌شده:</span>` +
    ["html", "md", "txt", "json"]
      .filter((fmt) => urls[fmt])
      .map((fmt) => `<a class="btn btn-ghost btn-sm" href="${escapeHtml(urls[fmt])}" download>${fmt.toUpperCase()}</a>`)
      .join("");

  const filter = state.debug.blockFilter;
  const blocks = (page.blocks || []).filter((blk) => {
    if (filter === "table") return blk.type === "table";
    if (filter === "ctc") return blk.recognizer && !blk.recognizer.includes("Koochik") && !blk.recognizer.includes("VLM");
    if (filter === "vlm") return blk.recognizer && (blk.recognizer.includes("Koochik") || blk.recognizer.includes("VLM"));
    return true;
  });

  if (!blocks.length) {
    list.innerHTML = '<div class="muted">بلوکی مطابق با فیلتر انتخاب‌شده در این صفحه وجود ندارد.</div>';
    return;
  }

  list.innerHTML = blocks
    .map((blk) => {
      const isVlm = blk.recognizer && (blk.recognizer.includes("Koochik") || blk.recognizer.includes("VLM"));
      const recCls = isVlm ? "badge-vlm" : "badge-ctc";
      const lineConfs = (blk.lines || []).map((l) => Number(l.confidence || 0));
      const meanConf = lineConfs.length
        ? (lineConfs.reduce((a, b) => a + b, 0) / lineConfs.length).toFixed(3)
        : Number(blk.confidence || 0).toFixed(3);
      const isSel = blk.block_id === state.debug.selectedBlockId;

      const editorBody =
        Array.isArray(blk.cells) && blk.cells.length > 0
          ? buildEditableTableHtml(blk)
          : `<textarea dir="rtl" lang="fa" class="block-textarea" data-block-id="${escapeHtml(blk.block_id)}">${escapeHtml(blk.text || "")}</textarea>`;

      return `
        <article class="block-card ${isSel ? "selected" : ""}" data-block-id="${escapeHtml(blk.block_id)}">
          <div class="block-card-head">
            <div class="block-badges">
              <span class="type-pill type-${escapeHtml(blk.type)}">${escapeHtml(blk.type)}</span>
              <span class="mono-chip">${escapeHtml(blk.block_id)} · #${blk.reading_order}</span>
              ${blk.recognizer ? `<span class="badge ${recCls}" dir="ltr">${escapeHtml(blk.recognizer)}</span>` : ""}
              ${blk.routing_reason ? `<span class="mono-chip" dir="ltr">${escapeHtml(blk.routing_reason)}</span>` : ""}
            </div>
            <div class="mono-chip" dir="ltr">
              conf=${meanConf} · lines=${(blk.lines || []).length}${blk.cells?.length ? ` · cells=${blk.cells.length}` : ""}
            </div>
          </div>
          ${editorBody}
        </article>
      `;
    })
    .join("");

  $$("#block-editor-list .block-card").forEach((card) => {
    card.addEventListener("click", () => {
      selectDebugBlock(card.dataset.blockId, false);
    });
  });

  $$("#block-editor-list .block-textarea").forEach((ta) => {
    attachPersianEditorBehavior(ta, () => {
      const blk = (page.blocks || []).find((b) => b.block_id === ta.dataset.blockId);
      if (!blk) return;
      blk.text = ta.value;
      markDebugDirty();
    });
  });

  $$("#block-editor-list .cell-input").forEach((inp) => {
    attachPersianEditorBehavior(inp, () => {
      const blk = (page.blocks || []).find((b) => b.block_id === inp.dataset.blockId);
      if (!blk || !blk.cells) return;
      const cellIdx = Number(inp.dataset.cellIdx);
      if (blk.cells[cellIdx]) {
        blk.cells[cellIdx].text = inp.value;
        markDebugDirty();
      }
    });
  });
}

function markDebugDirty() {
  state.debug.dirty = true;
  updateDirtyStateUI();
}

function updateDirtyStateUI() {
  const badge = $("#edit-dirty-badge");
  const revertBtn = $("#btn-revert-edits");
  const saveBtn = $("#btn-save-edits");
  if (state.debug.dirty) {
    badge.textContent = "تغییرات ذخیره‌نشده";
    badge.className = "dirty-badge dirty";
    revertBtn.disabled = false;
    saveBtn.disabled = false;
  } else {
    badge.textContent = "ذخیره‌شده / بدون تغییر";
    badge.className = "dirty-badge clean";
    revertBtn.disabled = true;
    saveBtn.disabled = !state.debug.workingDoc;
  }
}

async function saveDebugEdits() {
  if (!state.debug.workingDoc) return;
  const saveBtn = $("#btn-save-edits");
  saveBtn.disabled = true;
  try {
    const res = await fetch("/ocr/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document: state.debug.workingDoc,
        formats: ["html", "md", "txt", "json"],
        page_images: state.debug.pageImages,
      }),
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
    loadPayloadIntoDebug(payload);
    loadRunHistory();
    showToast("تغییرات در HTML / MD / TXT / JSON ذخیره شد.", "success");
  } catch (err) {
    saveBtn.disabled = false;
    showToast(`خطا در ذخیره‌سازی: ${err.message}`, "error");
  }
}

function revertDebugEdits() {
  if (!state.debug.originalDoc) return;
  state.debug.workingDoc = structuredClone(state.debug.originalDoc);
  state.debug.dirty = false;
  renderDebugAll();
  showToast("تغییرات به نسخه قبل بازگردانده شد.", "info");
}

function renderTracesReport() {
  const doc = state.debug.workingDoc;
  const wrap = $("#traces-content");
  if (!doc) {
    wrap.innerHTML = "";
    return;
  }
  const st = doc.metrics?.stage_totals || {};
  const maxStage = Math.max(0.01, ...STAGE_LABELS.map((s) => Number(st[s.key] || 0)));

  const stageHtml = STAGE_LABELS.map((s) => {
    const val = Number(st[s.key] || 0);
    const pct = Math.min(100, (val / maxStage) * 100);
    return `
      <div class="stage-row">
        <span>${s.label}</span>
        <div class="stage-track"><div class="stage-fill" style="width:${pct.toFixed(1)}%;background:${s.color}"></div></div>
        <span dir="ltr">${val.toFixed(3)}s</span>
      </div>`;
  }).join("");

  const pageRows = (doc.pages || [])
    .map((p) => {
      const m = p.metrics || {};
      const pst = m.stages || {};
      return `
        <tr>
          <td>صفحه ${p.page_number}</td>
          <td dir="ltr">${Number(m.latency_sec || 0).toFixed(2)}s</td>
          <td dir="ltr">${Number(pst.layout_sec || 0).toFixed(2)}s</td>
          <td dir="ltr">${Number(pst.detect_sec || 0).toFixed(2)}s</td>
          <td dir="ltr">${Number(pst.table_sec || 0).toFixed(2)}s</td>
          <td dir="ltr">${Number(pst.ctc_sec || 0).toFixed(2)}s / ${Number(pst.vlm_sec || 0).toFixed(2)}s</td>
          <td dir="ltr">${m.ctc_blocks ?? 0} / ${m.vlm_blocks ?? 0}</td>
          <td dir="ltr">${m.peak_vram_mb != null ? `${m.peak_vram_mb.toFixed(0)} MB` : "—"}</td>
        </tr>`;
    })
    .join("");

  const meta = doc.metadata || {};
  const metaPairs = [
    ["شناسه سند", meta.rec_model_variant ? `${doc.doc_id}` : doc.doc_id],
    ["مدل تشخیص", meta.rec_model_label || meta.rec_model_variant],
    ["گرانولاریتی", meta.granularity],
    ["مدل چیدمان", meta.layout_model],
    ["تشخیص خطوط", meta.det_model || "—"],
    ["مدل جدول", meta.table_model || "—"],
    ["سخت‌افزار", meta.gpu_name ? `${meta.device} (${meta.gpu_name})` : meta.device],
    ["DPI / سیاست ارقام", `${meta.dpi || 200} / ${meta.digit_policy || "persian"}`],
  ];

  wrap.innerHTML = `
    <div>
      <h4>تفکیک زمان مراحل کل سند</h4>
      <div class="stage-bars">${stageHtml}</div>
    </div>
    <div>
      <h4>گزارش ردیابی صفحه‌به‌صفحه</h4>
      <table class="audit-table">
        <thead>
          <tr><th>صفحه</th><th>زمان کل</th><th>چیدمان</th><th>خطوط</th><th>جدول</th><th>CTC / VLM</th><th>بلوک‌ها</th><th>اوج VRAM</th></tr>
        </thead>
        <tbody>${pageRows}</tbody>
      </table>
    </div>
    <div>
      <h4>شناسنامه اجرایی (Provenance)</h4>
      <div class="meta-grid">
        ${metaPairs
          .map(
            ([k, v]) =>
              `<div class="meta-item"><div class="muted">${k}</div><div dir="ltr"><strong>${escapeHtml(v || "—")}</strong></div></div>`
          )
          .join("")}
      </div>
    </div>`;
}

function renderSpellcheckReport() {
  const doc = state.debug.workingDoc;
  const wrap = $("#spellcheck-content");
  if (!doc) {
    wrap.innerHTML = "";
    return;
  }
  const corrs = doc.metrics?.corrections || [];
  const total = doc.metrics?.total_corrections ?? corrs.length;

  const counts = {};
  for (const c of corrs) {
    counts[c.rule] = (counts[c.rule] || 0) + 1;
  }

  const ruleButtons = ["all", ...Object.keys(counts)]
    .map((r) => {
      const lbl = r === "all" ? `همه (${total})` : `${r} (${counts[r]})`;
      return `<button class="rule-chip ${state.debug.spellRuleFilter === r ? "active" : ""}" data-rule="${escapeHtml(r)}" type="button">${escapeHtml(lbl)}</button>`;
    })
    .join("");

  const filtered =
    state.debug.spellRuleFilter === "all"
      ? corrs
      : corrs.filter((c) => c.rule === state.debug.spellRuleFilter);

  const rows = filtered.length
    ? filtered
        .map(
          (c) => `
        <tr>
          <td>${c.page_number}</td>
          <td class="corr-orig">${escapeHtml(c.original)}</td>
          <td class="corr-fixed">${escapeHtml(c.corrected)}</td>
          <td dir="ltr">${escapeHtml(c.rule)}</td>
          <td dir="ltr">${Number(c.cost || 0).toFixed(2)}</td>
        </tr>`
        )
        .join("")
    : '<tr><td colspan="5" class="muted">موردی برای نمایش وجود ندارد.</td></tr>';

  wrap.innerHTML = `
    <div class="rule-chips">${ruleButtons}</div>
    <table class="audit-table">
      <thead>
        <tr><th>صفحه</th><th>متن اولیه</th><th>متن اصلاح‌شده</th><th>قانون</th><th>هزینه</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;

  $$("#spellcheck-content .rule-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.debug.spellRuleFilter = btn.dataset.rule;
      renderSpellcheckReport();
    });
  });
}

function renderDebugLiveOutputs() {
  const doc = state.debug.workingDoc;
  if (!doc) return;
  const fmt = state.debug.activeLiveFmt;
  $$("#debug-output-subtabs .subtab").forEach((b) => {
    b.classList.toggle("active", b.dataset.fmt === fmt);
  });

  const iframe = $("#debug-live-html");
  const pre = $("#debug-live-pre");
  if (fmt === "html") {
    iframe.classList.remove("hidden");
    pre.classList.add("hidden");
    renderHtmlPreview(iframe, state.debug.dirty ? synthesizeHtml(doc) : state.debug.htmlText || synthesizeHtml(doc));
  } else {
    iframe.classList.add("hidden");
    pre.classList.remove("hidden");
    pre.classList.toggle("rtl-preview", fmt === "txt" || fmt === "md");
    if (fmt === "md") pre.textContent = state.debug.dirty ? synthesizeMarkdown(doc) : state.debug.mdText || synthesizeMarkdown(doc);
    else if (fmt === "txt") pre.textContent = state.debug.dirty ? synthesizePlainText(doc) : state.debug.txtText || synthesizePlainText(doc);
    else pre.textContent = JSON.stringify(doc, null, 2);
  }
}

/* ==================== Event Wiring ==================== */

function bindEvents() {
  $("#btn-mode-minimal").addEventListener("click", () => setMode("minimal"));
  $("#btn-mode-debug").addEventListener("click", () => setMode("debug"));

  $("#select-model").addEventListener("change", (e) => {
    state.model = e.target.value;
    syncServerConfig();
  });
  $("#select-spellcheck").addEventListener("change", (e) => {
    state.spellcheck = e.target.value;
    syncServerConfig();
  });

  // Minimal upload controls
  $("#btn-pick-files").addEventListener("click", () => $("#input-files").click());
  $("#btn-pick-folder").addEventListener("click", () => $("#input-folder").click());
  $("#input-files").addEventListener("change", (e) => {
    enqueueMinimalFiles(Array.from(e.target.files || []));
    e.target.value = "";
  });
  $("#input-folder").addEventListener("change", (e) => {
    enqueueMinimalFiles(Array.from(e.target.files || []));
    e.target.value = "";
  });

  const dropzone = $("#minimal-dropzone");
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    const files = await extractDroppedFiles(e.dataTransfer);
    enqueueMinimalFiles(files);
  });

  $("#btn-run-queue").addEventListener("click", processMinimalQueue);
  $("#btn-clear-queue").addEventListener("click", () => {
    state.minimal.queue = [];
    state.minimal.results = [];
    renderMinimalQueue();
    renderMinimalResults();
  });

  $$("#minimal-preview-tabs .subtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.minimal.activeFmt = btn.dataset.fmt;
      renderMinimalResults();
    });
  });

  $("#btn-copy-active").addEventListener("click", async () => {
    const active = state.minimal.results[state.minimal.activeDocIndex]?.data;
    if (!active) return;
    try {
      await navigator.clipboard.writeText(getFormattedText(active, state.minimal.activeFmt));
      showToast("خروجی در کلیپ‌بورد کپی شد.", "success");
    } catch {
      showToast("امکان دسترسی به کلیپ‌بورد وجود ندارد.", "error");
    }
  });

  $("#btn-download-zip").addEventListener("click", downloadAllZip);

  $("#btn-open-in-debug").addEventListener("click", () => {
    const active = state.minimal.results[state.minimal.activeDocIndex]?.data;
    if (!active) return;
    loadPayloadIntoDebug(active);
    setMode("debug");
  });

  // Debug controls
  $("#btn-debug-upload").addEventListener("click", () => $("#input-debug-file").click());
  $("#input-debug-file").addEventListener("change", (e) => {
    const file = (e.target.files || [])[0];
    e.target.value = "";
    if (file) handleDebugFileUpload(file);
  });
  $("#select-history-run").addEventListener("change", (e) => loadHistoryRunByIndex(e.target.value));
  $("#btn-refresh-runs").addEventListener("click", loadRunHistory);

  $("#btn-prev-page").addEventListener("click", () => {
    if (state.debug.activePageIndex > 0) {
      state.debug.activePageIndex -= 1;
      state.debug.selectedBlockId = null;
      renderDebugCanvas();
      renderBlockEditor();
    }
  });
  $("#btn-next-page").addEventListener("click", () => {
    const total = state.debug.workingDoc?.pages?.length || 0;
    if (state.debug.activePageIndex < total - 1) {
      state.debug.activePageIndex += 1;
      state.debug.selectedBlockId = null;
      renderDebugCanvas();
      renderBlockEditor();
    }
  });

  $("#toggle-blocks").addEventListener("change", (e) => {
    state.debug.layers.blocks = e.target.checked;
    renderDebugCanvas();
  });
  $("#toggle-lines").addEventListener("change", (e) => {
    state.debug.layers.lines = e.target.checked;
    renderDebugCanvas();
  });
  $("#toggle-cells").addEventListener("change", (e) => {
    state.debug.layers.cells = e.target.checked;
    renderDebugCanvas();
  });
  $("#toggle-labels").addEventListener("change", (e) => {
    state.debug.layers.labels = e.target.checked;
    renderDebugCanvas();
  });

  $("#btn-zoom-in").addEventListener("click", () => {
    const cur = state.debug.zoom ?? 0.6;
    state.debug.zoom = Math.min(2.5, Number((cur + 0.15).toFixed(2)));
    renderDebugCanvas();
  });
  $("#btn-zoom-out").addEventListener("click", () => {
    const cur = state.debug.zoom ?? 0.6;
    state.debug.zoom = Math.max(0.25, Number((cur - 0.15).toFixed(2)));
    renderDebugCanvas();
  });
  $("#btn-zoom-fit").addEventListener("click", () => {
    state.debug.zoom = null;
    renderDebugCanvas();
  });

  $$("#debug-tab-buttons .subtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      state.debug.activeTab = tab;
      $$("#debug-tab-buttons .subtab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
      ["editor", "traces", "spellcheck", "outputs"].forEach((t) => {
        $(`#tab-${t}`).classList.toggle("hidden", t !== tab);
      });
      if (tab === "outputs") renderDebugLiveOutputs();
    });
  });

  $$("#debug-output-subtabs .subtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.debug.activeLiveFmt = btn.dataset.fmt;
      renderDebugLiveOutputs();
    });
  });

  $("#block-filter-select").addEventListener("change", (e) => {
    state.debug.blockFilter = e.target.value;
    renderBlockEditor();
  });

  $("#toggle-persian-digits")?.addEventListener("change", (e) => {
    state.debug.autoPersianDigits = e.target.checked;
  });

  $("#btn-save-edits").addEventListener("click", saveDebugEdits);
  $("#btn-revert-edits").addEventListener("click", revertDebugEdits);

  window.addEventListener("resize", () => {
    if (state.mode === "debug" && state.debug.zoom === null) renderDebugCanvas();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  refreshHealth();
});

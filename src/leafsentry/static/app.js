"use strict";

const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const inspectButton = document.querySelector("#inspect-button");
const resetButton = document.querySelector("#reset-button");
const previewShell = document.querySelector("#preview-shell");
const previewImage = document.querySelector("#preview-image");
const previewName = document.querySelector("#preview-name");
const previewSize = document.querySelector("#preview-size");
const message = document.querySelector("#message");
const result = document.querySelector("#result");
let previewUrl = null;

function setText(selector, value) {
  const element = document.querySelector(selector);
  if (element) element.textContent = value;
}

function percent(value) {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

function humanLabel(value) {
  return value.replaceAll("_", " ");
}

function clearChildren(element) {
  while (element.firstChild) element.removeChild(element.firstChild);
}

function setFile(file) {
  if (!file) return;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewName.textContent = file.name || "Selected image";
  previewSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
  previewShell.hidden = false;
  dropZone.hidden = true;
  resetButton.hidden = false;
  inspectButton.disabled = false;
  message.textContent = "";
  result.hidden = true;
}

function resetConsole() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  form.reset();
  previewImage.removeAttribute("src");
  previewShell.hidden = true;
  dropZone.hidden = false;
  resetButton.hidden = true;
  inspectButton.disabled = true;
  inspectButton.removeAttribute("aria-busy");
  message.textContent = "";
  result.hidden = true;
  fileInput.focus();
}

function renderScores(scores) {
  const list = document.querySelector("#score-list");
  clearChildren(list);
  scores.forEach((score) => {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const track = document.createElement("span");
    const fill = document.createElement("span");
    const value = document.createElement("span");
    row.className = "score-row";
    label.className = "score-label";
    track.className = "score-track";
    fill.className = "score-fill";
    value.className = "score-value";
    label.textContent = humanLabel(score.label);
    fill.style.width = `${Math.max(0, Math.min(100, score.probability * 100))}%`;
    value.textContent = percent(score.probability);
    track.appendChild(fill);
    row.append(label, track, value);
    list.appendChild(row);
  });
}

function appendMetric(list, name, value) {
  const row = document.createElement("div");
  const term = document.createElement("dt");
  const detail = document.createElement("dd");
  term.textContent = name;
  detail.textContent = value;
  row.append(term, detail);
  list.appendChild(row);
}

function renderResult(payload) {
  const accepted = payload.decision === "accepted";
  const badge = document.querySelector("#decision-badge");
  badge.textContent = payload.decision;
  badge.classList.toggle("abstained", !accepted);
  setText("#result-label", accepted ? humanLabel(payload.prediction) : "Needs review");
  setText(
    "#result-reason",
    accepted
      ? "Prediction cleared both confidence and class-separation thresholds."
      : `The system abstained: ${payload.abstention_reasons.map(humanLabel).join(", ")}.`
  );
  setText("#confidence-value", percent(payload.confidence));
  setText("#margin-value", percent(payload.margin));
  setText("#entropy-value", percent(payload.normalized_entropy));
  setText("#latency-value", `${payload.latency_ms.toFixed(1)} ms`);
  renderScores(payload.scores);

  const findings = payload.quality.findings;
  setText(
    "#quality-summary",
    payload.quality.passed ? "Input cleared all image-quality gates." : `Findings: ${findings.map(humanLabel).join(", ")}.`
  );
  const qualityList = document.querySelector("#quality-metrics");
  clearChildren(qualityList);
  const metrics = payload.quality.metrics;
  appendMetric(qualityList, "Dimensions", `${metrics.width} × ${metrics.height}`);
  appendMetric(qualityList, "Format", metrics.format);
  appendMetric(qualityList, "Brightness", metrics.mean_brightness.toFixed(1));
  appendMetric(qualityList, "Contrast", metrics.contrast.toFixed(1));
  appendMetric(qualityList, "Sharpness", metrics.sharpness.toFixed(1));

  setText("#request-id", payload.request_id);
  setText("#model-revision", payload.model.revision);
  setText(
    "#policy-summary",
    `confidence ≥ ${payload.policy.min_confidence.toFixed(2)} · margin ≥ ${payload.policy.min_margin.toFixed(2)} · temperature ${payload.policy.temperature.toFixed(2)}`
  );
  result.hidden = false;
  result.focus({ preventScroll: true });
  result.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function checkReadiness() {
  const state = document.querySelector("#model-state");
  try {
    const response = await fetch("/health/ready", { headers: { Accept: "application/json" } });
    const payload = await response.json();
    state.classList.toggle("ready", response.ok);
    state.classList.toggle("unavailable", !response.ok);
    setText("#model-state-text", response.ok ? "Model ready" : "Model loads on first request");
    if (payload.model && payload.model.revision) state.title = `Revision ${payload.model.revision}`;
  } catch (_error) {
    state.classList.add("unavailable");
    setText("#model-state-text", "API unavailable");
  }
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
resetButton.addEventListener("click", resetConsole);
["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});
dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  setFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  inspectButton.disabled = true;
  inspectButton.setAttribute("aria-busy", "true");
  message.textContent = "Inspecting image quality and model uncertainty…";
  result.hidden = true;
  try {
    const data = new FormData();
    data.append("file", file);
    const response = await fetch("/v1/predictions", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "Inspection failed.");
    message.textContent = "";
    renderResult(payload);
  } catch (error) {
    message.textContent = error instanceof Error ? error.message : "Inspection failed.";
  } finally {
    inspectButton.disabled = false;
    inspectButton.removeAttribute("aria-busy");
  }
});

checkReadiness();

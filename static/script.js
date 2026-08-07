let videos = [];
let selectedVideo = null;
let tapeWidthPx = null;
let calibClickPoints = [];
let previewImageObj = null;

const videoListEl = document.getElementById("video-list");
const goBtn = document.getElementById("go-btn");
const spinner = document.getElementById("spinner");
const statusLine = document.getElementById("status-line");
const previewBlock = document.getElementById("preview-block");
const previewCanvas = document.getElementById("preview-canvas");
const orientationBlock = document.getElementById("orientation-block");
const calibrationBlock = document.getElementById("calibration-block");
const calibReadout = document.getElementById("calib-readout");
const beltSpeedBlock = document.getElementById("belt-speed-block");
const resultsEl = document.getElementById("results");

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    const text = await res.text();
    const err = new Error(`Server returned a non-JSON response (HTTP ${res.status}). ${text.slice(0, 200)}`);
    err.status = res.status;
    throw err;
  }
  const data = await res.json();
  if (!res.ok) {
    const err = new Error(data.message || data.error || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

async function loadVideos() {
  try {
    const data = await fetchJson("/api/videos");
    videos = data.videos;
    renderVideoList();
  } catch (err) {
    videoListEl.innerHTML = `<div class="video-item">Could not load video list: ${err.message}</div>`;
  }
}

function renderVideoList() {
  videoListEl.innerHTML = "";
  if (videos.length === 0) {
    videoListEl.innerHTML = '<div class="video-item">No videos yet. Upload one above.</div>';
    return;
  }
  videos.forEach((v) => {
    const item = document.createElement("div");
    item.className = "video-item" + (v.name === selectedVideo ? " selected" : "");
    let tag = "";
    if (v.has_own_reference) tag = '<span class="ref-tag">reference</span>';
    else if (v.auto_calibrated) tag = '<span class="ref-tag">auto-cal</span>';
    else if (v.uploaded) tag = '<span class="ref-tag">uploaded</span>';
    item.innerHTML = `<span>${v.name}</span>${tag}`;
    item.onclick = () => selectVideo(v);
    videoListEl.appendChild(item);
  });
}

function resetCalibCanvas() {
  calibClickPoints = [];
  tapeWidthPx = null;
  const ctx = previewCanvas.getContext("2d");
  if (previewImageObj) {
    ctx.drawImage(previewImageObj, 0, 0);
  } else {
    // No frame loaded (yet, or it failed) -- clear the canvas rather
    // than leaving whatever was drawn on it before (this used to leave
    // a stale image, or a scribble of old click marks, on screen).
    ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  }
  calibReadout.textContent = previewImageObj
    ? "Click two points on the frame above"
    : "Waiting for the frame to load...";
}

function onPreviewCanvasClick(evt) {
  if (calibrationBlock.classList.contains("hidden")) return;
  if (!previewImageObj) return; // nothing to measure against yet
  const rect = previewCanvas.getBoundingClientRect();
  const scaleX = previewCanvas.width / rect.width;
  const scaleY = previewCanvas.height / rect.height;
  const x = (evt.clientX - rect.left) * scaleX;
  const y = (evt.clientY - rect.top) * scaleY;

  if (calibClickPoints.length >= 2) {
    resetCalibCanvas();
  }
  calibClickPoints.push({ x, y });

  const ctx = previewCanvas.getContext("2d");
  ctx.fillStyle = "#ff3355";
  ctx.beginPath();
  ctx.arc(x, y, Math.max(2, previewCanvas.width * 0.006), 0, 2 * Math.PI);
  ctx.fill();

  if (calibClickPoints.length === 2) {
    const [p1, p2] = calibClickPoints;
    ctx.strokeStyle = "#ff3355";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    tapeWidthPx = Math.hypot(p2.x - p1.x, p2.y - p1.y);
    calibReadout.textContent = `Tape width: ${tapeWidthPx.toFixed(1)} px`;
  }
}

document.getElementById("calib-reset-btn").onclick = resetCalibCanvas;
previewCanvas.onclick = onPreviewCanvasClick;

async function loadPreviewInto(videoName) {
  // Fetch (not a plain <img src=...>) so a failure carries the real
  // HTTP status and the server's actual error message -- an <img>'s
  // onerror event fires the same way for "file missing" as it does for
  // "this codec isn't supported here", which used to make every preview
  // failure show the same misleading "may no longer be on the server"
  // message even when the file was right there, just unreadable.
  const res = await fetch(`/api/preview/${encodeURIComponent(videoName)}?t=${Date.now()}`);
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    if ((res.headers.get("content-type") || "").includes("application/json")) {
      const data = await res.json();
      message = data.message || data.error || message;
    }
    const err = new Error(message);
    err.status = res.status;
    throw err;
  }

  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        previewImageObj = img;
        previewCanvas.width = img.naturalWidth;
        previewCanvas.height = img.naturalHeight;
        previewCanvas.getContext("2d").drawImage(img, 0, 0);
        resolve();
      };
      img.onerror = () => reject(new Error("Could not decode the preview image"));
      img.src = objectUrl;
    });
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function selectVideo(v) {
  selectedVideo = v.name;
  renderVideoList();
  goBtn.disabled = false;
  statusLine.textContent = "";
  calibClickPoints = [];
  tapeWidthPx = null;
  previewImageObj = null;

  // Preview + Nose Direction are always shown (not just when a video has
  // no detected orientation): even a correctly auto-detected direction
  // can be wrong for an unusual video, so it needs to stay visible and
  // changeable, not just appear when something's missing.
  previewBlock.classList.remove("hidden");
  try {
    await loadPreviewInto(v.name);
  } catch (e) {
    previewCanvas.getContext("2d").clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    goBtn.disabled = true;
    if (e.status === 404) {
      // File genuinely gone (free-tier storage resets on restart) --
      // refresh the list so the stale entry disappears.
      statusLine.textContent = "This video is no longer on the server (storage resets on restart/redeploy). Refreshing the video list; please re-upload it.";
      await loadVideos();
    } else if (e.status === 504) {
      // The file exists but this server's OpenCV/FFmpeg build can't
      // read it -- re-uploading the same file won't help.
      statusLine.textContent = "This video's format could not be read on the server (timed out) -- it likely uses a codec that isn't supported here. Try a different export of this video, or a different video.";
    } else {
      statusLine.textContent = `Could not load this video's first frame (${e.message}).`;
    }
    return;
  }

  orientationBlock.classList.remove("hidden");
  document.getElementById("nose-direction-select").value = v.detected_nose_direction || "up";

  calibrationBlock.classList.toggle("hidden", !v.needs_calibration);
  if (v.needs_calibration) resetCalibCanvas();

  beltSpeedBlock.classList.toggle("hidden", false);
  // Belt speed block always available as an override, but only really
  // needed if the filename doesn't carry it -- harmless to show either way.

  if (v.has_own_reference) {
    statusLine.textContent = "Reference data found. Calibration, belt speed, and orientation load automatically.";
  } else if (v.auto_calibrated) {
    statusLine.textContent = "Using calibration/orientation shared with other videos in this session folder.";
  } else {
    statusLine.textContent = "No reference data for this video. Set nose direction and/or calibration above.";
  }
}

function renderTable(rows) {
  if (!rows || rows.length === 0) return "<em>No data</em>";
  const cols = Object.keys(rows[0]);
  let html = "<table><thead><tr>";
  cols.forEach((c) => (html += `<th>${c}</th>`));
  html += "</tr></thead><tbody>";
  rows.forEach((row) => {
    html += "<tr>";
    cols.forEach((c) => {
      let val = row[c];
      if (typeof val === "number") val = Number.isInteger(val) ? val : val.toFixed(3);
      html += `<td>${val === null || val === undefined ? "" : val}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  return html;
}

function renderResults(data) {
  document.getElementById("results-info").innerHTML =
    `<strong>${data.video}</strong> &nbsp;|&nbsp; fps: ${data.fps} &nbsp;|&nbsp; ` +
    `belt speed: ${data.belt_speed_cms} cm/s &nbsp;|&nbsp; ` +
    `calibration: ${data.cm_per_pixel_x} cm/px (x), ${data.cm_per_pixel_y} cm/px (y) &nbsp;|&nbsp; ` +
    (data.used_reference_data ? "using DigiGait reference data" :
      data.used_session_default ? "using session-shared calibration/orientation" : "manual setup");

  document.getElementById("snapshot-img").src = data.snapshot_url + `?t=${Date.now()}`;
  document.getElementById("clip-img").src = data.clip_url + `?t=${Date.now()}`;
  document.getElementById("gait-signals-img").src = data.gait_signals_url + `?t=${Date.now()}`;
  document.getElementById("ensemble-paws-img").src = data.ensemble_paws_url + `?t=${Date.now()}`;
  document.getElementById("posture-img").src = data.posture_url + `?t=${Date.now()}`;

  document.getElementById("summary-table-wrap").innerHTML = renderTable(data.summary_rows);
  document.getElementById("download-csv").href = data.summary_csv_url;

  const comparisonPanel = document.getElementById("comparison-panel");
  if (data.comparison_rows && data.comparison_rows.length > 0) {
    document.getElementById("comparison-table-wrap").innerHTML = renderTable(data.comparison_rows);
    comparisonPanel.classList.remove("hidden");
  } else {
    comparisonPanel.classList.add("hidden");
  }

  resultsEl.classList.remove("hidden");
  resultsEl.scrollIntoView({ behavior: "smooth" });
}

const POLL_INTERVAL_MS = 2500;
const MAX_WAIT_MS = 10 * 60 * 1000; // 10 minutes -- real videos on a free-tier host can be slow

async function pollJob(jobId) {
  const startTime = Date.now();

  while (true) {
    const elapsedSec = Math.round((Date.now() - startTime) / 1000);

    if (Date.now() - startTime > MAX_WAIT_MS) {
      statusLine.textContent = `Gave up waiting after ${elapsedSec}s. The server may be overloaded, try again or try a smaller video.`;
      return;
    }

    let job;
    try {
      job = await fetchJson(`/api/run/${jobId}`);
    } catch (err) {
      statusLine.textContent = "Error checking status: " + err.message;
      return;
    }

    if (job.status === "running") {
      statusLine.textContent = `Running analysis... ${elapsedSec}s elapsed (this can take a while on a free-tier server)`;
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      continue;
    }
    if (job.status === "done") {
      renderResults(job.result);
      statusLine.textContent = `Done (${elapsedSec}s).`;
      return;
    }
    // "input_needed" or "error"
    statusLine.textContent = "Error: " + (job.message || job.error || "unknown error");
    return;
  }
}

goBtn.onclick = async () => {
  if (!selectedVideo) return;
  goBtn.disabled = true;
  spinner.classList.remove("hidden");
  statusLine.textContent = "Starting analysis...";
  resultsEl.classList.add("hidden");

  const payload = { video: selectedVideo };
  if (!orientationBlock.classList.contains("hidden")) {
    payload.nose_direction = document.getElementById("nose-direction-select").value;
  }
  if (!calibrationBlock.classList.contains("hidden") && tapeWidthPx) {
    payload.tape_width_px = tapeWidthPx;
  }
  if (!beltSpeedBlock.classList.contains("hidden")) {
    payload.belt_speed = document.getElementById("belt-speed-input").value;
  }

  try {
    const started = await fetchJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await pollJob(started.job_id);
  } catch (err) {
    if (err.status === 404) {
      // This video was uploaded before a server restart/redeploy and no
      // longer exists there (free-tier hosts don't keep uploaded files
      // across restarts) -- refresh the list so it's obvious it's gone.
      statusLine.textContent = "This video is no longer on the server (it may have been cleared by a restart). Please re-upload it.";
      selectedVideo = null;
      await loadVideos();
    } else {
      statusLine.textContent = "Error: " + err.message;
    }
  } finally {
    goBtn.disabled = false;
    spinner.classList.add("hidden");
  }
};

// --- Upload ---
const dropZone = document.getElementById("drop-zone");
const dropZoneText = document.getElementById("drop-zone-text");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");

dropZone.onclick = () => fileInput.click();
fileInput.onchange = () => {
  if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
};
dropZone.ondragover = (evt) => {
  evt.preventDefault();
  dropZone.classList.add("dragover");
};
dropZone.ondragleave = () => dropZone.classList.remove("dragover");
dropZone.ondrop = (evt) => {
  evt.preventDefault();
  dropZone.classList.remove("dragover");
  if (evt.dataTransfer.files.length > 0) uploadFile(evt.dataTransfer.files[0]);
};

async function uploadFile(file) {
  dropZone.classList.add("uploading");
  const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
  dropZoneText.textContent = `Uploading ${file.name} (${sizeMb} MB)...`;
  uploadStatus.textContent = "";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const data = await fetchJson("/api/upload", { method: "POST", body: formData });
    uploadStatus.textContent = `Uploaded: ${data.name}`;
    await loadVideos();
    const uploadedVideo = videos.find((v) => v.name === data.name);
    if (uploadedVideo) selectVideo(uploadedVideo);
  } catch (err) {
    uploadStatus.textContent = "Upload failed: " + err.message;
  } finally {
    dropZone.classList.remove("uploading");
    dropZoneText.textContent = "Drag a .mp4 / .avi / .mov here, or click to choose a file";
    fileInput.value = "";
  }
}

loadVideos();

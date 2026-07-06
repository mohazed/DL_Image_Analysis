const photoEl = document.getElementById("photo");
const sourceTagEl = document.getElementById("source-tag");
const shuffleBtn = document.getElementById("shuffle-btn");
const guessBtn = document.getElementById("guess-btn");
const resultEl = document.getElementById("result");
const predictedValueEl = document.getElementById("predicted-value");
const trueValueEl = document.getElementById("true-value");
const errorBadgeEl = document.getElementById("error-badge");
const errorValueEl = document.getElementById("error-value");
const statusEl = document.getElementById("status");

let currentSample = null;

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function resetResult() {
  resultEl.classList.add("hidden");
  errorBadgeEl.classList.remove("good", "mid", "bad");
  guessBtn.disabled = false;
}

async function loadRandomSample() {
  setStatus("Loading a sample...");
  guessBtn.disabled = true;
  shuffleBtn.disabled = true;
  try {
    const res = await fetch("/api/random-sample");
    if (!res.ok) throw new Error(`random-sample failed: ${res.status}`);
    currentSample = await res.json();
    photoEl.src = currentSample.image_url;
    sourceTagEl.textContent = `SOURCE ${currentSample.source}`;
    sourceTagEl.className = `source-tag source-${currentSample.source}`;
    resetResult();
    setStatus("");
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  } finally {
    shuffleBtn.disabled = false;
  }
}

async function guessCalories() {
  if (!currentSample) return;
  guessBtn.disabled = true;
  setStatus("Running 5-fold ensemble...");
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: currentSample.image_id }),
    });
    if (!res.ok) throw new Error(`predict failed: ${res.status}`);
    const data = await res.json();

    predictedValueEl.textContent = `${Math.round(data.predicted_calories)} kcal`;
    trueValueEl.textContent = `${Math.round(data.true_calories)} kcal`;

    const err = data.absolute_error;
    errorValueEl.textContent = `${err.toFixed(1)} kcal off`;
    errorBadgeEl.classList.remove("good", "mid", "bad");
    if (err < 50) errorBadgeEl.classList.add("good");
    else if (err <= 150) errorBadgeEl.classList.add("mid");
    else errorBadgeEl.classList.add("bad");

    resultEl.classList.remove("hidden");
    setStatus("");
  } catch (err) {
    setStatus(`Error: ${err.message}`);
    guessBtn.disabled = false;
  }
}

shuffleBtn.addEventListener("click", loadRandomSample);
guessBtn.addEventListener("click", guessCalories);

// ============ Tab switching ============
const tabButtons = document.querySelectorAll(".tab");
const views = {
  game: document.getElementById("view-game"),
  upload: document.getElementById("view-upload"),
};

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const view = btn.dataset.view;
    tabButtons.forEach((b) => b.classList.toggle("active", b === btn));
    Object.entries(views).forEach(([name, el]) => {
      el.classList.toggle("hidden", name !== view);
    });
  });
});

// ============ Upload + predict ============
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const uploadPhotoEl = document.getElementById("upload-photo");
const uploadSourceTagEl = document.getElementById("upload-source-tag");
const dropzoneHintEl = document.getElementById("dropzone-hint");
const sourceInputEl = document.getElementById("source-input");
const predictBtn = document.getElementById("predict-btn");
const uploadResultEl = document.getElementById("upload-result");
const uploadPredictedEl = document.getElementById("upload-predicted");
const uploadClipNoteEl = document.getElementById("upload-clip-note");
const foldBarsEl = document.getElementById("fold-bars");
const uploadStatusEl = document.getElementById("upload-status");

let selectedFile = null;
let previewUrl = null;

function setUploadStatus(msg) {
  uploadStatusEl.textContent = msg || "";
}

function acceptFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setUploadStatus("Please choose an image file.");
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  uploadPhotoEl.src = previewUrl;
  uploadPhotoEl.classList.remove("hidden");
  dropzoneHintEl.classList.add("hidden");
  uploadSourceTagEl.classList.add("hidden");
  uploadResultEl.classList.add("hidden");
  predictBtn.disabled = false;
  setUploadStatus(file.name);
}

fileInput.addEventListener("change", (e) => acceptFile(e.target.files[0]));

["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragging");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragging");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) acceptFile(file);
});

function renderFoldBars(folds) {
  foldBarsEl.innerHTML = "";
  const max = Math.max(...folds, 1);
  folds.forEach((val, i) => {
    const bar = document.createElement("div");
    bar.className = "fold-bar";
    bar.innerHTML = `
      <span class="fold-idx">f${i}</span>
      <span class="fold-track"><span class="fold-fill" style="width:${(val / max) * 100}%"></span></span>
      <span class="fold-val">${Math.round(val)}</span>`;
    foldBarsEl.appendChild(bar);
  });
}

async function predictUpload() {
  if (!selectedFile) return;
  predictBtn.disabled = true;
  setUploadStatus("Running 5-fold ensemble...");
  try {
    const form = new FormData();
    form.append("file", selectedFile);
    form.append("source", sourceInputEl.value);
    const res = await fetch("/api/predict-upload", { method: "POST", body: form });
    if (!res.ok) {
      let detail = res.status;
      try {
        detail = (await res.json()).detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const data = await res.json();

    uploadPredictedEl.textContent = Math.round(data.predicted_calories);

    uploadSourceTagEl.textContent = `SOURCE ${data.source}${data.source_auto ? " (auto)" : ""}`;
    uploadSourceTagEl.className = `source-tag source-${data.source}`;

    if (data.was_clipped) {
      uploadClipNoteEl.textContent = `Raw ensemble ${Math.round(
        data.raw_calories
      )} kcal → clipped to source-${data.source} range`;
      uploadClipNoteEl.classList.remove("hidden");
    } else {
      uploadClipNoteEl.classList.add("hidden");
    }

    renderFoldBars(data.fold_predictions);
    uploadResultEl.classList.remove("hidden");
    setUploadStatus("");
  } catch (err) {
    setUploadStatus(`Error: ${err.message}`);
  } finally {
    predictBtn.disabled = false;
  }
}

predictBtn.addEventListener("click", predictUpload);

loadRandomSample();

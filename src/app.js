(() => {
  "use strict";

  const MODEL_BASE = "facetrace-models";
  const DISTANCE_FOR_ZERO_PERCENT = 1.20;
  const MAX_ANALYSIS_SIDE = 1600;
  const THUMBNAIL_SIDE = 260;
  const FACE_CROP_SIDE = 144;
  const SUPPORTED_IMAGE_EXTENSION = /\.(avif|bmp|gif|jpe?g|png|webp)$/i;
  const READBACK_CONTEXT_OPTIONS = { alpha: false, willReadFrequently: true };
  const DRAW_CONTEXT_OPTIONS = { alpha: false, willReadFrequently: false };
  let analysisQueue = Promise.resolve();

  const state = {
    modelsReady: false,
    modelError: null,
    runToken: 0,
    processing: false,
    reference: null,
    referenceFaceIndex: 0,
    candidates: [],
    nextCandidateId: 1,
    sortMode: "similarity-desc"
  };

  const elements = {
    modelStatus: document.getElementById("modelStatus"),
    modelDot: document.getElementById("modelDot"),
    modelStatusText: document.getElementById("modelStatusText"),
    progressWrap: document.getElementById("progressWrap"),
    progressBar: document.getElementById("progressBar"),
    progressText: document.getElementById("progressText"),
    clearButton: document.getElementById("clearButton"),
    exportButton: document.getElementById("exportButton"),
    referencePickButton: document.getElementById("referencePickButton"),
    referenceInput: document.getElementById("referenceInput"),
    referenceDropzone: document.getElementById("referenceDropzone"),
    referencePreview: document.getElementById("referencePreview"),
    referenceMessage: document.getElementById("referenceMessage"),
    referenceFaces: document.getElementById("referenceFaces"),
    candidatePickButton: document.getElementById("candidatePickButton"),
    candidateInput: document.getElementById("candidateInput"),
    candidateDropzone: document.getElementById("candidateDropzone"),
    candidateMessage: document.getElementById("candidateMessage"),
    sortSelect: document.getElementById("sortSelect"),
    summaryCounts: document.getElementById("summaryCounts"),
    resultsList: document.getElementById("resultsList")
  };

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;"
    })[char]);
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function formatNumber(value, digits = 4) {
    return Number.isFinite(value) ? value.toFixed(digits) : "";
  }

  function sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

  function yieldToBrowser() {
    if ("requestIdleCallback" in window) {
      return new Promise((resolve) => window.requestIdleCallback(resolve, { timeout: 60 }));
    }
    return sleep(0);
  }

  function runAnalysisExclusive(task) {
    const previous = analysisQueue.catch(() => undefined);
    const current = previous.then(task);
    analysisQueue = current.catch(() => undefined);
    return current;
  }

  function get2dContext(canvas, options) {
    const context = canvas.getContext("2d", options);
    if (!context) {
      throw new Error("Canvas 2D context is unavailable");
    }
    return context;
  }

  function isLikelyImage(file) {
    return Boolean(file && (file.type.startsWith("image/") || SUPPORTED_IMAGE_EXTENSION.test(file.name || "")));
  }

  function installOfflineModelFetch() {
    const embedded = window.FACETRACE_EMBEDDED_MODELS;
    if (!embedded || typeof embedded !== "object") {
      throw new Error("Local embedded model bundle is missing. This index.html copy may be incomplete.");
    }

    const decodedCache = new Map();
    function requestName(input) {
      const raw = typeof input === "string" ? input : input && input.url ? input.url : "";
      try {
        const url = new URL(raw, window.location.href);
        return decodeURIComponent(url.pathname.split("/").pop() || "");
      } catch (_error) {
        return decodeURIComponent(String(raw).split("?")[0].split("#")[0].split("/").pop() || "");
      }
    }

    function base64ToUint8Array(base64) {
      if (decodedCache.has(base64)) {
        return decodedCache.get(base64);
      }

      const binary = window.atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }
      decodedCache.set(base64, bytes);
      return bytes;
    }

    function offlineFetch(input, init) {
      const name = requestName(input);
      const entry = embedded[name];

      if (entry) {
        if (entry.kind === "json") {
          return Promise.resolve(new Response(entry.text, {
            status: 200,
            headers: { "content-type": "application/json" }
          }));
        }

        if (entry.kind === "binary") {
          const bytes = base64ToUint8Array(entry.base64);
          return Promise.resolve(new Response(bytes.slice().buffer, {
            status: 200,
            headers: { "content-type": "application/octet-stream" }
          }));
        }
      }

      const raw = typeof input === "string" ? input : input && input.url ? input.url : "";
      if (/^https?:/i.test(raw)) {
        return Promise.reject(new Error("Network requests are disabled in FaceTrace Offline."));
      }

      return Promise.reject(new Error(`Blocked non-embedded local request: ${name || "unknown asset"}`));
    }

    window.fetch = offlineFetch;
    if (window.faceapi && faceapi.env && typeof faceapi.env.getEnv === "function") {
      try {
        faceapi.env.getEnv().fetch = offlineFetch;
      } catch (_error) {
        if (typeof faceapi.env.createBrowserEnv === "function" && typeof faceapi.env.setEnv === "function") {
          const browserEnv = faceapi.env.createBrowserEnv();
          browserEnv.fetch = offlineFetch;
          faceapi.env.setEnv(browserEnv);
        }
      }
    }
  }

  async function loadModels() {
    setModelStatus("loading", "Loading local face models...");
    try {
      if (!window.faceapi) {
        throw new Error("Embedded face-api.js did not initialize. This index.html copy may be incomplete.");
      }
      installOfflineModelFetch();
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_BASE),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_BASE),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_BASE)
      ]);
      state.modelsReady = true;
      state.modelError = null;
      setModelStatus("ready", "Local face models loaded. Offline mode is ready.");
      updateCandidateMessage();
      processCandidateQueue();
    } catch (error) {
      state.modelsReady = false;
      state.modelError = error;
      setModelStatus("error", `Model not loaded: ${error.message}`);
      elements.referenceMessage.className = "message error";
      elements.referenceMessage.textContent = "Model not loaded. Confirm you are opening the generated self-contained index.html.";
      updateCandidateMessage();
    }
  }

  function setModelStatus(kind, text) {
    elements.modelDot.className = `status-dot ${kind === "ready" ? "ready" : kind === "error" ? "error" : ""}`;
    elements.modelStatusText.textContent = text;
  }

  function bindDropzone(dropzone, input, onFiles) {
    const openPicker = () => input.click();
    dropzone.addEventListener("click", openPicker);
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openPicker();
      }
    });
    input.addEventListener("change", () => {
      onFiles(Array.from(input.files || []));
      input.value = "";
    });
    ["dragenter", "dragover"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", (event) => {
      const files = Array.from(event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files : []);
      onFiles(files);
    });
  }

  elements.referencePickButton.addEventListener("click", () => elements.referenceInput.click());
  elements.candidatePickButton.addEventListener("click", () => elements.candidateInput.click());
  bindDropzone(elements.referenceDropzone, elements.referenceInput, handleReferenceFiles);
  bindDropzone(elements.candidateDropzone, elements.candidateInput, handleCandidateFiles);

  elements.clearButton.addEventListener("click", () => {
    state.runToken += 1;
    state.processing = false;
    state.reference = null;
    state.referenceFaceIndex = 0;
    state.candidates = [];
    state.nextCandidateId = 1;
    hideProgress();
    renderReference();
    renderResults();
    updateCandidateMessage();
  });

  elements.exportButton.addEventListener("click", exportCsv);
  elements.sortSelect.addEventListener("change", () => {
    state.sortMode = elements.sortSelect.value;
    renderResults();
  });

  async function handleReferenceFiles(files) {
    const imageFiles = files.filter(isLikelyImage);
    if (!imageFiles.length) {
      elements.referenceMessage.className = "message error";
      elements.referenceMessage.textContent = "Image could not be read. Select a browser-readable image file.";
      return;
    }

    state.runToken += 1;
    const token = state.runToken;
    state.reference = {
      fileName: imageFiles[0].name,
      status: "processing",
      faces: [],
      thumbnail: "",
      error: null
    };
    state.referenceFaceIndex = 0;
    renderReference();
    showProgress(0, 1, "Processing reference image...");

    try {
      ensureModelsReady();
      await yieldToBrowser();
      const analysis = await analyzeImageFile(imageFiles[0]);
      if (token !== state.runToken) return;
      state.reference = analysis;

      if (analysis.faces.length > 0) {
        state.referenceFaceIndex = chooseDefaultReferenceFace(analysis.faces);
        rescoreAllCandidates();
        await processCandidateQueue();
      }
    } catch (error) {
      if (token !== state.runToken) return;
      state.reference = {
        fileName: imageFiles[0].name,
        status: "error",
        faces: [],
        thumbnail: "",
        error: normalizeError(error)
      };
    } finally {
      if (token === state.runToken) {
        hideProgress();
        renderReference();
        renderResults();
        updateCandidateMessage();
      }
    }
  }

  function handleCandidateFiles(files) {
    const imageFiles = files.filter(isLikelyImage);
    const rejectedCount = files.length - imageFiles.length;

    if (!imageFiles.length) {
      elements.candidateMessage.className = "message error";
      elements.candidateMessage.textContent = "No browser-readable image files were selected.";
      return;
    }

    for (const file of imageFiles) {
      state.candidates.push({
        id: state.nextCandidateId++,
        file,
        fileName: file.name,
        status: "queued",
        result: null,
        error: null
      });
    }

    elements.candidateMessage.className = rejectedCount ? "message warn" : "message";
    elements.candidateMessage.textContent = rejectedCount
      ? `${imageFiles.length} image(s) queued. ${rejectedCount} non-image file(s) ignored.`
      : `${imageFiles.length} image(s) queued for local processing.`;

    renderResults();
    processCandidateQueue();
  }

  function ensureModelsReady() {
    if (!state.modelsReady) {
      throw new Error(state.modelError ? `Model not loaded: ${state.modelError.message}` : "Model not loaded yet.");
    }
  }

  function getReferenceFace() {
    if (!state.reference || !state.reference.faces || !state.reference.faces.length) {
      return null;
    }
    return state.reference.faces[state.referenceFaceIndex] || state.reference.faces[0] || null;
  }

  function hasUsableReference() {
    return Boolean(getReferenceFace());
  }

  async function processCandidateQueue() {
    if (state.processing || !state.candidates.length || !hasUsableReference()) {
      updateCandidateMessage();
      renderResults();
      return;
    }

    if (!state.modelsReady) {
      updateCandidateMessage();
      return;
    }

    const token = state.runToken;
    state.processing = true;
    const queue = state.candidates.filter((candidate) => candidate.status === "queued" || candidate.status === "error-read");
    let completed = 0;

    showProgress(0, Math.max(queue.length, 1), queue.length ? "Processing candidate images..." : "No queued images.");

    for (const candidate of queue) {
      if (token !== state.runToken) break;
      candidate.status = "processing";
      renderResults();
      showProgress(completed, queue.length, `Processing ${candidate.fileName}...`);

      try {
        await yieldToBrowser();
        const analysis = await analyzeImageFile(candidate.file);
        if (token !== state.runToken) break;
        candidate.result = scoreCandidateAnalysis(analysis);
        candidate.status = "done";
        candidate.error = null;
      } catch (error) {
        if (token !== state.runToken) break;
        candidate.result = {
          fileName: candidate.fileName,
          thumbnail: "",
          width: 0,
          height: 0,
          faces: [],
          comparisons: [],
          best: null,
          statusKind: "error",
          statusText: normalizeError(error),
          faceCount: 0,
          error: normalizeError(error)
        };
        candidate.status = "done";
        candidate.error = normalizeError(error);
      }

      completed += 1;
      showProgress(completed, queue.length, `${completed} of ${queue.length} candidate image(s) processed.`);
      renderResults();
      await yieldToBrowser();
    }

    if (token === state.runToken) {
      state.processing = false;
      hideProgress();
      updateCandidateMessage();
      renderResults();

      if (state.candidates.some((candidate) => candidate.status === "queued")) {
        processCandidateQueue();
      }
    }
  }

  async function analyzeImageFile(file) {
    return runAnalysisExclusive(() => analyzeImageFileNow(file));
  }

  async function analyzeImageFileNow(file) {
    const loaded = await loadImage(file);
    let analysisCanvas = null;
    try {
      analysisCanvas = drawImageToCanvas(loaded.image, MAX_ANALYSIS_SIDE);
      const thumbnail = canvasToDataUrl(analysisCanvas, THUMBNAIL_SIDE, 0.82);

      // Face detection, landmarks, and 128D embedding extraction all happen locally
      // through the bundled face-api.js models.
      const options = new faceapi.TinyFaceDetectorOptions({
        inputSize: 608,
        scoreThreshold: 0.35
      });
      const endTensorScope = beginTensorScope();
      let faces;
      try {
        const detections = await faceapi
          .detectAllFaces(analysisCanvas, options)
          .withFaceLandmarks()
          .withFaceDescriptors();

        faces = detections
          .map((detection, index) => {
            const box = detection.detection.box;
            return {
              index,
              descriptor: new Float32Array(detection.descriptor),
              score: detection.detection.score,
              box: {
                x: box.x,
                y: box.y,
                width: box.width,
                height: box.height
              },
              crop: cropFaceToDataUrl(analysisCanvas, box, FACE_CROP_SIDE)
            };
          })
          .sort((left, right) => (right.box.width * right.box.height) - (left.box.width * left.box.height));
      } finally {
        endTensorScope();
      }

      return {
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type || "unknown",
        width: analysisCanvas.width,
        height: analysisCanvas.height,
        thumbnail,
        faces,
        faceCount: faces.length,
        status: faces.length ? "ok" : "no-face",
        error: faces.length ? null : "No face detected"
      };
    } finally {
      loaded.release();
      if (analysisCanvas) {
        analysisCanvas.width = 0;
        analysisCanvas.height = 0;
      }
    }
  }

  function beginTensorScope() {
    const engine = window.faceapi && faceapi.tf && typeof faceapi.tf.engine === "function"
      ? faceapi.tf.engine()
      : null;

    if (!engine || typeof engine.startScope !== "function" || typeof engine.endScope !== "function") {
      return () => {};
    }

    engine.startScope();
    return () => {
      try {
        engine.endScope();
      } catch (_error) {
        // Older TensorFlow.js backends can be conservative about scopes.
        // A failed cleanup should not hide the actual image-processing result.
      }
    };
  }

  async function loadImage(file) {
    if ("createImageBitmap" in window) {
      try {
        const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
        return {
          image: bitmap,
          release() {
            if (typeof bitmap.close === "function") {
              bitmap.close();
            }
          }
        };
      } catch (_error) {
        // Fall back to HTMLImageElement decoding below for formats or
        // browsers that do not support createImageBitmap for this file.
      }
    }

    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const image = new Image();
      image.decoding = "async";
      image.onload = () => resolve({
        image,
        release() {
          URL.revokeObjectURL(url);
        }
      });
      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("Image could not be read"));
      };
      image.src = url;
    });
  }

  function drawImageToCanvas(image, maxSide) {
    const sourceWidth = image.naturalWidth || image.width;
    const sourceHeight = image.naturalHeight || image.height;
    if (!sourceWidth || !sourceHeight) {
      throw new Error("Image could not be read");
    }

    const scale = Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(sourceWidth * scale));
    canvas.height = Math.max(1, Math.round(sourceHeight * scale));
    const context = get2dContext(canvas, READBACK_CONTEXT_OPTIONS);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  function canvasToDataUrl(sourceCanvas, maxSide, quality) {
    const scale = Math.min(1, maxSide / Math.max(sourceCanvas.width, sourceCanvas.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(sourceCanvas.width * scale));
    canvas.height = Math.max(1, Math.round(sourceCanvas.height * scale));
    const context = get2dContext(canvas, DRAW_CONTEXT_OPTIONS);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", quality);
    canvas.width = 0;
    canvas.height = 0;
    return dataUrl;
  }

  function cropFaceToDataUrl(sourceCanvas, box, size) {
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    const square = Math.max(box.width, box.height) * 1.55;
    const sourceX = clamp(centerX - square / 2, 0, sourceCanvas.width);
    const sourceY = clamp(centerY - square / 2, 0, sourceCanvas.height);
    const sourceRight = clamp(centerX + square / 2, 0, sourceCanvas.width);
    const sourceBottom = clamp(centerY + square / 2, 0, sourceCanvas.height);
    const sourceWidth = Math.max(1, sourceRight - sourceX);
    const sourceHeight = Math.max(1, sourceBottom - sourceY);
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const context = get2dContext(canvas, DRAW_CONTEXT_OPTIONS);
    context.fillStyle = "#eef2f7";
    context.fillRect(0, 0, size, size);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(sourceCanvas, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, size, size);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.86);
    canvas.width = 0;
    canvas.height = 0;
    return dataUrl;
  }

  function chooseDefaultReferenceFace(faces) {
    let bestIndex = 0;
    let bestScore = -Infinity;
    faces.forEach((face, index) => {
      const area = face.box.width * face.box.height;
      const score = area * Math.max(face.score, 0.01);
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    });
    return bestIndex;
  }

  function rescoreAllCandidates() {
    for (const candidate of state.candidates) {
      if (candidate.result && candidate.result.faces) {
        candidate.result = scoreCandidateAnalysis(candidate.result);
      }
    }
  }

  function scoreCandidateAnalysis(analysis) {
    const referenceFace = getReferenceFace();
    const base = {
      ...analysis,
      comparisons: [],
      best: null,
      statusKind: "error",
      statusText: "",
      error: analysis.error || null
    };

    if (!referenceFace) {
      base.statusKind = "error";
      base.statusText = "Model not loaded or reference face missing";
      return base;
    }

    if (!analysis.faces || analysis.faces.length === 0) {
      base.statusKind = "error";
      base.statusText = "No face detected";
      base.error = "No face detected";
      return base;
    }

    // Raw Euclidean distance is converted to a stable monotonic percentage
    // for the main UI. The distance itself is kept in details only.
    base.comparisons = analysis.faces.map((face, index) => {
      const distance = euclideanDistance(referenceFace.descriptor, face.descriptor);
      const cosine = cosineSimilarity(referenceFace.descriptor, face.descriptor);
      const similarity = distanceToPercent(distance);
      return {
        faceIndex: index,
        originalFaceIndex: face.index,
        distance,
        cosine,
        similarity,
        interpretation: interpretSimilarity(similarity),
        detectorScore: face.score,
        crop: face.crop
      };
    });

    base.best = base.comparisons.reduce((best, item) => {
      if (!best || item.similarity > best.similarity) return item;
      if (item.similarity === best.similarity && item.distance < best.distance) return item;
      return best;
    }, null);

    if (analysis.faces.length > 1) {
      base.statusKind = "warn";
      base.statusText = `Multiple faces detected (${analysis.faces.length}); closest face compared`;
    } else {
      base.statusKind = "ok";
      base.statusText = "Face detected";
    }

    return base;
  }

  function euclideanDistance(left, right) {
    if (!left || !right || left.length !== right.length) {
      return Number.POSITIVE_INFINITY;
    }
    let sum = 0;
    for (let index = 0; index < left.length; index += 1) {
      const delta = left[index] - right[index];
      sum += delta * delta;
    }
    return Math.sqrt(sum);
  }

  function cosineSimilarity(left, right) {
    if (!left || !right || left.length !== right.length) {
      return Number.NaN;
    }
    let dot = 0;
    let leftNorm = 0;
    let rightNorm = 0;
    for (let index = 0; index < left.length; index += 1) {
      dot += left[index] * right[index];
      leftNorm += left[index] * left[index];
      rightNorm += right[index] * right[index];
    }
    if (!leftNorm || !rightNorm) return Number.NaN;
    return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
  }

  function distanceToPercent(distance) {
    if (!Number.isFinite(distance)) return 0;
    return Math.round(clamp(1 - distance / DISTANCE_FOR_ZERO_PERCENT, 0, 1) * 100);
  }

  function interpretSimilarity(percent) {
    if (percent >= 85) return "very similar";
    if (percent >= 70) return "similar";
    if (percent >= 50) return "possibly similar";
    return "low similarity";
  }

  function scoreClass(percent) {
    if (!Number.isFinite(percent)) return "unavailable";
    if (percent >= 85) return "strong";
    if (percent >= 70) return "";
    if (percent >= 50) return "maybe";
    return "low";
  }

  function normalizeError(error) {
    if (!error) return "Image could not be read";
    const message = error.message || String(error);
    if (/model not loaded/i.test(message)) return message;
    if (/could not be read|decode|image/i.test(message)) return "Image could not be read";
    return message;
  }

  function showProgress(value, max, text) {
    elements.progressWrap.classList.add("active");
    elements.progressBar.max = Math.max(1, max || 1);
    elements.progressBar.value = clamp(value || 0, 0, elements.progressBar.max);
    elements.progressText.textContent = text || "Processing...";
  }

  function hideProgress() {
    elements.progressWrap.classList.remove("active");
    elements.progressBar.value = 0;
    elements.progressText.textContent = "Idle";
  }

  function renderReference() {
    const reference = state.reference;
    elements.referenceFaces.innerHTML = "";

    if (!reference) {
      elements.referencePreview.innerHTML = `<div class="empty-state">No reference image selected.</div>`;
      elements.referenceMessage.className = "message";
      elements.referenceMessage.textContent = "Select a clear image with one visible face. If several faces are detected, choose the reference face below.";
      return;
    }

    if (reference.thumbnail) {
      elements.referencePreview.innerHTML = `<img src="${reference.thumbnail}" alt="Reference image preview">`;
    } else {
      elements.referencePreview.innerHTML = `<div class="empty-state">Processing ${escapeHtml(reference.fileName)}...</div>`;
    }

    if (reference.status === "processing") {
      elements.referenceMessage.className = "message";
      elements.referenceMessage.textContent = "Detecting faces and extracting the local descriptor...";
      return;
    }

    if (reference.error || !reference.faces.length) {
      elements.referenceMessage.className = "message error";
      elements.referenceMessage.textContent = reference.error || "No face detected";
      return;
    }

    if (reference.faces.length > 1) {
      elements.referenceMessage.className = "message warn";
      elements.referenceMessage.textContent = `Multiple faces detected (${reference.faces.length}). The selected face below is used as the reference.`;
    } else {
      elements.referenceMessage.className = "message";
      elements.referenceMessage.textContent = "Reference face ready. Candidate images will be sorted by similarity percentage.";
    }

    const label = document.createElement("div");
    label.className = "summary-counts";
    label.textContent = "Detected reference faces";
    const strip = document.createElement("div");
    strip.className = "face-strip";

    reference.faces.forEach((face, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `face-choice ${index === state.referenceFaceIndex ? "active" : ""}`;
      button.innerHTML = `
        <img src="${face.crop}" alt="Reference face ${index + 1}">
        <span>Face ${index + 1}</span>
      `;
      button.addEventListener("click", () => {
        state.referenceFaceIndex = index;
        rescoreAllCandidates();
        renderReference();
        renderResults();
      });
      strip.appendChild(button);
    });

    elements.referenceFaces.append(label, strip);
  }

  function renderResults() {
    elements.exportButton.disabled = !state.candidates.some((candidate) => candidate.result);

    if (!state.candidates.length) {
      elements.summaryCounts.textContent = "No candidate images selected.";
      elements.resultsList.innerHTML = `<div class="empty-state">Results will appear here after a reference and candidate image are processed.</div>`;
      return;
    }

    const done = state.candidates.filter((candidate) => candidate.result).length;
    const withScores = state.candidates.filter((candidate) => candidate.result && candidate.result.best).length;
    const errors = state.candidates.filter((candidate) => candidate.result && candidate.result.statusKind === "error").length;
    elements.summaryCounts.textContent = `${done} of ${state.candidates.length} processed, ${withScores} scored, ${errors} with errors.`;

    const sorted = getSortedCandidates();
    elements.resultsList.innerHTML = sorted.map(renderCandidateCard).join("");
  }

  function getSortedCandidates() {
    const candidates = [...state.candidates];
    const similarity = (candidate) => candidate.result && candidate.result.best ? candidate.result.best.similarity : -1;
    const name = (candidate) => candidate.fileName.toLocaleLowerCase();
    const status = (candidate) => candidate.result ? candidate.result.statusText || candidate.status : candidate.status;

    candidates.sort((left, right) => {
      if (state.sortMode === "similarity-asc") {
        return similarity(left) - similarity(right) || name(left).localeCompare(name(right));
      }
      if (state.sortMode === "name-asc") {
        return name(left).localeCompare(name(right));
      }
      if (state.sortMode === "status") {
        return status(left).localeCompare(status(right)) || similarity(right) - similarity(left);
      }
      return similarity(right) - similarity(left) || name(left).localeCompare(name(right));
    });
    return candidates;
  }

  function renderCandidateCard(candidate) {
    if (!candidate.result) {
      const status = candidate.status === "processing" ? "Processing" : hasUsableReference() ? "Queued" : "Waiting for reference";
      return `
        <article class="result-card">
          <div class="thumb"><div class="empty-state">${candidate.status === "processing" ? "..." : ""}</div></div>
          <div class="face-thumb"><div class="empty-state">Face</div></div>
          <div class="result-main">
            <div class="filename" title="${escapeHtml(candidate.fileName)}">${escapeHtml(candidate.fileName)}</div>
            <div class="status-row"><span class="badge">${status}</span></div>
          </div>
          <div class="score-block">
            <div class="score unavailable">Pending</div>
            <div class="interpretation">${state.modelsReady ? "Awaiting local processing" : "Model not loaded"}</div>
          </div>
        </article>
      `;
    }

    const result = candidate.result;
    const best = result.best;
    const percent = best ? best.similarity : null;
    const interpretation = best ? best.interpretation : result.statusText;
    const statusKind = result.statusKind || "error";
    const bestFace = best ? result.faces[best.faceIndex] : null;
    const scoreText = best ? `${percent}%` : "No score";
    const scoreLabel = best ? `${percent}% similar` : "No similarity score";
    const thumb = result.thumbnail
      ? `<img src="${result.thumbnail}" alt="Candidate image preview">`
      : `<div class="empty-state">No preview</div>`;
    const faceThumb = bestFace && bestFace.crop
      ? `<img src="${bestFace.crop}" alt="Best detected face preview">`
      : `<div class="empty-state">No face</div>`;

    return `
      <article class="result-card">
        <div>
          <div class="thumb">${thumb}</div>
          <div class="thumb-label">Image</div>
        </div>
        <div>
          <div class="face-thumb">${faceThumb}</div>
          <div class="face-label">${best ? "Best face" : "Face"}</div>
        </div>
        <div class="result-main">
          <div class="filename" title="${escapeHtml(candidate.fileName)}">${escapeHtml(candidate.fileName)}</div>
          <div class="status-row">
            <span class="badge ${statusKind}">${escapeHtml(result.statusText)}</span>
            <span class="badge">${result.faceCount || 0} face${result.faceCount === 1 ? "" : "s"}</span>
          </div>
        </div>
        <div class="score-block">
          <div class="score ${scoreClass(percent)}" aria-label="${scoreLabel}">${escapeHtml(scoreText)}</div>
          <div class="interpretation">${escapeHtml(interpretation)}</div>
        </div>
        ${renderDetails(result)}
      </article>
    `;
  }

  function renderDetails(result) {
    const best = result.best;
    const cells = [
      ["Filename", result.fileName],
      ["Detection status", result.statusText || ""],
      ["Detected faces", String(result.faceCount || 0)],
      ["Image size used", result.width && result.height ? `${result.width} x ${result.height}` : ""]
    ];

    if (best) {
      cells.push(
        ["Similarity", `${best.similarity}% (${best.interpretation})`],
        ["Euclidean distance", formatNumber(best.distance, 5)],
        ["Cosine similarity", formatNumber(best.cosine, 5)],
        ["Detector score", formatNumber(best.detectorScore, 4)]
      );
    } else if (result.error) {
      cells.push(["Error", result.error]);
    }

    const faceList = result.comparisons && result.comparisons.length
      ? `
        <div class="candidate-faces">
          ${result.comparisons.map((item) => `
            <div class="candidate-face ${best && item.faceIndex === best.faceIndex ? "best" : ""}">
              <img src="${item.crop}" alt="Detected candidate face ${item.faceIndex + 1}">
              <strong>${item.similarity}%</strong>
              <span>Face ${item.faceIndex + 1}</span>
              <span>d=${formatNumber(item.distance, 3)}</span>
            </div>
          `).join("")}
        </div>
      `
      : "";

    return `
      <details class="details">
        <summary>Technical details</summary>
        <div class="detail-grid">
          ${cells.map(([label, value]) => `
            <div class="detail-cell">
              <b>${escapeHtml(label)}</b>
              <span>${escapeHtml(value)}</span>
            </div>
          `).join("")}
        </div>
        ${faceList}
      </details>
    `;
  }

  function updateCandidateMessage() {
    if (!state.modelsReady) {
      elements.candidateMessage.className = state.modelError ? "message error" : "message";
      elements.candidateMessage.textContent = state.modelError
        ? "Model not loaded. Candidate images cannot be processed until the local model bundle is available."
        : "Local models are loading. Candidate images can be queued now.";
      return;
    }

    if (!hasUsableReference()) {
      elements.candidateMessage.className = "message";
      elements.candidateMessage.textContent = state.candidates.length
        ? `${state.candidates.length} candidate image(s) queued. Select a reference face to start comparison.`
        : "Add a reference face first or add candidate images now; queued images will process once the reference is ready.";
      return;
    }

    const queued = state.candidates.filter((candidate) => candidate.status === "queued").length;
    const processed = state.candidates.filter((candidate) => candidate.result).length;
    elements.candidateMessage.className = "message";
    elements.candidateMessage.textContent = queued
      ? `${queued} image(s) queued for local processing.`
      : `${processed} candidate image(s) processed locally.`;
  }

  function exportCsv() {
    const rows = [[
      "filename",
      "status",
      "face_count",
      "similarity_percent",
      "interpretation",
      "euclidean_distance",
      "cosine_similarity",
      "best_face_index",
      "detector_score",
      "error"
    ]];

    for (const candidate of getSortedCandidates()) {
      const result = candidate.result;
      const best = result && result.best ? result.best : null;
      rows.push([
        candidate.fileName,
        result ? result.statusText : candidate.status,
        result ? String(result.faceCount || 0) : "",
        best ? String(best.similarity) : "",
        best ? best.interpretation : "",
        best ? formatNumber(best.distance, 6) : "",
        best ? formatNumber(best.cosine, 6) : "",
        best ? String(best.faceIndex + 1) : "",
        best ? formatNumber(best.detectorScore, 6) : "",
        result && result.error ? result.error : ""
      ]);
    }

    const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `facetrace-results-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 250);
  }

  function csvEscape(value) {
    const text = String(value == null ? "" : value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, "\"\"")}"` : text;
  }

  renderReference();
  renderResults();
  updateCandidateMessage();
  loadModels();
})();

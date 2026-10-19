import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

const THUMB_RATIO = 0.5;

let cvReady = false;
let handLandmarker = null;
let imgBd1 = null;
let imgBd2 = null;

let video = null;
let outputCanvas = null;
let ctx = null;

let ultimaNumero = null;
let somAtual = null;
let ultimoTimestamp = 0;

const audioAcerto = new Audio("audio/lula.mp3");
const audioErro = new Audio("audio/fogo.mp3");
const audioBg = new Audio("audio/estrela.mp3");
audioAcerto.loop = true;
audioErro.loop = true;
audioBg.loop = true;

function setStatus(msg) { const el = document.getElementById("status"); if (el) el.textContent = msg; }

function dedosLevantados(lm) {
  let d = 0;
  if (lm[8].y < lm[6].y) d++;
  if (lm[12].y < lm[10].y) d++;
  if (lm[16].y < lm[14].y) d++;
  if (lm[20].y < lm[18].y) d++;
  const palmSize = Math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y);
  const thumbDist = Math.hypot(lm[4].x - lm[5].x, lm[4].y - lm[5].y);
  if (palmSize > 0 && thumbDist > THUMB_RATIO * palmSize) d++;
  return d;
}

function overlayPng(frame, overlay, centerX, centerY, scale = 1.0) {
  if (!overlay) return;
  let ov = overlay;
  let created = false;
  let ow = ov.cols, oh = ov.rows;
  if (scale !== 1.0) {
    const dst = new cv.Mat();
    const nw = Math.max(1, Math.round(ow * scale));
    const nh = Math.max(1, Math.round(oh * scale));
    cv.resize(ov, dst, new cv.Size(nw, nh), 0, 0, cv.INTER_AREA);
    ov = dst;
    created = true;
    ow = ov.cols; oh = ov.rows;
  }
  const fh = frame.rows, fw = frame.cols;
  if (centerX === null || centerX === undefined) centerX = Math.floor(fw / 2);
  if (centerY === null || centerY === undefined) centerY = Math.floor(fh / 2);
  const x = centerX - Math.floor(ow / 2);
  const y = centerY - Math.floor(oh / 2);
  const x1 = Math.max(0, x), y1 = Math.max(0, y);
  const x2 = Math.min(fw, x + ow), y2 = Math.min(fh, y + oh);
  if (x1 >= x2 || y1 >= y2) { if (created) ov.delete(); return; }
  const ovX1 = x1 - x, ovY1 = y1 - y;
  const cropW = x2 - x1, cropH = y2 - y1;
  const frameStride = fw * 4;
  const ovStride = ov.cols * 4;
  const fd = frame.data, od = ov.data;
  for (let ry = 0; ry < cropH; ry++) {
    for (let rx = 0; rx < cropW; rx++) {
      const fi = (y1 + ry) * frameStride + (x1 + rx) * 4;
      const oi = (ovY1 + ry) * ovStride + (ovX1 + rx) * 4;
      const a = od[oi + 3] / 255.0;
      fd[fi] = Math.round((1 - a) * fd[fi] + a * od[oi]);
      fd[fi + 1] = Math.round((1 - a) * fd[fi + 1] + a * od[oi + 1]);
      fd[fi + 2] = Math.round((1 - a) * fd[fi + 2] + a * od[oi + 2]);
    }
  }
  if (created) ov.delete();
}

function desenharMaos(frame, landmarksList, corPorIndice) {
  const h = frame.rows, w = frame.cols;
  for (let i = 0; i < landmarksList.length; i++) {
    let cor = [0, 255, 0, 255];
    if (corPorIndice && corPorIndice[i]) cor = corPorIndice[i];
    const pts = landmarksList[i].map(lm => ({ x: Math.round(lm.x * w), y: Math.round(lm.y * h) }));
    for (const [a, b] of HAND_CONNECTIONS) {
      cv.line(frame, new cv.Point(pts[a].x, pts[a].y), new cv.Point(pts[b].x, pts[b].y), new cv.Scalar(cor[0], cor[1], cor[2], cor[3]), 2, cv.LINE_AA);
    }
    for (const p of pts) {
      cv.circle(frame, new cv.Point(p.x, p.y), 4, new cv.Scalar(255, 0, 0, 255), -1, cv.LINE_AA);
      cv.circle(frame, new cv.Point(p.x, p.y), 4, new cv.Scalar(cor[0], cor[1], cor[2], cor[3]), 1, cv.LINE_AA);
    }
  }
}

function desenharNumero(frame, numero) {
  let cor = [255, 255, 255, 255];
  if (numero === null || numero === undefined) cor = [200, 200, 200, 255];
  else if (numero === 22) cor = [1, 17, 180, 255];
  else if (numero === 13) cor = [255, 204, 1, 255];
  const texto = (numero !== null && numero !== undefined) ? String(numero).padStart(2, "0") : "?";
  let textSize;
  try { textSize = cv.getTextSize(texto, cv.FONT_HERSHEY_TRIPLEX, 4.0, 8, 0); }
  catch (e) { textSize = { width: 100, height: 50 }; }
  const tw = (textSize.size ? textSize.size.width : (textSize.width || 100));
  const th = (textSize.size ? textSize.size.height : (textSize.height || 50));
  const x = Math.floor((frame.cols - tw) / 2);
  const y = Math.floor((frame.rows + th) / 2);
  cv.putText(frame, texto, new cv.Point(x-20, y), cv.FONT_HERSHEY_SIMPLEX, 4.0, new cv.Scalar(cor[0], cor[1], cor[2], cor[3]), 8, cv.LINE_AA);
}

function tocarSom(numero) {
  if (numero === null || numero === undefined) {
    if (ultimaNumero === null) return;
    if (somAtual) { somAtual.pause(); somAtual.currentTime = 0; somAtual = null; }
    ultimaNumero = null;
    return;
  }
  if (numero !== 13 && numero !== 22) numero = 10;
  if (numero === ultimaNumero) return;
  let som = null;
  if (numero === 13) som = audioAcerto;
  else if (numero === 22) som = audioErro;
  else if (numero === 10) som = audioBg;
  if (somAtual && somAtual !== som) { somAtual.pause(); somAtual.currentTime = 0; somAtual = null; }
  if (som) { som.play().catch(() => {}); somAtual = som; }
  else if (somAtual) { somAtual.pause(); somAtual.currentTime = 0; somAtual = null; }
  ultimaNumero = numero;
}

async function loadImageMat(path) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.width; c.height = img.height;
      const x = c.getContext("2d");
      x.drawImage(img, 0, 0);
      let mat = cv.imread(c);
      if (mat.channels() !== 4) {
        const rgba = new cv.Mat();
        if (mat.channels() === 3) cv.cvtColor(mat, rgba, cv.COLOR_RGB2RGBA);
        else if (mat.channels() === 1) cv.cvtColor(mat, rgba, cv.COLOR_GRAY2RGBA);
        mat.delete();
        mat = rgba;
      }
      resolve(mat);
    };
    img.onerror = reject;
    img.src = path;
  });
}

async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  handLandmarker = await Promise.race([
    HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: "hand_landmarker.task", delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 2,
    }),
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout createFromOptions")), 15000))
  ]);
}

function processarFrame() {
  if (!video || video.readyState < 2) return;
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;
  if (outputCanvas.width !== w || outputCanvas.height !== h) {
    outputCanvas.width = w; outputCanvas.height = h;
  }
  // draw flipped frame to a temp canvas
  const tmp = document.createElement("canvas");
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext("2d");
  tctx.translate(w, 0); tctx.scale(-1, 1);
  tctx.drawImage(video, 0, 0);
  tctx.setTransform(1, 0, 0, 1, 0, 0);

  let timestamp = performance.now();
  if (timestamp <= ultimoTimestamp) timestamp = ultimoTimestamp + 1;
  ultimoTimestamp = timestamp;

  let results = null;
  try { results = handLandmarker.detectForVideo(tmp, timestamp); }
  catch (e) { results = null; }

  let src = cv.imread(tmp);
  // color filter: R*0.4, G=0, B=0 (RGBA: channel0=R,1=G,2=B)
  const data = src.data;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = Math.round(data[i] * 0.4);
    data[i + 1] = 0;
    data[i + 2] = 0;
  }

  let numeroFinal = null;
  let dezenaLm = null;
  let unidadeLm = null;

  const landmarks = results ? results.landmarks : null;
  const handedness = results ? results.handedness : null;

  if (landmarks && landmarks.length >= 2) {
    const handsInfo = [];
    if (handedness && handedness.length === landmarks.length) {
      for (let i = 0; i < handedness.length; i++) {
        const hi = handedness[i];
        const cat = (hi && hi[0]) ? hi[0].categoryName : null;
        handsInfo.push([i, cat || (i === 0 ? "Right" : "Left")]);
      }
    } else {
      for (let i = 0; i < landmarks.length; i++) handsInfo.push([i, i === 0 ? "Right" : "Left"]);
    }
    if (handsInfo.length !== landmarks.length) {
      // fallback
    }

    const lmPorIdx = {};
    for (let i = 0; i < landmarks.length; i++) lmPorIdx[i] = landmarks[i];

    let leftIdx = null, rightIdx = null;
    for (const [lmIdx, cat] of handsInfo) {
      if (cat === "Left") leftIdx = lmIdx;
      else if (cat === "Right") rightIdx = lmIdx;
    }

    const getX = (lmIdx) => {
      const lm = lmPorIdx[lmIdx];
      if (!lm) return 9999;
      return lm[0].x * src.cols;
    };

    if (leftIdx !== null && rightIdx !== null) {
      if (getX(leftIdx) < getX(rightIdx)) {
        dezenaLm = lmPorIdx[leftIdx];
        unidadeLm = lmPorIdx[rightIdx];
      } else {
        dezenaLm = lmPorIdx[rightIdx];
        unidadeLm = lmPorIdx[leftIdx];
      }
      const d = dedosLevantados(dezenaLm);
      const u = dedosLevantados(unidadeLm);
      numeroFinal = d * 10 + u;
    } else {
      const lmList = Object.entries(lmPorIdx).slice(0, 2);
      lmList.sort((a, b) => (a[1][0].x * src.cols) - (b[1][0].x * src.cols));
      dezenaLm = lmList[0][1];
      unidadeLm = lmList[1][1];
      const d = dedosLevantados(dezenaLm);
      const u = dedosLevantados(unidadeLm);
      numeroFinal = d * 10 + u;
    }

    if (numeroFinal === 13 && imgBd1) {
      overlayPng(src, imgBd1, Math.floor(src.cols / 2), Math.floor(src.rows / 2), 1.0);
    } else if (numeroFinal === 22 && imgBd2) {
      overlayPng(src, imgBd2, Math.floor(src.cols / 2), Math.floor(src.rows / 2), 1.0);
    }
    desenharNumero(src, numeroFinal, null);
  }

  if (landmarks && landmarks.length) {
    const corPorIndice = {};
    for (let idx = 0; idx < landmarks.length; idx++) {
      const lm = landmarks[idx];
      if (lm === dezenaLm) corPorIndice[idx] = [0, 255, 0, 255];
      else if (lm === unidadeLm) corPorIndice[idx] = [0, 255, 255, 255];
    }
    desenharMaos(src, landmarks, corPorIndice);
  }

  tocarSom(numeroFinal);

  cv.imshow(outputCanvas, src);
  if (window.matchMedia("(max-width: 768px)").matches) {
    outputCanvas.style.width = "100%";
    outputCanvas.style.maxWidth = "100%";
    outputCanvas.style.height = "auto";
  } else {
    outputCanvas.style.width = "";
    outputCanvas.style.maxWidth = "";
    outputCanvas.style.height = "";
  }
  src.delete();
}

let running = false;
function loop() {
  if (!running) return;
  try { processarFrame(); } catch (e) { console.error(e); }
  requestAnimationFrame(loop);
}

async function start() {
  try {
    setStatus("Solicitando camera...");
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video = document.getElementById("video");
    video.srcObject = stream;
    await video.play();
    running = true;
    setStatus("Rodando");
    document.getElementById("startBtn").style.display = "none";
    loop();
  } catch (e) {
    console.error(e);
    setStatus("Erro camera: " + e.message);
  }
}

async function waitForCv() {
  while (typeof cv === "undefined") await new Promise(r => setTimeout(r, 100));
  if (cv.Mat) { cvReady = true; return; }
  await new Promise(resolve => {
    cv.onRuntimeInitialized = () => { cvReady = true; resolve(); };
    setTimeout(() => { if (cv.Mat) { cvReady = true; resolve(); } }, 2000);
  });
}

async function init() {
  setStatus("Carregando MediaPipe...");
  await initMediaPipe();
  setStatus("Carregando OpenCV.js...");
  await waitForCv();
  imgBd1 = await loadImageMat("imagem/bd1.png");
  imgBd2 = await loadImageMat("imagem/bd2.png");
  outputCanvas = document.getElementById("output");
  setStatus("Pronto. Clique em Iniciar camera.");
  document.getElementById("startBtn").addEventListener("click", start);
}

init().catch(e => { console.error(e); setStatus("Erro init: " + e.message); });

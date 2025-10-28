import './style.css';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-converter';
import '@tensorflow/tfjs-backend-webgl';
import * as posedetection from '@tensorflow-models/pose-detection';

const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('overlay');
const canvasCtx = canvasElement.getContext('2d');

const recordToggleBtn = document.getElementById('recordToggle');
const stopRecordingBtn = document.getElementById('stopRecording');
const playRecordingBtn = document.getElementById('playRecording');
const pausePlaybackBtn = document.getElementById('pausePlayback');
const clearRecordingBtn = document.getElementById('clearRecording');
const playbackSlider = document.getElementById('playbackSlider');
const playbackStatus = document.getElementById('playbackStatus');
const playbackNotice = document.getElementById('playbackNotice');

const visibleCountLabel = document.getElementById('visibleCount');
const totalCountLabel = document.getElementById('totalCount');
const frameRateLabel = document.getElementById('frameRate');
const landmarkTableBody = document.getElementById('landmarkTableBody');

const KEYPOINT_LABELS = [
  'nose',
  'left_eye_inner',
  'left_eye',
  'left_eye_outer',
  'right_eye_inner',
  'right_eye',
  'right_eye_outer',
  'left_ear',
  'right_ear',
  'mouth_left',
  'mouth_right',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_pinky',
  'right_pinky',
  'left_index',
  'right_index',
  'left_thumb',
  'right_thumb',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
  'left_heel',
  'right_heel',
  'left_foot_index',
  'right_foot_index',
];

const KEYPOINT_INDEX = KEYPOINT_LABELS.reduce((acc, label, index) => {
  acc[label] = index;
  return acc;
}, {});

const SKELETON_LABEL_CONNECTIONS = [
  ['nose', 'left_eye_inner'],
  ['nose', 'right_eye_inner'],
  ['left_eye_inner', 'left_eye'],
  ['left_eye', 'left_eye_outer'],
  ['right_eye_inner', 'right_eye'],
  ['right_eye', 'right_eye_outer'],
  ['left_eye_outer', 'left_ear'],
  ['right_eye_outer', 'right_ear'],
  ['nose', 'mouth_left'],
  ['nose', 'mouth_right'],
  ['mouth_left', 'mouth_right'],
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['left_wrist', 'left_pinky'],
  ['left_wrist', 'left_index'],
  ['left_wrist', 'left_thumb'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  ['right_wrist', 'right_pinky'],
  ['right_wrist', 'right_index'],
  ['right_wrist', 'right_thumb'],
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['left_ankle', 'left_heel'],
  ['left_heel', 'left_foot_index'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle'],
  ['right_ankle', 'right_heel'],
  ['right_heel', 'right_foot_index'],
];

const SKELETON_CONNECTIONS = SKELETON_LABEL_CONNECTIONS
  .map(([from, to]) => [KEYPOINT_INDEX[from], KEYPOINT_INDEX[to]])
  .filter(([from, to]) =>
    Number.isInteger(from) && Number.isInteger(to)
  );

totalCountLabel.textContent = KEYPOINT_LABELS.length.toString();

const recordedFrames = [];
let detector = null;
let isRecording = false;
let isPlayingBack = false;
let playbackIndex = 0;
let playbackTimeoutId = null;
let lastFrameTimestamp = performance.now();

async function init() {
  await tf.setBackend('webgl');
  await tf.ready();
  detector = await posedetection.createDetector(posedetection.SupportedModels.BlazePose, {
    runtime: 'tfjs',
    modelType: 'full',
  });

  await startCamera();
  requestAnimationFrame(poseLoop);
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 960 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    videoElement.srcObject = stream;
    await videoElement.play();
  } catch (error) {
    console.error('Unable to start camera', error);
    playbackStatus.textContent = 'Camera permission denied or unavailable.';
  }
}

async function poseLoop() {
  if (!detector || videoElement.readyState < 2) {
    requestAnimationFrame(poseLoop);
    return;
  }

  const poses = await detector.estimatePoses(videoElement, {
    maxPoses: 1,
    flipHorizontal: false,
  });

  updateFrameRate();

  if (!poses.length || !poses[0].keypoints || poses[0].keypoints.length === 0) {
    if (!isPlayingBack) {
      updateVisibleCount(0);
      clearTable();
      clearCanvas();
    }
    requestAnimationFrame(poseLoop);
    return;
  }

  const keypoints = normalizeKeypoints(poses[0].keypoints);

  if (isRecording) {
    recordedFrames.push({
      timestamp: performance.now(),
      keypoints: cloneKeypoints(keypoints),
    });
    updateRecordingUi();
  }

  if (!isPlayingBack) {
    drawLiveOverlay(keypoints);
    updateVisibleCount(countVisibleKeypoints(keypoints));
    updateTable(keypoints);
  }

  requestAnimationFrame(poseLoop);
}

function normalizeKeypoints(keypoints) {
  const { videoWidth, videoHeight } = videoElement;
  const keypointMap = new Map();

  keypoints.forEach((kp, fallbackIndex) => {
    const name = kp.name ?? KEYPOINT_LABELS[fallbackIndex] ?? `point_${fallbackIndex}`;
    const index = KEYPOINT_INDEX[name] ?? fallbackIndex;
    keypointMap.set(name, {
      name,
      index,
      x: (kp.x ?? 0) / videoWidth,
      y: (kp.y ?? 0) / videoHeight,
      z: kp.z ?? 0,
      score: kp.score ?? kp.visibility ?? 0,
    });
  });

  return KEYPOINT_LABELS.map((label, index) => {
    const kp = keypointMap.get(label);
    if (kp) {
      return kp;
    }
    return {
      name: label,
      index,
      x: 0,
      y: 0,
      z: 0,
      score: 0,
    };
  });
}

function drawLiveOverlay(keypoints) {
  syncCanvasSize();
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.globalAlpha = 0.9;
  canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.globalAlpha = 1;
  drawSkeleton(keypoints, {
    lineColor: '#22d3ee',
    pointColor: '#f97316',
    pointRadius: 4,
  });
  canvasCtx.restore();
}

function drawSkeleton(keypoints, { lineColor, pointColor, pointRadius }) {
  const scaled = scaleKeypoints(keypoints);
  canvasCtx.lineWidth = 4;
  canvasCtx.lineJoin = 'round';
  canvasCtx.lineCap = 'round';
  canvasCtx.strokeStyle = lineColor;
  canvasCtx.fillStyle = pointColor;

  SKELETON_CONNECTIONS.forEach(([start, end]) => {
    const kp1 = scaled[start];
    const kp2 = scaled[end];
    if (!isKeypointVisible(kp1) || !isKeypointVisible(kp2)) return;
    canvasCtx.beginPath();
    canvasCtx.moveTo(kp1.x, kp1.y);
    canvasCtx.lineTo(kp2.x, kp2.y);
    canvasCtx.stroke();
  });

  scaled.forEach((kp) => {
    if (!isKeypointVisible(kp)) return;
    canvasCtx.beginPath();
    canvasCtx.arc(kp.x, kp.y, pointRadius, 0, Math.PI * 2);
    canvasCtx.fill();
  });
}

function scaleKeypoints(keypoints) {
  return keypoints.map((kp) => ({
    ...kp,
    x: kp.x * canvasElement.width,
    y: kp.y * canvasElement.height,
  }));
}

function isKeypointVisible(keypoint, threshold = 0.5) {
  return keypoint && (keypoint.score ?? 0) >= threshold;
}

function syncCanvasSize() {
  const { videoWidth, videoHeight } = videoElement;
  if (!videoWidth || !videoHeight) {
    return;
  }
  if (canvasElement.width !== videoWidth || canvasElement.height !== videoHeight) {
    canvasElement.width = videoWidth;
    canvasElement.height = videoHeight;
  }
}

function clearCanvas() {
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

function updateVisibleCount(count) {
  visibleCountLabel.textContent = count.toString();
}

function countVisibleKeypoints(keypoints, threshold = 0.5) {
  return keypoints.filter((kp) => (kp.score ?? 0) >= threshold).length;
}

function updateFrameRate() {
  const now = performance.now();
  const delta = now - lastFrameTimestamp;
  if (delta > 0) {
    const fps = Math.round((1000 / delta) * 10) / 10;
    frameRateLabel.textContent = `${fps.toFixed(1)} fps`;
  }
  lastFrameTimestamp = now;
}

function updateTable(keypoints) {
  landmarkTableBody.innerHTML = '';
  keypoints.forEach((kp, index) => {
    const label = KEYPOINT_LABELS[index] ?? kp.name ?? `Point ${index + 1}`;
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${label}</td>
      <td>${formatNumber(kp.score)}</td>
      <td>${formatNumber(kp.x)}</td>
      <td>${formatNumber(kp.y)}</td>
      <td>${formatNumber(kp.z)}</td>
    `;
    landmarkTableBody.appendChild(row);
  });
}

function clearTable() {
  landmarkTableBody.innerHTML = `
    <tr>
      <td colspan="5" class="empty">No landmarks yet.</td>
    </tr>
  `;
}

function formatNumber(value) {
  if (value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(3);
}

function cloneKeypoints(keypoints) {
  return keypoints.map((kp) => ({ ...kp }));
}

recordToggleBtn.addEventListener('click', () => {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
});

stopRecordingBtn.addEventListener('click', stopRecording);
playRecordingBtn.addEventListener('click', startPlayback);
pausePlaybackBtn.addEventListener('click', pausePlayback);
clearRecordingBtn.addEventListener('click', clearRecording);

playbackSlider.addEventListener('input', () => {
  if (!recordedFrames.length) return;
  const index = Number(playbackSlider.value);
  playbackIndex = index;
  pausePlayback();
  renderPlaybackFrame(recordedFrames[playbackIndex]);
  updatePlaybackStatus();
});

function startRecording() {
  if (isPlayingBack) {
    pausePlayback();
  }
  recordedFrames.length = 0;
  isRecording = true;
  recordToggleBtn.textContent = 'Recording…';
  recordToggleBtn.disabled = true;
  stopRecordingBtn.disabled = false;
  clearRecordingBtn.disabled = true;
  playRecordingBtn.disabled = true;
  playbackSlider.disabled = true;
  playbackSlider.value = '0';
  playbackSlider.max = '0';
  playbackStatus.textContent = 'Recording landmarks…';
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  recordToggleBtn.textContent = 'Start Recording';
  recordToggleBtn.disabled = false;
  stopRecordingBtn.disabled = true;
  clearRecordingBtn.disabled = recordedFrames.length === 0;
  playRecordingBtn.disabled = recordedFrames.length === 0;
  playbackSlider.disabled = recordedFrames.length === 0;
  playbackSlider.max = Math.max(0, recordedFrames.length - 1).toString();
  playbackSlider.value = '0';
  playbackStatus.textContent = recordedFrames.length
    ? `Captured ${recordedFrames.length} frames.`
    : 'Recording canceled.';
}

function updateRecordingUi() {
  playbackStatus.textContent = `Recording… ${recordedFrames.length} frames captured.`;
}

function startPlayback() {
  if (!recordedFrames.length) return;
  isPlayingBack = true;
  playbackIndex = 0;
  playRecordingBtn.disabled = true;
  pausePlaybackBtn.disabled = false;
  clearRecordingBtn.disabled = true;
  recordToggleBtn.disabled = true;
  stopRecordingBtn.disabled = true;
  playbackSlider.disabled = false;
  playbackSlider.value = '0';
  videoElement.classList.add('video-hidden');
  playbackNotice.hidden = false;
  renderPlaybackFrame(recordedFrames[playbackIndex]);
  updatePlaybackStatus();
  scheduleNextPlaybackFrame();
}

function scheduleNextPlaybackFrame() {
  clearTimeout(playbackTimeoutId);
  if (!isPlayingBack) {
    return;
  }

  const nextIndex = playbackIndex + 1;
  if (nextIndex >= recordedFrames.length) {
    finishPlayback();
    return;
  }

  const currentTimestamp = recordedFrames[playbackIndex].timestamp;
  const nextTimestamp = recordedFrames[nextIndex].timestamp;
  const delay = Math.max(0, nextTimestamp - currentTimestamp);

  playbackTimeoutId = setTimeout(() => {
    if (!isPlayingBack) return;
    playbackIndex = nextIndex;
    renderPlaybackFrame(recordedFrames[playbackIndex]);
    playbackSlider.value = playbackIndex.toString();
    updatePlaybackStatus();
    scheduleNextPlaybackFrame();
  }, delay);
}

function renderPlaybackFrame(frame) {
  if (!frame) return;
  syncCanvasSize();
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.fillStyle = 'rgba(15, 23, 42, 0.9)';
  canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
  drawSkeleton(frame.keypoints, {
    lineColor: '#a855f7',
    pointColor: '#facc15',
    pointRadius: 5,
  });
  canvasCtx.restore();

  updateVisibleCount(countVisibleKeypoints(frame.keypoints));
  updateTable(frame.keypoints);
}

function updatePlaybackStatus() {
  if (!recordedFrames.length) {
    playbackStatus.textContent = 'No recording yet.';
    return;
  }
  playbackStatus.textContent = `Frame ${playbackIndex + 1} of ${recordedFrames.length}`;
}

function pausePlayback() {
  if (!isPlayingBack) return;
  isPlayingBack = false;
  playRecordingBtn.disabled = recordedFrames.length === 0;
  pausePlaybackBtn.disabled = true;
  clearRecordingBtn.disabled = recordedFrames.length === 0;
  recordToggleBtn.disabled = false;
  playbackNotice.hidden = true;
  videoElement.classList.remove('video-hidden');
  clearTimeout(playbackTimeoutId);
  playbackTimeoutId = null;
}

function finishPlayback() {
  pausePlayback();
  playbackStatus.textContent = 'Playback finished.';
  if (recordedFrames.length) {
    playbackIndex = recordedFrames.length - 1;
    playbackSlider.value = playbackIndex.toString();
    renderPlaybackFrame(recordedFrames[playbackIndex]);
  }
}

function clearRecording() {
  pausePlayback();
  recordedFrames.length = 0;
  playbackSlider.value = '0';
  playbackSlider.max = '0';
  playbackSlider.disabled = true;
  playRecordingBtn.disabled = true;
  clearRecordingBtn.disabled = true;
  playbackStatus.textContent = 'Recording cleared.';
}

init();

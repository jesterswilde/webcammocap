# Webcam Mocap Explorer

A Vite-powered web application that turns your webcam into a lightweight motion capture playground. It uses [TensorFlow.js BlazePose](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection) to detect 33 body landmarks in real time and provides tools to inspect and play back the captured points.

## Features

- 📸 **Live capture** – Overlay pose landmarks and connections directly on top of the webcam feed.
- 🔢 **Visibility metrics** – Track how many landmarks are confidently detected at any moment.
- 🏷️ **Landmark labels** – Inspect x/y/z coordinates, visibility, and labels for each of the 33 pose points.
- 💾 **Recording & playback** – Capture sequences of landmarks, scrub through them on a timeline slider, and replay the motion without the video feed.
- 🖥️ **Responsive UI** – A dark-themed dashboard with organized controls and tables for quick inspection.

## Getting started

```bash
npm install
npm run dev
```

Open the printed local URL (typically `http://localhost:5173`) and grant webcam permission when prompted.

## Building for production

```bash
npm run build
npm run preview
```

The `build` command creates an optimized bundle in the `dist/` directory. `npm run preview` serves the built files locally so you can verify the production output.

## Notes

- Recording stores only landmark coordinates—not the video itself—so playback renders the points over a stylized background.
- Playback speed matches the timing captured during recording. Large frame drops during capture will be reflected during playback.
- BlazePose weights are downloaded at runtime the first time the detector is created, so expect a short warm-up the first time you run the app.

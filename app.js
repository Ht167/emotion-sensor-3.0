/* ================================================================
   EMOTION DETECTION 2.0 — Frontend Logic
   Real-time webcam capture → WebSocket → CNN → UI update
   ================================================================ */

(() => {
    "use strict";

    // ──────────────────────────────────────────────────────
    // CONFIG
    // ──────────────────────────────────────────────────────
    const CAPTURE_FPS = 8;
    const JPEG_QUALITY = 0.65;
    const CAPTURE_WIDTH = 640;
    const CAPTURE_HEIGHT = 480;
    const RECONNECT_DELAY = 2000;
    const TIMELINE_MAX = 80;

    // ──────────────────────────────────────────────────────
    // EMOTION METADATA
    // ──────────────────────────────────────────────────────
    const EMOTIONS = {
        Angry:    { emoji: "😠", color: "#ef4444" },
        Disgust:  { emoji: "🤢", color: "#22c55e" },
        Fear:     { emoji: "😨", color: "#a855f7" },
        Happy:    { emoji: "😊", color: "#fbbf24" },
        Sad:      { emoji: "😢", color: "#3b82f6" },
        Surprise: { emoji: "😲", color: "#f97316" },
        Neutral:  { emoji: "😐", color: "#94a3b8" },
    };

    // ──────────────────────────────────────────────────────
    // DOM ELEMENTS
    // ──────────────────────────────────────────────────────
    const $ = (id) => document.getElementById(id);

    const video       = $("webcam");
    const overlay     = $("overlay");
    const ctx         = overlay.getContext("2d");
    const startBtn    = $("startBtn");
    const startOvr    = $("startOverlay");
    const videoCard   = $("videoCard");
    const statusBadge = $("statusBadge");
    const statusText  = $("statusText");
    const fpsDisplay  = $("fpsDisplay");
    const faceCount   = $("faceCount");
    const camToggle   = $("camToggle");
    const camIconOn   = $("camIconOn");
    const camIconOff  = $("camIconOff");
    const emotionEmoji   = $("emotionEmoji");
    const emotionLabel   = $("emotionLabel");
    const emotionConf    = $("emotionConfText");
    const ringFill       = $("ringFill");
    const totalFramesEl  = $("totalFrames");
    const totalFacesEl   = $("totalFaces");
    const avgConfEl      = $("avgConf");
    const timelineCanvas = $("timeline");
    const timelineCtx    = timelineCanvas.getContext("2d");

    // ──────────────────────────────────────────────────────
    // STATE
    // ──────────────────────────────────────────────────────
    let ws = null;
    let isProcessing = false;
    let captureInterval = null;
    let cameraActive = false;

    // Hidden capture canvas
    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = CAPTURE_WIDTH;
    captureCanvas.height = CAPTURE_HEIGHT;
    const captureCtx = captureCanvas.getContext("2d");

    // Stats
    let framesSent = 0;
    let facesTotal = 0;
    let confidenceSum = 0;
    let confidenceCount = 0;
    let lastEmotion = null;

    // FPS tracking
    let fpsFrames = 0;
    let fpsTime = performance.now();

    // Timeline history
    const timeline = [];

    // ──────────────────────────────────────────────────────
    // WEBSOCKET
    // ──────────────────────────────────────────────────────
    function connectWebSocket() {
        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${protocol}//${location.host}/ws`);

        ws.onopen = () => {
            setStatus("connected", "Connected");
        };

        ws.onclose = () => {
            setStatus("error", "Disconnected");
            setTimeout(connectWebSocket, RECONNECT_DELAY);
        };

        ws.onerror = () => {
            setStatus("error", "Connection Error");
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handlePrediction(data);
            } catch (e) {
                console.error("Parse error:", e);
            }
            isProcessing = false;
        };
    }

    function setStatus(state, text) {
        statusBadge.className = "status-badge " + state;
        statusText.textContent = text;
    }

    // ──────────────────────────────────────────────────────
    // CAMERA
    // ──────────────────────────────────────────────────────
    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: CAPTURE_WIDTH },
                    height: { ideal: CAPTURE_HEIGHT },
                    facingMode: "user",
                },
                audio: false,
            });

            video.srcObject = stream;
            await video.play();
            cameraActive = true;

            // Hide start overlay
            startOvr.classList.add("hidden");

            // Size overlay canvas to match displayed video
            resizeOverlay();
            window.addEventListener("resize", resizeOverlay);

            // Connect WS and start capture
            connectWebSocket();
            startCapture();

        } catch (err) {
            console.error("Camera error:", err);
            startBtn.querySelector("span").textContent = "Camera Blocked";
            setStatus("error", "Camera Access Denied");
        }
    }

    function stopCamera() {
        // Stop capture interval
        if (captureInterval) {
            clearInterval(captureInterval);
            captureInterval = null;
        }

        // Stop all video tracks
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(t => t.stop());
            video.srcObject = null;
        }

        // Close WebSocket
        if (ws) {
            ws.onclose = null; // prevent auto-reconnect
            ws.close();
            ws = null;
        }

        cameraActive = false;
        isProcessing = false;

        // Clear overlay
        clearOverlay();
        fpsDisplay.textContent = "— FPS";
        faceCount.textContent = "0 faces";
        setStatus("error", "Camera Off");

        // Show start overlay again
        startOvr.classList.remove("hidden");
    }

    function toggleCamera() {
        if (cameraActive) {
            stopCamera();
            camToggle.classList.add("off");
            camIconOn.classList.add("hidden");
            camIconOff.classList.remove("hidden");
        } else {
            startCamera();
            camToggle.classList.remove("off");
            camIconOff.classList.add("hidden");
            camIconOn.classList.remove("hidden");
        }
    }

    function resizeOverlay() {
        const rect = video.getBoundingClientRect();
        overlay.width = rect.width;
        overlay.height = rect.height;

        // Also resize timeline canvas
        timelineCanvas.width = timelineCanvas.parentElement.clientWidth - 32;
    }

    // ──────────────────────────────────────────────────────
    // FRAME CAPTURE LOOP
    // ──────────────────────────────────────────────────────
    function startCapture() {
        captureInterval = setInterval(() => {
            if (!ws || ws.readyState !== WebSocket.OPEN || isProcessing) return;

            // Draw mirrored frame to capture canvas
            captureCtx.save();
            captureCtx.translate(CAPTURE_WIDTH, 0);
            captureCtx.scale(-1, 1);
            captureCtx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
            captureCtx.restore();

            // Send as JPEG
            const dataUrl = captureCanvas.toDataURL("image/jpeg", JPEG_QUALITY);
            ws.send(dataUrl);
            isProcessing = true;
            framesSent++;

            // Update FPS counter
            fpsFrames++;
            const now = performance.now();
            if (now - fpsTime >= 1000) {
                fpsDisplay.textContent = fpsFrames + " FPS";
                fpsFrames = 0;
                fpsTime = now;
            }

            totalFramesEl.textContent = framesSent;

        }, 1000 / CAPTURE_FPS);
    }

    // ──────────────────────────────────────────────────────
    // HANDLE PREDICTIONS
    // ──────────────────────────────────────────────────────
    function handlePrediction(data) {
        const faces = data.faces || [];
        const fw = data.frameWidth || CAPTURE_WIDTH;
        const fh = data.frameHeight || CAPTURE_HEIGHT;

        // Draw face boxes on overlay
        drawOverlay(faces, fw, fh);

        // Update face count
        faceCount.textContent = faces.length + (faces.length === 1 ? " face" : " faces");
        facesTotal += faces.length;
        totalFacesEl.textContent = facesTotal;

        if (faces.length > 0) {
            // Use the first (largest/main) face
            const face = faces[0];
            updateEmotionUI(face);
            updateBars(face.probabilities);
            updateTimeline(face.emotion);

            // Stats
            confidenceSum += face.confidence;
            confidenceCount++;
            avgConfEl.textContent = Math.round((confidenceSum / confidenceCount) * 100) + "%";

            // Glow effect on video card
            setVideoGlow(face.emotion);
        } else {
            clearOverlay();
        }
    }

    // ──────────────────────────────────────────────────────
    // DRAW FACE OVERLAY
    // ──────────────────────────────────────────────────────
    function drawOverlay(faces, frameW, frameH) {
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        const scaleX = overlay.width / frameW;
        const scaleY = overlay.height / frameH;

        for (const face of faces) {
            // Mirror x-coordinate to match CSS-flipped video
            const x = overlay.width - (face.x * scaleX) - (face.w * scaleX);
            const y = face.y * scaleY;
            const w = face.w * scaleX;
            const h = face.h * scaleY;
            const color = EMOTIONS[face.emotion]?.color || "#fff";

            // Bounding box — rounded corners
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.lineJoin = "round";
            const r = 8;
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + w - r, y);
            ctx.quadraticCurveTo(x + w, y, x + w, y + r);
            ctx.lineTo(x + w, y + h - r);
            ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
            ctx.lineTo(x + r, y + h);
            ctx.quadraticCurveTo(x, y + h, x, y + h - r);
            ctx.lineTo(x, y + r);
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath();
            ctx.stroke();

            // Corner accents
            const cornerLen = 14;
            ctx.lineWidth = 3.5;
            ctx.strokeStyle = color;

            // Top-left
            ctx.beginPath();
            ctx.moveTo(x, y + cornerLen);
            ctx.lineTo(x, y);
            ctx.lineTo(x + cornerLen, y);
            ctx.stroke();

            // Top-right
            ctx.beginPath();
            ctx.moveTo(x + w - cornerLen, y);
            ctx.lineTo(x + w, y);
            ctx.lineTo(x + w, y + cornerLen);
            ctx.stroke();

            // Bottom-left
            ctx.beginPath();
            ctx.moveTo(x, y + h - cornerLen);
            ctx.lineTo(x, y + h);
            ctx.lineTo(x + cornerLen, y + h);
            ctx.stroke();

            // Bottom-right
            ctx.beginPath();
            ctx.moveTo(x + w - cornerLen, y + h);
            ctx.lineTo(x + w, y + h);
            ctx.lineTo(x + w, y + h - cornerLen);
            ctx.stroke();

            // Label background
            const label = `${face.emotion} ${Math.round(face.confidence * 100)}%`;
            ctx.font = "600 13px Inter, sans-serif";
            const textW = ctx.measureText(label).width;
            const labelH = 24;
            const labelY = y - labelH - 4;

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.roundRect(x, labelY, textW + 16, labelH, 6);
            ctx.fill();

            // Label text
            ctx.fillStyle = "#000";
            ctx.fillText(label, x + 8, labelY + 16);
        }
    }

    function clearOverlay() {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    // ──────────────────────────────────────────────────────
    // UPDATE EMOTION UI
    // ──────────────────────────────────────────────────────
    function updateEmotionUI(face) {
        const emo = EMOTIONS[face.emotion];
        if (!emo) return;

        // Emoji (bounce on change)
        if (lastEmotion !== face.emotion) {
            emotionEmoji.textContent = emo.emoji;
            emotionEmoji.classList.remove("bounce");
            void emotionEmoji.offsetWidth; // reflow
            emotionEmoji.classList.add("bounce");
            lastEmotion = face.emotion;
        }

        // Label & confidence
        emotionLabel.textContent = face.emotion;
        emotionLabel.style.color = emo.color;
        emotionConf.textContent = `${Math.round(face.confidence * 100)}% confidence`;

        // Confidence ring
        const circumference = 2 * Math.PI * 52; // r=52
        const offset = circumference * (1 - face.confidence);
        ringFill.style.strokeDashoffset = offset;
        ringFill.style.stroke = emo.color;
    }

    // ──────────────────────────────────────────────────────
    // UPDATE BARS
    // ──────────────────────────────────────────────────────
    function updateBars(probabilities) {
        if (!probabilities) return;

        // Find max emotion
        let maxEmotion = null;
        let maxProb = 0;
        for (const [emo, prob] of Object.entries(probabilities)) {
            if (prob > maxProb) { maxProb = prob; maxEmotion = emo; }
        }

        for (const [emo, prob] of Object.entries(probabilities)) {
            const barFill = $("bar-" + emo);
            const barVal  = $("val-" + emo);
            const barRow  = barFill?.closest(".bar-row");

            if (barFill) barFill.style.width = (prob * 100) + "%";
            if (barVal) barVal.textContent = Math.round(prob * 100) + "%";

            if (barRow) {
                barRow.classList.toggle("active", emo === maxEmotion);
            }
        }
    }

    // ──────────────────────────────────────────────────────
    // VIDEO GLOW
    // ──────────────────────────────────────────────────────
    function setVideoGlow(emotion) {
        // Remove all glow classes
        videoCard.className = videoCard.className
            .replace(/glow-\w+/g, "")
            .trim();
        videoCard.classList.add("glow-" + emotion.toLowerCase());
    }

    // ──────────────────────────────────────────────────────
    // TIMELINE
    // ──────────────────────────────────────────────────────
    function updateTimeline(emotion) {
        timeline.push(emotion);
        if (timeline.length > TIMELINE_MAX) timeline.shift();
        drawTimeline();
    }

    function drawTimeline() {
        const w = timelineCanvas.width;
        const h = timelineCanvas.height;
        timelineCtx.clearRect(0, 0, w, h);

        if (timeline.length < 2) return;

        const segW = w / (TIMELINE_MAX - 1);
        const dotR = 4;
        const y = h / 2;

        // Draw connecting line
        timelineCtx.strokeStyle = "rgba(255,255,255,0.06)";
        timelineCtx.lineWidth = 2;
        timelineCtx.beginPath();
        timelineCtx.moveTo(0, y);
        timelineCtx.lineTo(w, y);
        timelineCtx.stroke();

        // Draw dots
        for (let i = 0; i < timeline.length; i++) {
            const emo = EMOTIONS[timeline[i]];
            const x = (TIMELINE_MAX - timeline.length + i) * segW;

            // Glow
            timelineCtx.shadowBlur = 8;
            timelineCtx.shadowColor = emo.color;

            timelineCtx.fillStyle = emo.color;
            timelineCtx.beginPath();
            timelineCtx.arc(x, y, dotR, 0, Math.PI * 2);
            timelineCtx.fill();

            timelineCtx.shadowBlur = 0;
        }
    }

    // ──────────────────────────────────────────────────────
    // INIT
    // ──────────────────────────────────────────────────────
    startBtn.addEventListener("click", () => {
        startCamera();
        camToggle.classList.remove("off");
        camIconOff.classList.add("hidden");
        camIconOn.classList.remove("hidden");
    });
    camToggle.addEventListener("click", toggleCamera);

    // Check for roundRect support (polyfill for older browsers)
    if (!CanvasRenderingContext2D.prototype.roundRect) {
        CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
            if (typeof r === "number") r = [r, r, r, r];
            this.moveTo(x + r[0], y);
            this.lineTo(x + w - r[1], y);
            this.quadraticCurveTo(x + w, y, x + w, y + r[1]);
            this.lineTo(x + w, y + h - r[2]);
            this.quadraticCurveTo(x + w, y + h, x + w - r[2], y + h);
            this.lineTo(x + r[3], y + h);
            this.quadraticCurveTo(x, y + h, x, y + h - r[3]);
            this.lineTo(x, y + r[0]);
            this.quadraticCurveTo(x, y, x + r[0], y);
            this.closePath();
        };
    }

})();

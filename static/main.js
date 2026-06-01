// Web UI State Management
const STATE = {
    connected: false,
    cameraActive: false,
    recording: false,
    audioContext: null,
    mediaStream: null,
    recorder: null,
    audioChunks: [],
    cameraStream: null,
    snapshotBlob: null,
};

// UI Elements
const backendStatusDot = document.getElementById('backend-status-dot');
const backendStatusText = document.getElementById('backend-status-text');
const webcam = document.getElementById('webcam');
const btnToggleCamera = document.getElementById('btn-toggle-camera');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const cameraIndicator = document.getElementById('camera-indicator');
const btnSnapshot = document.getElementById('btn-snapshot');
const btnMic = document.getElementById('btn-mic');
const micStatusLabel = document.getElementById('mic-status-label');
const chatLogsContainer = document.getElementById('chat-logs-container');
const textInput = document.getElementById('text-input');
const btnSend = document.getElementById('btn-send');
const btnClearChat = document.getElementById('btn-clear-chat');
const ttsAudio = document.getElementById('tts-audio');
const chkAutoCapture = document.getElementById('chk-auto-capture');
const sliderSensitivity = document.getElementById('slider-sensitivity');

// Canvas visualizers
const visualizerCanvas = document.getElementById('visualizer-canvas');
const visCtx = visualizerCanvas.getContext('2d');
const snapshotCanvas = document.getElementById('snapshot-canvas');

// Sensitivity multiplier for voice visualizer
let sensitivity = 0.7;
sliderSensitivity.addEventListener('input', (e) => {
    sensitivity = e.target.value / 100;
});

// Fit Visualizer Canvas size
function resizeCanvas() {
    visualizerCanvas.width = visualizerCanvas.parentElement.clientWidth;
    visualizerCanvas.height = visualizerCanvas.parentElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Draw visualizer idle wave
let animationFrameId = null;
function drawVisualizer() {
    visCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
    visCtx.lineWidth = 2;
    
    if (STATE.recording) {
        // High activity wave when recording
        visCtx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
        visCtx.beginPath();
        const sliceWidth = visualizerCanvas.width / 100;
        let x = 0;
        
        for (let i = 0; i < 100; i++) {
            const val = (Math.sin(i * 0.15 + Date.now() * 0.01) * Math.cos(i * 0.05 + Date.now() * 0.005)) * 40 * sensitivity;
            const y = (visualizerCanvas.height / 2) + val;
            if (i === 0) visCtx.moveTo(x, y);
            else visCtx.lineTo(x, y);
            x += sliceWidth;
        }
        visCtx.stroke();
    } else {
        // Slow glowing pulse wave when idle
        visCtx.strokeStyle = 'rgba(139, 92, 246, 0.3)';
        visCtx.beginPath();
        const sliceWidth = visualizerCanvas.width / 50;
        let x = 0;
        for (let i = 0; i < 50; i++) {
            const val = Math.sin(i * 0.1 + Date.now() * 0.002) * 15;
            const y = (visualizerCanvas.height / 2) + val;
            if (i === 0) visCtx.moveTo(x, y);
            else visCtx.lineTo(x, y);
            x += sliceWidth;
        }
        visCtx.stroke();
    }
    
    animationFrameId = requestAnimationFrame(drawVisualizer);
}
drawVisualizer();

// 1. Connection check loop
async function checkBackendConnection() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.status === 'ready') {
            backendStatusDot.className = 'status-dot active';
            backendStatusText.textContent = 'Ready';
            STATE.connected = true;
        } else {
            backendStatusDot.className = 'status-dot pulsing';
            backendStatusText.textContent = 'Models Loading...';
            STATE.connected = false;
        }
    } catch (e) {
        backendStatusDot.className = 'status-dot';
        backendStatusText.textContent = 'Disconnected';
        STATE.connected = false;
    }
}
setInterval(checkBackendConnection, 5000);
checkBackendConnection();

// 2. Camera Integration
async function toggleCamera() {
    if (STATE.cameraActive) {
        // Stop Camera
        if (STATE.cameraStream) {
            STATE.cameraStream.getTracks().forEach(track => track.stop());
        }
        webcam.style.display = 'none';
        cameraPlaceholder.style.display = 'flex';
        cameraIndicator.textContent = 'Inoperative';
        cameraIndicator.classList.remove('active');
        btnToggleCamera.textContent = 'Enable Camera';
        btnSnapshot.disabled = true;
        STATE.cameraActive = false;
        STATE.cameraStream = null;
    } else {
        // Start Camera
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
                audio: false
            });
            webcam.srcObject = stream;
            webcam.style.display = 'block';
            cameraPlaceholder.style.display = 'none';
            cameraIndicator.textContent = 'Live';
            cameraIndicator.classList.add('active');
            btnToggleCamera.textContent = 'Disable Camera';
            btnSnapshot.disabled = false;
            STATE.cameraActive = true;
            STATE.cameraStream = stream;
        } catch (e) {
            console.error('Camera access denied or failed', e);
            alert('Unable to open camera: ' + e.message);
        }
    }
}
btnToggleCamera.addEventListener('click', toggleCamera);

// Capture current video frame as blob
function captureSnapshot() {
    if (!STATE.cameraActive) return null;
    
    snapshotCanvas.width = webcam.videoWidth;
    snapshotCanvas.height = webcam.videoHeight;
    const ctx = snapshotCanvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0, snapshotCanvas.width, snapshotCanvas.height);
    
    // Convert to blob and return
    return new Promise((resolve) => {
        snapshotCanvas.toBlob((blob) => {
            resolve(blob);
        }, 'image/jpeg', 0.85);
    });
}

btnSnapshot.addEventListener('click', async () => {
    const blob = await captureSnapshot();
    if (blob) {
        STATE.snapshotBlob = blob;
        // Temporary feedback
        btnSnapshot.textContent = 'Captured ✓';
        setTimeout(() => { btnSnapshot.textContent = 'Capture Snapshot'; }, 1500);
    }
});

// 3. Audio / Microphone Handling & WAV Encoder
async function startRecording() {
    STATE.audioChunks = [];
    try {
        STATE.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Setup AudioContext to obtain standard floats
        STATE.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = STATE.audioContext.createMediaStreamSource(STATE.mediaStream);
        
        // Processor node for saving chunks
        const bufferSize = 4096;
        STATE.recorder = STATE.audioContext.createScriptProcessor(bufferSize, 1, 1);
        
        STATE.recorder.onaudioprocess = (e) => {
            if (!STATE.recording) return;
            const inputData = e.inputBuffer.getChannelData(0);
            // Save a deep copy of the float array
            STATE.audioChunks.push(new Float32Array(inputData));
        };
        
        source.connect(STATE.recorder);
        STATE.recorder.connect(STATE.audioContext.destination);
        
        STATE.recording = true;
        btnMic.classList.add('recording');
        micStatusLabel.textContent = 'Listening... Tap to send';
    } catch (e) {
        console.error('Microphone access denied', e);
        alert('Could not start microphone: ' + e.message);
    }
}

function stopRecording() {
    if (!STATE.recording) return;
    
    STATE.recording = false;
    btnMic.classList.remove('recording');
    micStatusLabel.textContent = 'Processing voice...';
    
    // Stop recording processes
    if (STATE.recorder) {
        STATE.recorder.disconnect();
    }
    if (STATE.audioContext) {
        STATE.audioContext.close();
    }
    if (STATE.mediaStream) {
        STATE.mediaStream.getTracks().forEach(track => track.stop());
    }
    
    // Compile recorded chunks into one continuous array
    const fullBuffer = mergeBuffers(STATE.audioChunks);
    
    // Encode to 16-bit PCM WAV (Mono, default input sample rate, typically 44.1k/48k)
    const wavBlob = encodeWAV(fullBuffer, STATE.audioContext.sampleRate);
    
    // Send it to interaction pipeline
    handleInteraction(null, wavBlob);
}

function mergeBuffers(channelBuffer) {
    let result = new Float32Array(channelBuffer.reduce((acc, val) => acc + val.length, 0));
    let offset = 0;
    for (let i = 0; i < channelBuffer.length; i++) {
        result.set(channelBuffer[i], offset);
        offset += channelBuffer[i].length;
    }
    return result;
}

// standard 16-bit PCM WAV encoder
function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + samples.length * 2, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, 1, true);
    /* channel count (mono) */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, samples.length * 2, true);
    
    // Write floats normalized to 16-bit integers
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    
    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

btnMic.addEventListener('click', () => {
    if (STATE.recording) {
        stopRecording();
    } else {
        startRecording();
    }
});

// 4. API Interaction & Pipeline
async function handleInteraction(textQuery = null, audioBlob = null) {
    if (!STATE.connected) {
        alert("Backend not ready or disconnected. Please check status bar.");
        micStatusLabel.textContent = 'Tap Mic to Speak';
        return;
    }
    
    // Check if user is asking the camera to trigger
    let isCameraTriggered = false;
    const triggers = ["look", "see", "camera", "view", "what do you see"];
    
    if (textQuery) {
        const textLower = textQuery.toLowerCase();
        isCameraTriggered = triggers.some(t => textLower.includes(t));
    }
    
    // If using audio input, we let the backend transcribe first. We upload camera capture if Auto-Capture is on.
    const autoCaptureOn = chkAutoCapture.checked;
    let finalImageBlob = STATE.snapshotBlob;
    
    if (STATE.cameraActive && (autoCaptureOn || isCameraTriggered)) {
        finalImageBlob = await captureSnapshot();
    }
    
    // Build Form Payload
    const formData = new FormData();
    if (textQuery) formData.append('text', textQuery);
    if (audioBlob) formData.append('audio', audioBlob, 'mic.wav');
    if (finalImageBlob) formData.append('image', finalImageBlob, 'snapshot.jpg');
    
    // Append to Chat UI immediately as user bubble
    const userBubble = appendChatBubble('user', textQuery || '🎙️ (Voice Message...)');
    
    // Attach snapshot image preview in user chat bubble if sent
    if (finalImageBlob) {
        const imgUrl = URL.createObjectURL(finalImageBlob);
        const imgEl = document.createElement('img');
        imgEl.src = imgUrl;
        imgEl.className = 'chat-bubble-image';
        userBubble.querySelector('.bubble-content').prepend(imgEl);
    }
    
    // Clear snapshot state
    STATE.snapshotBlob = null;
    
    try {
        const res = await fetch('/api/interact', {
            method: 'POST',
            body: formData
        });
        
        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || 'Interaction failed');
        }
        
        const data = await res.json();
        
        // Update user bubble text if transcribed from audio
        if (audioBlob && data.user_text) {
            userBubble.querySelector('p').textContent = data.user_text;
        }
        
        // Append response
        appendChatBubble('assistant', data.assistant_response, `${data.elapsed_seconds}s`);
        
        // Play response voice if audio base64 is present
        if (data.audio) {
            playTtsAudio(data.audio);
        }
        
    } catch (e) {
        console.error(e);
        appendChatBubble('assistant', `⚠️ Error: ${e.message}`, 'System');
    } finally {
        micStatusLabel.textContent = 'Tap Mic to Speak';
    }
}

// 5. Chat UI Helpers
function appendChatBubble(role, text, timeStr = '') {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    
    const content = document.createElement('div');
    content.className = 'bubble-content';
    
    const textEl = document.createElement('p');
    textEl.innerHTML = text.replace(/\n/g, '<br>');
    content.appendChild(textEl);
    
    const time = document.createElement('span');
    time.className = 'bubble-time';
    time.textContent = timeStr || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    bubble.appendChild(content);
    bubble.appendChild(time);
    
    chatLogsContainer.appendChild(bubble);
    chatLogsContainer.scrollTop = chatLogsContainer.scrollHeight;
    
    return bubble;
}

function playTtsAudio(base64Data) {
    try {
        const audioUrl = `data:audio/wav;base64,${base64Data}`;
        ttsAudio.src = audioUrl;
        ttsAudio.play();
    } catch (e) {
        console.error('Audio playback error', e);
    }
}

// 6. Text input listeners
btnSend.addEventListener('click', () => {
    const text = textInput.value.trim();
    if (text) {
        handleInteraction(text);
        textInput.value = '';
    }
});

textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const text = textInput.value.trim();
        if (text) {
            handleInteraction(text);
            textInput.value = '';
        }
    }
});

btnClearChat.addEventListener('click', () => {
    chatLogsContainer.innerHTML = '';
});

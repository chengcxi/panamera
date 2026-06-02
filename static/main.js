// Web UI State Management
const STATE = {
    connected: false,
    running: false,
    pid: null,
    uptime: 0,
    lastLogIndex: 0,
    logs: []
};

// UI Elements
const backendStatusDot = document.getElementById('backend-status-dot');
const backendStatusText = document.getElementById('backend-status-text');
const engineIndicator = document.getElementById('engine-indicator');
const btnMic = document.getElementById('btn-mic');
const micStatusLabel = document.getElementById('mic-status-label');
const micToggleTitle = document.getElementById('mic-toggle-title');
const micToggleDesc = document.getElementById('mic-toggle-desc');
const deviceSelect = document.getElementById('device');
const logsContainer = document.getElementById('logs-container');
const btnClearLogs = document.getElementById('btn-clear-logs');
const settingsForm = document.getElementById('settings-form');

// Canvas visualizers
const visualizerCanvas = document.getElementById('visualizer-canvas');
const visCtx = visualizerCanvas.getContext('2d');

// Fit Visualizer Canvas size
function resizeCanvas() {
    visualizerCanvas.width = visualizerCanvas.parentElement.clientWidth;
    visualizerCanvas.height = visualizerCanvas.parentElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Draw visualizer wave animation
function drawVisualizer() {
    visCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
    visCtx.lineWidth = 2.5;
    
    if (STATE.running) {
        // High activity wave when running (simulating real-time voice processing)
        visCtx.strokeStyle = 'rgba(217, 70, 239, 0.7)'; // Magenta pulse
        visCtx.beginPath();
        const sliceWidth = visualizerCanvas.width / 100;
        let x = 0;
        
        for (let i = 0; i < 100; i++) {
            const val = (Math.sin(i * 0.15 + Date.now() * 0.012) * Math.cos(i * 0.04 + Date.now() * 0.006)) * 35;
            const y = (visualizerCanvas.height / 2) + val;
            if (i === 0) visCtx.moveTo(x, y);
            else visCtx.lineTo(x, y);
            x += sliceWidth;
        }
        visCtx.stroke();
        
        // Secondary subtle inner wave
        visCtx.strokeStyle = 'rgba(6, 182, 212, 0.4)'; // Cyan
        visCtx.lineWidth = 1.5;
        visCtx.beginPath();
        let x2 = 0;
        for (let i = 0; i < 100; i++) {
            const val = (Math.cos(i * 0.1 + Date.now() * 0.01) * Math.sin(i * 0.06 + Date.now() * 0.008)) * 20;
            const y = (visualizerCanvas.height / 2) + val;
            if (i === 0) visCtx.moveTo(x2, y);
            else visCtx.lineTo(x2, y);
            x2 += sliceWidth;
        }
        visCtx.stroke();
    } else {
        // Slow glowing pulse wave when idle
        visCtx.strokeStyle = 'rgba(139, 92, 246, 0.25)'; // Darker purple
        visCtx.beginPath();
        const sliceWidth = visualizerCanvas.width / 50;
        let x = 0;
        for (let i = 0; i < 50; i++) {
            const val = Math.sin(i * 0.12 + Date.now() * 0.002) * 12;
            const y = (visualizerCanvas.height / 2) + val;
            if (i === 0) visCtx.moveTo(x, y);
            else visCtx.lineTo(x, y);
            x += sliceWidth;
        }
        visCtx.stroke();
    }
    
    requestAnimationFrame(drawVisualizer);
}
drawVisualizer();

// Add lines to the terminal console
function appendLog(line) {
    const isSystem = line.includes('[SYSTEM]');
    const isError = line.includes('error:') || line.includes('[SYSTEM] Failed') || line.includes('STDERR');
    
    const lineEl = document.createElement('div');
    lineEl.className = 'log-line';
    if (isSystem) lineEl.classList.add('system');
    if (isError) lineEl.classList.add('error');
    
    // Simple timestamp
    const now = new Date();
    const ts = now.toTimeString().split(' ')[0] + '.' + String(now.getMilliseconds()).padStart(3, '0');
    
    lineEl.innerHTML = `<span class="log-time">${ts}</span> ${line}`;
    
    logsContainer.appendChild(lineEl);
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// 1. Connection and Status Loop
async function checkStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        STATE.connected = true;
        backendStatusDot.className = 'status-dot active';
        backendStatusText.textContent = 'Connected';

        const proc = data.process;
        updateProcessState(proc.running, proc.pid, proc.uptime);
    } catch (e) {
        STATE.connected = false;
        backendStatusDot.className = 'status-dot';
        backendStatusText.textContent = 'Disconnected';
        updateProcessState(false, null, 0);
    }
}

function updateProcessState(running, pid, uptime) {
    const prevRunning = STATE.running;
    STATE.running = running;
    STATE.pid = pid;
    STATE.uptime = uptime;

    if (running) {
        engineIndicator.textContent = `Active (PID: ${pid})`;
        engineIndicator.className = 'indicator active';
        btnMic.classList.add('recording');
        micStatusLabel.textContent = 'Assistant Active - Speaking/Listening';
        micToggleTitle.textContent = 'Assistant Online';
        micToggleDesc.textContent = `Uptime: ${Math.floor(uptime)}s. Tap to terminate.`;
    } else {
        engineIndicator.textContent = 'Stopped';
        engineIndicator.className = 'indicator';
        btnMic.classList.remove('recording');
        micStatusLabel.textContent = 'Assistant Stopped';
        micToggleTitle.textContent = 'Assistant Offline';
        micToggleDesc.textContent = 'Subprocess inactive. Tap to run.';
    }

    // Trigger log poll immediately if we just transitioned to running
    if (running && !prevRunning) {
        pollLogs();
    }
}

// 2. Poll Logs from backend
async function pollLogs() {
    if (!STATE.connected) return;
    try {
        const res = await fetch('/api/logs');
        const data = await res.json();
        const serverLogs = data.logs || [];
        
        // Show newly generated logs
        if (serverLogs.length > STATE.lastLogIndex) {
            for (let i = STATE.lastLogIndex; i < serverLogs.length; i++) {
                appendLog(serverLogs[i]);
            }
            STATE.lastLogIndex = serverLogs.length;
        } else if (serverLogs.length < STATE.lastLogIndex) {
            // Log buffer cleared or restarted on server
            logsContainer.innerHTML = '';
            STATE.lastLogIndex = 0;
            for (let i = 0; i < serverLogs.length; i++) {
                appendLog(serverLogs[i]);
            }
            STATE.lastLogIndex = serverLogs.length;
        }
    } catch (e) {
        console.error('Failed to poll logs', e);
    }
}

// 3. Audio Device Fetching
async function fetchDevices() {
    try {
        const res = await fetch('/api/devices');
        const data = await res.json();
        if (data.devices && data.devices.length > 0) {
            // Keep default option
            deviceSelect.innerHTML = '<option value="">Default System Device</option>';
            data.devices.forEach(dev => {
                const opt = document.createElement('option');
                opt.value = dev;
                opt.textContent = dev;
                deviceSelect.appendChild(opt);
            });
        }
    } catch (e) {
        console.error('Error fetching device list', e);
    }
}

// 4. Toggle Run State
async function toggleEngine() {
    if (!STATE.connected) {
        alert("Cannot interact: Backend API is disconnected.");
        return;
    }

    if (STATE.running) {
        // Stop
        appendLog("[SYSTEM] Stopping Assistant Subprocess...");
        try {
            const res = await fetch('/api/stop', { method: 'POST' });
            const data = await res.json();
            appendLog(`[SYSTEM] ${data.message}`);
            checkStatus();
        } catch (e) {
            appendLog(`[SYSTEM] Stop error: ${e.message}`);
        }
    } else {
        // Start
        appendLog("[SYSTEM] Starting Assistant Subprocess...");
        
        // Retrieve form parameters
        const formData = new FormData(settingsForm);
        const params = {
            model_root: formData.get('model_root') || null,
            model_path: formData.get('model_path') || null,
            voice: formData.get('voice') || null,
            prompt: formData.get('prompt') || null,
            temperature: parseFloat(formData.get('temperature')) || 0.8,
            threads: parseInt(formData.get('threads')) || null,
            quantize: formData.get('quantize') || null,
            context: parseInt(formData.get('context')) || null,
            device: formData.get('device') || null,
            gguf_caching: document.getElementById('gguf_caching').checked
        };

        try {
            const res = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            const data = await res.json();
            if (res.ok) {
                appendLog(`[SYSTEM] ${data.message}`);
                updateProcessState(data.status.running, data.status.pid, data.status.uptime);
            } else {
                appendLog(`[SYSTEM] Start error: ${data.detail}`);
            }
        } catch (e) {
            appendLog(`[SYSTEM] Start error: ${e.message}`);
        }
    }
}

// Event Listeners
btnMic.addEventListener('click', toggleEngine);
btnClearLogs.addEventListener('click', () => {
    logsContainer.innerHTML = '';
    appendLog("[SYSTEM] Log console cleared.");
});

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    fetchDevices();
    checkStatus();
    
    // Polling Intervals
    setInterval(checkStatus, 3000);
    setInterval(pollLogs, 1000);
});

import os
import subprocess
import threading
import queue
import time
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Panamera AI Assistant (Personaplex)")

# Path configurations
WORKSPACE_DIR = Path(__file__).parent
PERSONAPLEX_BINARY = WORKSPACE_DIR / "personaplex"

# Process state management
class ProcessManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.log_queue = queue.Queue(maxsize=1000)
        self.logs: List[str] = []
        self.logs_lock = threading.Lock()
        self.thread_stdout: Optional[threading.Thread] = None
        self.thread_stderr: Optional[threading.Thread] = None
        self.active_params: Dict = {}
        self.start_time: float = 0

    def add_log(self, line: str):
        with self.logs_lock:
            # Keep log buffer to a max of 500 lines
            if len(self.logs) >= 500:
                self.logs.pop(0)
            self.logs.append(line.strip())

    def _read_stream(self, stream, prefix: str):
        try:
            for line in iter(stream.readline, ''):
                if not line:
                    break
                self.add_log(f"[{prefix}] {line}")
        except Exception as e:
            self.add_log(f"[SYSTEM] Log reader error ({prefix}): {e}")
        finally:
            stream.close()

    def start(self, params: dict):
        if self.process and self.process.poll() is None:
            raise HTTPException(status_code=400, detail="Personaplex is already running.")

        # Build command arguments
        cmd = [str(PERSONAPLEX_BINARY)]

        if params.get("model_root"):
            cmd.extend(["-r", params["model_root"]])
        if params.get("model_path"):
            cmd.extend(["-m", params["model_path"]])
        if params.get("voice"):
            cmd.extend(["-v", params["voice"]])
        if params.get("prompt"):
            cmd.extend(["-p", params["prompt"]])
        if params.get("temperature") is not None:
            cmd.extend(["-t", str(params["temperature"])])
        if params.get("threads") is not None:
            cmd.extend(["--threads", str(params["threads"])])
        if params.get("quantize"):
            cmd.extend(["-q", params["quantize"]])
        if params.get("device"):
            cmd.extend(["-d", params["device"]])
        if params.get("gguf_caching"):
            cmd.append("-g")
        if params.get("context") is not None:
            cmd.extend(["-c", str(params["context"])])
        if params.get("seed") is not None:
            cmd.extend(["-s", str(params["seed"])])
        if params.get("delay") is not None:
            cmd.extend(["--delay", str(params["delay"])])
        if params.get("bench"):
            cmd.append("-b")

        # Clear previous log buffer
        with self.logs_lock:
            self.logs.clear()

        self.add_log(f"[SYSTEM] Starting command: {' '.join(cmd)}")

        # Start process with the active environment
        env = os.environ.copy()
        # Ensure LD_LIBRARY_PATH includes workspace dir for libmoshi.so
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{WORKSPACE_DIR}:{ld_path}" if ld_path else str(WORKSPACE_DIR)
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(WORKSPACE_DIR)
            )
            self.start_time = time.time()
            self.active_params = params

            # Start thread pools to consume stdout/stderr
            self.thread_stdout = threading.Thread(
                target=self._read_stream, 
                args=(self.process.stdout, "STDOUT"), 
                daemon=True
            )
            self.thread_stderr = threading.Thread(
                target=self._read_stream, 
                args=(self.process.stderr, "STDERR"), 
                daemon=True
            )
            
            self.thread_stdout.start()
            self.thread_stderr.start()
            
            self.add_log(f"[SYSTEM] Process started with PID: {self.process.pid}")
        except Exception as e:
            self.add_log(f"[SYSTEM] Failed to start process: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start Personaplex: {e}")

    def stop(self):
        if not self.process or self.process.poll() is not None:
            self.add_log("[SYSTEM] Stop requested, but no process is running.")
            return

        self.add_log(f"[SYSTEM] Terminating process {self.process.pid}...")
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
            self.add_log(f"[SYSTEM] Process terminated cleanly.")
        except subprocess.TimeoutExpired:
            self.add_log(f"[SYSTEM] Process did not terminate in 3 seconds, killing...")
            self.process.kill()
            self.process.wait()
            self.add_log(f"[SYSTEM] Process killed.")
        finally:
            self.process = None
            self.active_params = {}
            self.start_time = 0

    def get_status(self) -> dict:
        is_running = self.process is not None and self.process.poll() is None
        exit_code = self.process.poll() if self.process else None
        uptime = round(time.time() - self.start_time, 2) if is_running else 0
        return {
            "running": is_running,
            "pid": self.process.pid if is_running else None,
            "exit_code": exit_code,
            "uptime": uptime,
            "active_params": self.active_params
        }

    def get_logs(self) -> List[str]:
        with self.logs_lock:
            return list(self.logs)

manager = ProcessManager()

# Setup static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def read_root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Panamera Web UI</h1><p>Static directory is empty. Please add index.html</p>")

@app.get("/api/status")
def get_status():
    proc_status = manager.get_status()
    return {
        "status": "ready",
        "backend": "personaplex",
        "process": proc_status
    }

class StartParams(BaseModel):
    model_root: Optional[str] = None
    model_path: Optional[str] = None
    voice: Optional[str] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = 0.8
    threads: Optional[int] = None
    quantize: Optional[str] = None
    device: Optional[str] = None
    gguf_caching: Optional[bool] = False
    context: Optional[int] = None
    seed: Optional[int] = None
    delay: Optional[int] = None
    bench: Optional[bool] = False

@app.post("/api/start")
def start_backend(params: StartParams):
    param_dict = params.model_dump() if hasattr(params, 'model_dump') else params.dict()
    manager.start(param_dict)
    return {"message": "Personaplex started successfully", "status": manager.get_status()}

@app.post("/api/stop")
def stop_backend():
    manager.stop()
    return {"message": "Personaplex stopped successfully"}

@app.get("/api/logs")
def get_logs():
    return {"logs": manager.get_logs()}

@app.get("/api/devices")
def get_devices():
    """Run personaplex --list-devices to fetch list of host audio hardware."""
    try:
        env = os.environ.copy()
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{WORKSPACE_DIR}:{ld_path}" if ld_path else str(WORKSPACE_DIR)
        result = subprocess.run(
            [str(PERSONAPLEX_BINARY), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
            cwd=str(WORKSPACE_DIR)
        )
        # Parse stdout/stderr lines
        output = result.stdout or result.stderr
        lines = output.split("\n")
        devices = []
        for line in lines:
            line_str = line.strip()
            if line_str:
                devices.append(line_str)
        return {"devices": devices}
    except Exception as e:
        return {"devices": [], "error": str(e)}

@app.on_event("shutdown")
def shutdown_event():
    manager.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
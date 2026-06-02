import os
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# CONFIGURATION: Hardcoded Backend Target Options
WORKSPACE_DIR = Path(__file__).parent
PERSONAPLEX_BINARY = "/home/cheng/Downloads/moshi-bin-linux-x64-v0.8.0-beta/personaplex"

DEFAULT_VOICE = "/home/cheng/Downloads/AI/panamera-1/her.wav"
DEFAULT_PROMPT = "You are a personal assistant."

# PROCESS MANAGEMENT ENGINE
class AutomatedProcessManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.logs: List[str] = []
        self.logs_lock = threading.Lock()
        self.thread_stdout: Optional[threading.Thread] = None
        self.thread_stderr: Optional[threading.Thread] = None

    def add_log(self, line: str):
        with self.logs_lock:
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

    def launch_backend(self):
        """Spins up Personaplex automatically using hardcoded settings."""
        if self.process and self.process.poll() is None:
            print("[SYSTEM] Personaplex is already running.")
            return

        # Direct hardcoded command payload execution
        cmd = [
            str(PERSONAPLEX_BINARY),
            "-v", DEFAULT_VOICE,
            "-p", DEFAULT_PROMPT
        ]

        self.add_log(f"[SYSTEM] Auto-launching command: {' '.join(cmd)}")

        env = os.environ.copy()
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
            
            print(f"🚀 [SYSTEM] Personaplex booted automatically! PID: {self.process.pid}")
            self.add_log(f"[SYSTEM] Process started automatically with PID: {self.process.pid}")
        except Exception as e:
            print(f"❌ [SYSTEM] Auto-boot execution failure: {e}")
            self.add_log(f"[SYSTEM] Failed to start process: {e}")

    def terminate_backend(self):
        """Cleans up the child sub-process on server stop."""
        if not self.process or self.process.poll() is not None:
            return

        print(f"Stopping Personaplex sub-process [PID {self.process.pid}]...")
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        finally:
            self.process = None

manager = AutomatedProcessManager()

# FASTAPI LIFESPAN CONTROLLER (Handles Setup and Teardown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs exactly when Uvicorn boots up
    manager.launch_backend()
    yield
    # This runs exactly when you shut down the terminal server
    manager.terminate_backend()

app = FastAPI(title="Panamera AI Assistant (Personaplex)", lifespan=lifespan)

# Setup static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# SIMPLIFIED WEB INTERFACES
@app.get("/")
def read_root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Panamera Web UI</h1><p>Static directory is empty.</p>")

@app.get("/api/status")
def get_status():
    is_running = manager.process is not None and manager.process.poll() is None
    return {
        "status": "ready",
        "backend": "personaplex",
        "backend_running": is_running,
        "pid": manager.process.pid if is_running else None
    }

@app.get("/api/logs")
def get_logs():
    return {"logs": manager.get_logs()}

if __name__ == "__main__":
    import uvicorn
    # Booting the app locally
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""
Startup script: installs dependencies, then runs FastAPI + Gradio together.
Run with:  python startup.py
"""

import os
import sys
import time
import logging
import subprocess
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [startup] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("startup")

DEPENDENCIES = [
    "fastapi",
    "uvicorn[standard]",
    "gradio",
    "python-dotenv",
    "pydantic>=2.0",
    "langchain",
    "langchain-core",
    "langchain-google-genai",
    "pypdf",
    "python-multipart",
    "requests",
]


def _stream(proc, name):
    """Forward a subprocess's stdout into our logger, tagged by name."""
    for line in iter(proc.stdout.readline, b""):
        try:
            text = line.decode("utf-8", errors="replace").rstrip()
        except Exception:
            text = str(line)
        if text:
            log.info("[%s] %s", name, text)
    proc.stdout.close()


def install_dependencies():
    log.info("Installing dependencies (this can take a minute)...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *DEPENDENCIES]
        )
    except subprocess.CalledProcessError as e:
        log.error("Dependency install failed: %s", e)
        sys.exit(1)
    log.info("Dependencies installed.")


def check_env():
    log.info("Checking environment...")
    if not os.path.exists(".env"):
        log.warning("No .env file found.")
        log.warning("Create one from .env.example and add your GEMINI_API_KEY:")
        log.warning("  1. Copy .env.example to .env")
        log.warning("  2. Open .env and paste your key after GEMINI_API_KEY=")
        choice = input("Continue anyway? (y/N): ").strip().lower()
        if choice != "y":
            log.info("Aborted by user.")
            sys.exit(1)
    else:
        log.info(".env found.")
    log.info("Python: %s", sys.version.split()[0])
    log.info("Executable: %s", sys.executable)
    log.info("Working dir: %s", os.getcwd())


def wait_for_backend(host="127.0.0.1", port=8000, timeout=30):
    """Poll the backend until it accepts connections (or give up)."""
    import socket
    log.info("Waiting for backend to accept connections on %s:%d ...", host, port)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                log.info("Backend is up on %s:%d", host, port)
                return True
        except OSError:
            time.sleep(0.5)
    log.warning("Backend did not respond within %ds — starting frontend anyway.", timeout)
    return False


def main():
    log.info("=== Multi-Agent Resume Screener — startup ===")
    check_env()
    install_dependencies()

    log.info("Starting FastAPI backend at http://127.0.0.1:8000 ...")
    backend = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app:app",
            "--host", "127.0.0.1",
            "--port", "8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.info("Backend PID: %d", backend.pid)
    threading.Thread(target=_stream, args=(backend, "backend"), daemon=True).start()

    wait_for_backend()

    log.info("Starting Gradio frontend at http://127.0.0.1:7860 ...")
    log.info("=" * 60)
    log.info("  Open in your browser:  http://127.0.0.1:7860")
    log.info("  Press Ctrl+C to stop both servers.")
    log.info("=" * 60)

    frontend = None
    try:
        frontend = subprocess.Popen(
            [sys.executable, "frontend.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.info("Frontend PID: %d", frontend.pid)
        threading.Thread(target=_stream, args=(frontend, "frontend"), daemon=True).start()
        frontend.wait()
        log.info("Frontend exited with code %s", frontend.returncode)
    except KeyboardInterrupt:
        log.info("Ctrl+C received — shutting down...")
    finally:
        for proc, name in [(frontend, "frontend"), (backend, "backend")]:
            if proc and proc.poll() is None:
                log.info("Terminating %s (PID %d)...", name, proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    log.info("%s stopped.", name)
                except subprocess.TimeoutExpired:
                    log.warning("%s did not stop in time — killing.", name)
                    proc.kill()
        log.info("Done.")


if __name__ == "__main__":
    main()

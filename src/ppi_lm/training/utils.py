"""Small helpers shared by every training run: config, seed, device, run directory,
memory monitoring, keeping the Mac awake and launching TensorBoard."""

import importlib.util
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import warnings
import webbrowser
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def set_seed(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch (CPU and every GPU backend)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """auto -> cuda, then mps, then cpu. Any other name is used as is."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(output_dir: str | Path, name: str) -> Path:
    """Creates output_dir/<name>_<YYYYmmdd-HHMMSS>/, never reusing an existing directory."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{name}_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = Path(output_dir) / f"{name}_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


# --- memory ---------------------------------------------------------------------

GB = 1024**3


def memory_stats(device: torch.device) -> dict[str, float]:
    """Memory in GB: the Python process (RSS) and the GPU backend in use.

    mps_allocated: tensors alive; mps_driver: everything the Metal driver holds
    (with unified memory, this is RAM taken from the rest of the Mac).
    """
    stats = {}
    if importlib.util.find_spec("psutil") is not None:
        import psutil

        stats["process_rss_gb"] = psutil.Process().memory_info().rss / GB
        stats["system_memory_used_percent"] = psutil.virtual_memory().percent
    if device.type == "mps":
        stats["mps_allocated_gb"] = torch.mps.current_allocated_memory() / GB
        stats["mps_driver_gb"] = torch.mps.driver_allocated_memory() / GB
    elif device.type == "cuda":
        stats["cuda_allocated_gb"] = torch.cuda.memory_allocated(device) / GB
        stats["cuda_max_allocated_gb"] = torch.cuda.max_memory_allocated(device) / GB
    return stats


# --- keep the Mac awake ---------------------------------------------------------


@contextmanager
def keep_awake(enabled: bool = True):
    """Prevents macOS from sleeping while the block runs, with `caffeinate`.

    -i: no idle sleep, -m: no disk sleep, -s: no system sleep (on AC power only);
    the display may still turn off. -w <pid>: caffeinate also stops by itself if
    Python dies. Closing the lid still puts a MacBook to sleep.
    """
    process = None
    if enabled:
        if shutil.which("caffeinate") is None:
            warnings.warn("caffeinate not found (macOS only): the machine may sleep", stacklevel=3)
        else:
            process = subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
            print(f"caffeinate: the Mac stays awake during the run (pid {process.pid})")
    try:
        yield
    finally:
        if process is not None:
            process.terminate()
            process.wait()


# --- TensorBoard ----------------------------------------------------------------


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _tensorboard_logdir(port: int) -> str | None:
    """Directory watched by the TensorBoard already listening on port (best effort)."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/data/environment", timeout=1) as r:
            return json.loads(r.read()).get("data_location")
    except (OSError, ValueError):
        return None


def launch_tensorboard(
    logdir: str | Path,
    port: int = 6006,
    open_browser: bool = True,
    wait_s: float = 15.0,
    max_ports: int = 20,
) -> str | None:
    """Starts TensorBoard on logdir in the background and opens it in the browser.

    A TensorBoard already watching this exact logdir (absolute path) is reused: one
    TensorBoard on the parent `runs` directory shows every run. A port taken by anything
    else (e.g. a TensorBoard of another project) is skipped: the next free port is used.
    The started TensorBoard keeps running after the run ends, so the curves stay
    available; its output goes to logdir/tensorboard.log.

    returns: the URL, or None if tensorboard is not installed or no port is free
    """
    if importlib.util.find_spec("tensorboard") is None:
        warnings.warn("tensorboard is not installed: not launched", stacklevel=2)
        return None
    logdir = Path(logdir).resolve()

    for candidate in range(port, port + max_ports):
        url = f"http://localhost:{candidate}/"
        if not port_in_use(candidate):
            break
        if _tensorboard_logdir(candidate) == str(logdir):
            print(f"TensorBoard already running: {url} (watching {logdir})")
            return url
    else:
        warnings.warn(f"No free port in {port}-{port + max_ports - 1}", stacklevel=2)
        return None

    logdir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir)]
    with open(logdir / "tensorboard.log", "a") as log_file:
        subprocess.Popen(
            [*command, "--port", str(candidate)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # survives Ctrl+C on the training and the end of the run
        )
    deadline = time.monotonic() + wait_s
    while not port_in_use(candidate) and time.monotonic() < deadline:
        time.sleep(0.2)
    print(f"TensorBoard: {url} (watching {logdir})")
    if open_browser:
        webbrowser.open(url)
    return url

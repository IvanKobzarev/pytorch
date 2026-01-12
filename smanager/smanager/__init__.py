# smanager - Slurm job management web UI
import subprocess
from pathlib import Path

def _get_version():
    # Try to get version from git (works in dev/git checkout)
    try:
        count = subprocess.check_output(
            ['git', 'rev-list', '--count', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent
        ).strip()
        return f"0.{count}"
    except Exception:
        pass
    # Fallback to _version.py (works for installed package)
    try:
        from smanager._version import __version__ as v
        return v
    except Exception:
        return "0.dev"

__version__ = _get_version()

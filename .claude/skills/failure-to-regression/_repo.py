"""Repo paths shared by this skill's scripts (importing it puts the repo on sys.path)."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main_checkout():
    """The main worktree, which holds the gitignored data/, .env and .streamlit/secrets.toml (REPO if unknown)."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "worktree", "list", "--porcelain"],
                             capture_output=True, text=True, check=True).stdout
        return Path(out.splitlines()[0].split(" ", 1)[1])
    except (OSError, subprocess.CalledProcessError, IndexError):
        return REPO


def rel(path):
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path)

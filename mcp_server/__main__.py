"""Serve the MCP tools over stdio.

    python -m mcp_server                          # from the repo root
    python C:\\path\\to\\Clinical-Diagnostics-\\mcp_server   # from any directory (what MCP clients run)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Run as a directory, Python puts this package's own folder first on sys.path, where paths.py or
# schemas.py could shadow other modules: replace it with the repo root (agent/, services/, evals/).
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(HERE.parent))

from mcp_server.server import main  # noqa: E402

main()

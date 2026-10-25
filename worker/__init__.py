"""Report worker process (see worker/report_worker.py)."""
import matplotlib

# Same reason as api/__init__.py: no display, so never let matplotlib pick a GUI backend
matplotlib.use("Agg")

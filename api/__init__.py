"""FastAPI backend for the React UI (see api/main.py)."""
import matplotlib

# Grad-CAM figures are drawn in worker threads with no display. Outside Streamlit, matplotlib
# would otherwise pick a GUI backend (Tk on Windows), which fails off the main thread.
matplotlib.use("Agg")

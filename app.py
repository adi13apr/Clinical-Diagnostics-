# app.py
import streamlit as st
from services.auth_handler import AuthService
from services.data_handler import DataService
from services.model_handler import ModelService
from services.prediction_handler import PredictionService
from services.tracing import SupabaseTraceStore
from ui.login_pages import show_login_ui
from ui.dashboard import show_dashboard  # <--- This is the important part

# 1. Initialize the background workers
# Keep one auth client per browser session so its login survives reruns (RLS needs it)
if "auth_provider" not in st.session_state:
    st.session_state.auth_provider = AuthService()
auth_provider = st.session_state.auth_provider
prediction_provider = PredictionService(auth_provider.client)
trace_store = SupabaseTraceStore(auth_provider.client)  # same logged-in client, so RLS applies
data_provider = DataService()
model_provider = ModelService("best_student_model_aligned_final.pth")

# 2. Check if anyone is logged in
if "user" not in st.session_state:
    st.session_state.user = None

# 3. Direct the traffic
if not st.session_state.user:
    # Go to the login file
    show_login_ui(auth_provider)
else:
    # Go to the dashboard file and bring the workers along
    show_dashboard(
        st.session_state.user.email, 
        auth_provider, 
        data_provider, 
        model_provider,
        prediction_provider,
        trace_store
    )
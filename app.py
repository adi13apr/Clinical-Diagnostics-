# app.py
import streamlit as st
from services.auth_handler import AuthService
from services.data_handler import DataService
from services.model_handler import ModelService
from ui.login_pages import show_login_ui
from ui.dashboard import show_dashboard  # <--- This is the important part

# 1. Initialize the background workers
auth_provider = AuthService()
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
        model_provider
    )
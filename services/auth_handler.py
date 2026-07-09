import streamlit as st
from supabase import create_client

class AuthService:
    def __init__(self):
        self.client = create_client(
            st.secrets["supabase"]["url"], 
            st.secrets["supabase"]["key"]
        )

    def login_email(self, email, password):
        return self.client.auth.sign_in_with_password({"email": email, "password": password})

    def sign_up(self, email, password):
        return self.client.auth.sign_up({"email": email, "password": password})

    def login_google(self):
        return self.client.auth.sign_in_with_oauth({"provider": "google"})

    def logout(self):
        self.client.auth.sign_out()
        st.session_state.user = None
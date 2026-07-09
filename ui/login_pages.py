import streamlit as st

def show_login_ui(auth_service):
    st.title("🫁 Clinical Diagnostic Portal")
    mode = st.radio("Access Level", ["Login", "Create New Account"])
    
    with st.container(border=True):
        email = st.text_input("Email Address")
        pwd = st.text_input("Password", type="password")
        
        if mode == "Login":
            if st.button("Log In"):
                try:
                    res = auth_service.login_email(email, pwd)
                    st.session_state.user = res.user
                    st.rerun()
                except:
                    st.error("Invalid credentials")
            
            st.write("--- or ---")
            if st.button("Continue with Google"):
                auth_service.login_google()
        else:
            if st.button("Register Account"):
                try:
                    auth_service.sign_up(email, pwd)
                    st.success("Verification email sent! Check your inbox.")
                except Exception as e:
                    if "already registered" in str(e).lower():
                        st.error("This email is already registered. Please log in instead.")
                    else:
                        st.error(f"Registration failed: {str(e)}")
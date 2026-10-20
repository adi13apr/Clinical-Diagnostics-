import time

import streamlit as st
from services.auth_handler import OAUTH_STATE_TTL_S, OAuthStateError, strip_query

def app_base_url():
    """Where Google should send the browser back: [supabase] redirect_url, else this page's URL (or None)."""
    try:
        configured = st.secrets["supabase"].get("redirect_url")
    except Exception:
        configured = None
    url = configured or st.context.url  # st.context.url is None when Streamlit doesn't know the page address
    return strip_query(url) if url else None

def google_sign_in_url(auth_service):
    """Sign-in URL reused across reruns, refreshed before its state expires (None if the app URL is unknown)"""
    cached = st.session_state.get("google_oauth")
    if cached is None or time.monotonic() - cached["created"] > OAUTH_STATE_TTL_S - 60:
        base = app_base_url()
        if base is None:
            return None
        cached = {"url": auth_service.google_sign_in_url(base), "created": time.monotonic()}
        st.session_state["google_oauth"] = cached
    return cached["url"]

def handle_oauth_callback(auth_service):
    """Finish a Google sign-in when Supabase redirects back with ?code=...&oauth_state=..."""
    params = st.query_params
    if "error" in params or "error_description" in params:
        st.error(f"Google sign-in failed: {params.get('error_description') or params.get('error')}")
        st.query_params.clear()
        return
    code, state = params.get("code"), params.get("oauth_state")
    if not code:
        return
    st.query_params.clear()  # the code is single-use; don't leave it in the address bar
    if not state:
        st.error("Google sign-in came back without its sign-in state. In Supabase, add this app's URL with "
                 "/** to Authentication → URL Configuration → Redirect URLs, then try again.")
        return
    try:
        user = auth_service.complete_google_sign_in(code, state)
    except OAuthStateError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Google sign-in failed: {str(e)}")
        return
    st.session_state.user = user
    st.session_state.pop("google_oauth", None)
    st.rerun()

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
            # A real link: the server can't redirect the browser, so the button must point at Google itself
            url = google_sign_in_url(auth_service)
            if url:
                st.link_button("Continue with Google", url)
                st.caption("Opens Google sign-in; you'll come back to the portal signed in.")
            else:
                st.button("Continue with Google", disabled=True)
                st.caption("Google sign-in is unavailable: set [supabase] redirect_url in the app secrets.")
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

    from ui.privacy import PRIVACY_PATH
    st.markdown(f"[Privacy policy]({PRIVACY_PATH})")
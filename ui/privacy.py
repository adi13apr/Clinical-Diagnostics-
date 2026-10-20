import streamlit as st

PRIVACY_PATH = "?page=privacy"
CONTACT_URL = "https://github.com/adi13apr/Clinical-Diagnostics-/issues"

def show_privacy_page():
    """Public privacy policy (linked from Google's sign-in screen); needs no login"""
    st.title("Privacy policy")
    st.caption("Clinical Diagnostic Portal · last updated 20 October 2026")
    st.markdown(f"""
This portal is a research and portfolio project, **not a medical device**. Its results are a screening
aid, not a diagnosis.

**What we collect when you sign in**
- Your email address, and if you use Google sign-in, the basic profile Google shares (name, email,
  profile picture). Sign-in is handled by Supabase; the app never sees your Google password.

**What we store when you use the app**
- *Diagnosis history:* the patient ID you type, the AI prediction, its confidence and triage status,
  and the time. The app shows each user only their own records.
- *Prediction log and traces:* the predicted label, confidence, triage status, processing times and
  technical step timings, linked to your account. Patient IDs are not part of this log.
- Uploaded X-ray images are processed in memory to make the prediction. They are **not stored**.

**Who else processes data**
- **Supabase** stores the account and records above.
- **Groq** (an AI provider) receives the classifier's output, a summary of where the model looked, and
  standard condition text when you click "Generate report". It never receives the image, the patient ID
  or your email.
- **Streamlit Community Cloud** hosts the app.

**What we don't do**
- We don't sell your data, use it for advertising, or share it with anyone else.

**Keeping and deleting data**
- Records are kept until you ask for them to be deleted. To request deletion, or for any question about
  this policy, open an issue at [{CONTACT_URL}]({CONTACT_URL}).
""")
    st.link_button("Back to the portal", "./")

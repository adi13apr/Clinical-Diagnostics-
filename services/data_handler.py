from supabase import create_client
import streamlit as st

class DataService:
    def __init__(self):
        self.client = create_client(
            st.secrets["supabase"]["url"], 
            st.secrets["supabase"]["key"]
        )

    def save_diagnosis(self, email, result, patient_name):
        # Handle both old format (string) and new format (dict with diagnosis + confidence)
        if isinstance(result, dict):
            diagnosis = result['diagnosis']
            confidence = result['confidence']
        else:
            diagnosis = result
            confidence = None
            
        data = {
            "radiologist_email": email,
            "prediction": diagnosis,
            "patient_name": patient_name
        }
        
        # Add confidence if available
        if confidence is not None:
            data["confidence_score"] = round(confidence * 100, 2)
            
        return self.client.table("diagnosis_history").insert(data).execute()

    def get_history(self, email):
        return self.client.table("diagnosis_history").select("*").eq("radiologist_email", email).order("created_at", desc=True).execute()
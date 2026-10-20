class DataService:
    def __init__(self, client):
        # Must be the logged-in auth client: RLS (sql/005) only shows a user their own rows
        self.client = client

    def save_diagnosis(self, email, result, patient_name):
        # Handle both old format (string) and new format (dict with diagnosis + confidence)
        if isinstance(result, dict):
            diagnosis = result['diagnosis']
            confidence = result['confidence']
            status = result.get('status')
        else:
            diagnosis = result
            confidence = None
            status = None
            
        data = {
            "radiologist_email": email,
            "prediction": diagnosis,
            "patient_name": patient_name
        }
        
        # Add confidence if available
        if confidence is not None:
            data["confidence_score"] = round(confidence * 100, 2)

        # Triage status (needs sql/003_diagnosis_history_status.sql)
        if status is not None:
            data["status"] = status

        return self.client.table("diagnosis_history").insert(data).execute()

    def get_history(self, email):
        return self.client.table("diagnosis_history").select("*").eq("radiologist_email", email).order("created_at", desc=True).execute()
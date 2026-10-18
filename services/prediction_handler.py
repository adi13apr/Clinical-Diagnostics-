class PredictionService:
    def __init__(self, client):
        # Must be the logged-in auth client so RLS sees the user's session
        self.client = client

    def log_prediction(self, result, inference_ms):
        data = {
            "predicted_label": result['diagnosis'],
            "confidence": result['confidence'],
            "inference_ms": inference_ms
        }
        return self.client.table("predictions").insert(data).execute()

    def get_daily_summary(self):
        return self.client.table("prediction_daily_summary").select("*").order("day", desc=True).execute()

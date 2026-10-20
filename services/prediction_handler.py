class PredictionService:
    def __init__(self, client):
        # Must be the logged-in auth client so RLS sees the user's session
        self.client = client

    def log_prediction(self, result, inference_ms, trace_id=None):
        data = {
            "predicted_label": result['diagnosis'],
            "confidence": result['confidence'],
            "inference_ms": inference_ms,
            "status": result['status'],
            "temperature": result['temperature']
        }
        # Links this row to its spans in the `traces` table (needs sql/004_traces.sql)
        if trace_id is not None:
            data["trace_id"] = trace_id
        return self.client.table("predictions").insert(data).execute()

    def get_daily_summary(self):
        return self.client.table("prediction_daily_summary").select("*").order("day", desc=True).execute()

    def get_history(self, user_id, limit=20, offset=0, label=None, status=None, since=None, until=None):
        """One page of the user's logged predictions, newest first; `.count` holds the total.

        since/until are ISO timestamps (until is exclusive). RLS already limits rows to the
        session's user; filtering on user_id too keeps that true if a policy is ever loosened.
        """
        query = self.client.table("predictions").select("*", count="exact").eq("user_id", user_id)
        if label is not None:
            query = query.eq("predicted_label", label)
        if status is not None:
            query = query.eq("status", status)
        if since is not None:
            query = query.gte("created_at", since)
        if until is not None:
            query = query.lt("created_at", until)
        return query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()

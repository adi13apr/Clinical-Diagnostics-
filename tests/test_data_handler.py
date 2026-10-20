from services.data_handler import DataService


class FakeTable:
    def __init__(self, calls):
        self.calls = calls

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self
        return record


class FakeClient:
    def __init__(self):
        self.calls = []

    def table(self, name):
        self.calls.append(("table", (name,), {}))
        return FakeTable(self.calls)


def test_uses_the_given_logged_in_client():
    # The logged-in client carries the user's JWT, which the sql/005 RLS policies need
    client = FakeClient()
    service = DataService(client)
    service.save_diagnosis("dr@example.com", {"diagnosis": "Pneumonia", "confidence": 0.9731, "status": "confident"}, "P1")
    service.get_history("dr@example.com")

    assert service.client is client
    insert = next(c for c in client.calls if c[0] == "insert")
    assert insert[1][0] == {"radiologist_email": "dr@example.com", "prediction": "Pneumonia",
                            "patient_name": "P1", "confidence_score": 97.31, "status": "confident"}
    assert ("eq", ("radiologist_email", "dr@example.com"), {}) in client.calls

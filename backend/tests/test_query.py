"""
اختبارات أساسية لـ /query: حالة سليمة (happy path) وحالة مدخل غير صالح (422).
تستخدم mocking لخدمتَي الاسترجاع والتوليد حتى لا تحتاج vector store حقيقي
أو اتصال بـ Ollama أثناء الاختبار.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with patch("app.services.retrieval.retrieval_service.is_ready", return_value=True), \
         patch(
             "app.services.retrieval.retrieval_service.retrieve",
             return_value=[
                 {
                     "chunk_id": "01_activation_functions_0",
                     "text": "ReLU تُخرج القيمة نفسها إذا كانت موجبة، وصفر إذا كانت سالبة.",
                     "source": "01_activation_functions.md",
                     "distance": 0.12,
                 }
             ],
         ), \
         patch(
             "app.api.routes.query.generate_answer",
             return_value="دالة ReLU تُخرج max(0, x). [المصدر: 01_activation_functions.md]",
         ):
        yield TestClient(app)


def test_query_happy_path(client):
    response = client.post("/query", json={"question": "إيه هي دالة ReLU؟"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert "01_activation_functions.md" in body["sources"]


def test_query_invalid_input_returns_422(client):
    # question فاضية غير مسموح بها (min_length=1)
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

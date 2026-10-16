# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Data Science Academy Pvt. Ltd.
"""Tests for persona-aware shopping recommendations and API handling."""

from pathlib import Path

from fastapi.testclient import TestClient

import agents.recommend as recommend
import backend.main as backend
from frontend.app import get_bot_avatar, query_chat_api


QUERY = "Suggest a good smartphone under ₹25,000."


def sample_product() -> dict:
    """Return one realistic, mocked phone result for deterministic tests."""
    return {
        "product_name": "Example Phone X",
        "brand": "Example",
        "store": "Store A",
        "price": 21999,
        "mrp": 24999,
        "rating": 4.4,
        "review_count": 120,
        "key_specs": "Processor: X1 | RAM: 8GB | Storage: 128GB | Battery: 5000mAh",
        "pros": "Bright display",
        "cons": "No charger included",
        "score": 87.0,
    }


def test_same_query_passes_each_personality_into_existing_llm_prompt(monkeypatch):
    """The API's chosen persona changes instructions sent through existing LLM flow."""
    product = sample_product()
    monkeypatch.setattr(backend, "search_products", lambda **kwargs: [product])
    monkeypatch.setattr(backend, "compare_and_rank", lambda *args, **kwargs: [product])
    monkeypatch.setattr(recommend.config, "LLM_PROVIDER", "gemini")
    prompts = []

    def mock_llm(prompt: str) -> str:
        prompts.append(prompt)
        return "mocked recommendation"

    monkeypatch.setattr(recommend, "_call_gemini", mock_llm)
    client = TestClient(backend.app)
    personalities = ["Tech Geek", "Bargain Hunter", "Everyday User"]

    for personality in personalities:
        response = client.post(
            "/chat", json={"message": QUERY, "personality": personality}
        )
        assert response.status_code == 200
        assert response.json()["recommendation"] == "mocked recommendation"

    assert len(prompts) == 3
    for personality, prompt in zip(personalities, prompts):
        assert QUERY in prompt
        assert f"Selected personality: {personality}" in prompt
    assert "processor, RAM, storage" in prompts[0]
    assert "prices, budget fit, value" in prompts[1]
    assert "friendly, accessible language" in prompts[2]


def test_chat_defaults_to_everyday_user_for_existing_clients(monkeypatch):
    """Requests without personality remain valid and use the legacy-friendly default."""
    product = sample_product()
    monkeypatch.setattr(backend, "search_products", lambda **kwargs: [product])
    monkeypatch.setattr(backend, "compare_and_rank", lambda *args, **kwargs: [product])
    monkeypatch.setattr(recommend.config, "LLM_PROVIDER", "gemini")
    prompts = []
    monkeypatch.setattr(
        recommend,
        "_call_gemini",
        lambda prompt: prompts.append(prompt) or "mocked recommendation",
    )

    response = TestClient(backend.app).post("/chat", json={"message": QUERY})

    assert response.status_code == 200
    assert "Selected personality: Everyday User" in prompts[0]


def test_chat_rejects_unknown_personality():
    """FastAPI rejects personality values outside the supported selector options."""
    response = TestClient(backend.app).post(
        "/chat", json={"message": QUERY, "personality": "Sales Bot"}
    )

    assert response.status_code == 422


def test_unconfigured_llm_fallback_uses_persona_and_grounded_product_facts(monkeypatch):
    """The existing no-LLM fallback varies its focus using actual ranked fields."""
    monkeypatch.setattr(recommend.config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(
        recommend,
        "_call_gemini",
        lambda prompt: (_ for _ in ()).throw(NotImplementedError()),
    )
    product = sample_product()
    answers = {
        name: recommend.get_recommendation(QUERY, [product], name)
        for name in ("Tech Geek", "Bargain Hunter", "Everyday User")
    }

    assert len(set(answers.values())) == 3
    assert "Processor: X1" in answers["Tech Geek"]
    assert "Rs 21999" in answers["Bargain Hunter"]
    assert "listed MRP Rs 24999" in answers["Bargain Hunter"]
    assert "rated 4.4/5 from 120 reviews" in answers["Everyday User"]


def test_frontend_sends_personality_with_chat_request(monkeypatch):
    """The selected frontend persona is included in the existing /chat payload."""

    class MockResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"recommendation": "ok"}

    sent = {}

    def mock_post(url: str, json: dict, timeout: float) -> MockResponse:
        sent.update(url=url, payload=json)
        return MockResponse()

    monkeypatch.setattr("frontend.app.requests.post", mock_post)
    query_chat_api("http://test", QUERY, personality="Tech Geek")

    assert sent["url"] == "http://test/chat"
    assert sent["payload"]["personality"] == "Tech Geek"


def test_original_avatar_is_used_and_upload_code_is_absent():
    """The UI uses the checked-in avatar and contains no upload implementation."""
    source = Path("frontend/app.py").read_text(encoding="utf-8")

    assert get_bot_avatar() == "frontend/assets/bot_avatar.png"
    assert Path(get_bot_avatar()).is_file()
    assert "st.sidebar.file_uploader(" not in source
    assert "_save_custom_bot_avatar" not in source
    assert "bot_avatar_path" not in source

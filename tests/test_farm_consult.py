import httpx
import pytest

from app.farm_consult import (
    FarmConsultError,
    chat_payload,
    compose_user_content,
    consult,
    parse_chat_completion,
)


def test_compose_question_only():
    assert compose_user_content("  hello  ") == "hello"


def test_compose_includes_snippet():
    text = compose_user_content("fix this", "package main")
    assert text.startswith("fix this")
    assert "Existing code:\n```go\npackage main\n```" in text


def test_compose_implement_contract():
    text = compose_user_content(
        "Todo HTTP API",
        mode="implement",
        constraints="modernc.org/sqlite, no CGO",
        files="internal/todo/repository.go",
    )
    assert "IMPLEMENTER" in text
    assert "Spec:\nTodo HTTP API" in text
    assert "modernc.org/sqlite" in text
    assert "internal/todo/repository.go" in text
    body = chat_payload("Todo HTTP API", mode="implement")
    assert body["max_tokens"] == 6144


def test_compose_rejects_empty_question():
    with pytest.raises(FarmConsultError, match="required"):
        compose_user_content("  ", "package main")


def test_chat_payload_is_specialist_shaped():
    body = chat_payload("hello", model="gopher-kungfu")
    assert body["model"] == "gopher-kungfu"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert "tools" not in body


def test_parse_completion_and_farm_error():
    assert (
        parse_chat_completion(
            {"choices": [{"message": {"content": "  use a mutex  "}}]}
        )
        == "use a mutex"
    )
    with pytest.raises(FarmConsultError, match="llama-server"):
        parse_chat_completion(
            {"error": {"message": "llama-server binary not found; set --llama-server or PATH"}}
        )


def test_consult_posts_minimal_body(monkeypatch):
    captured: dict = {}

    def fake_post(self, url, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    assert consult("hello", url="http://127.0.0.1:8080/v1/chat/completions") == "ok"
    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert captured["json"]["messages"][0]["content"] == "hello"
    assert captured["json"]["max_tokens"] == 2048
    assert "tools" not in captured["json"]


def test_consult_connection_error(monkeypatch):
    def boom(self, url, json=None, **kwargs):
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", boom)
    with pytest.raises(FarmConsultError, match="cartridge-server"):
        consult("hello")

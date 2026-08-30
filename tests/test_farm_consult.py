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
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0


def test_compose_implement_repair():
    text = compose_user_content(
        "merge CSVs",
        "func Merge(a, b []Row) ([]Row, error)",
        mode="implement",
        files="internal/merge/merge.go",
        previous_files="### internal/merge/merge.go\n```go\npackage merge\n\nvar h int\n```",
        test_error="merge.go:3:5: declared and not used: h",
    )
    assert "This is a REPAIR" in text
    assert "member list" in text
    assert "Spec:\nmerge CSVs" in text
    assert "declared and not used: h" in text
    assert "var h int" in text
    assert text.index("Previous attempt") < text.index("Compiler/test error")
    assert "Spec:\nmerge CSVs\n\nPrevious compile" not in text
    assert "Neighbor APIs (honor these signatures" not in text


def test_compose_implement_neighbor_apis():
    text = compose_user_content(
        "wire CLI",
        mode="implement",
        files="cmd/csvmerge/main.go",
        neighbor_apis="# internal/csvparse\nfunc Parse(path string) (*Table, error)",
    )
    assert "Neighbor APIs (honor these signatures" in text
    assert "func Parse(path string) (*Table, error)" in text
    assert "Honor Neighbor APIs" in text
    existing = text.index("Files to write")
    neighbor = text.index("Neighbor APIs (honor these signatures")
    assert existing < neighbor
    ask = compose_user_content(
        "review this",
        neighbor_apis="func Parse()",
    )
    assert "Neighbor APIs (honor these signatures" not in ask
    body = chat_payload("wire CLI", mode="implement", neighbor_apis="func Parse()")
    assert "tools" not in body
    assert "func Parse()" in body["messages"][0]["content"]



def test_compose_rejects_empty_question():
    with pytest.raises(FarmConsultError, match="required"):
        compose_user_content("  ", "package main")


def test_chat_payload_is_specialist_shaped():
    body = chat_payload("hello", model="gopher-kungfu")
    assert body["model"] == "gopher-kungfu"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["temperature"] == 0.2
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
    assert captured["json"]["temperature"] == 0.2
    assert "tools" not in captured["json"]


def test_consult_connection_error(monkeypatch):
    def boom(self, url, json=None, **kwargs):
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", boom)
    with pytest.raises(FarmConsultError, match="cartridge-server"):
        consult("hello")

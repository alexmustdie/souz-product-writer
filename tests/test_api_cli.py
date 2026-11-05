import json
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from product_writer.cli import main
from product_writer.export import text
from product_writer.llm import DeepSeek
from product_writer.models import AppError


def client(handler):
    return OpenAI(api_key="test-key-never-real", base_url="https://api.deepseek.com",
                  max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def response(content="{}", finish="stop"):
    return {"id": "test-response", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}


def test_real_request_contract_through_http_transport():
    def handler(request):
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        data = json.loads(request.content)
        assert data["response_format"] == {"type": "json_object"}
        assert data["thinking"] == {"type": "disabled"}
        assert data["messages"][0] == {"role": "system", "content": "system json"}
        assert data["messages"][1] == {"role": "user", "content": "payload"}
        return httpx.Response(200, json=response())
    llm = DeepSeek(client=client(handler))
    assert llm.complete("system json", "payload") == "{}"
    assert llm.calls[0]["usage"]["total_tokens"] == 20


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 429, 500, 503])
def test_api_errors_safe(status):
    llm = DeepSeek(client=client(lambda request: httpx.Response(status, json={"error": {"message": "SECRET"}})))
    with pytest.raises(AppError) as error:
        llm.complete("system", "payload")
    assert "SECRET" not in str(error.value)
    assert str(status) in str(error.value)


@pytest.mark.parametrize("exception,match", [(httpx.ConnectError, "подключиться"), (httpx.ReadTimeout, "ожидания")])
def test_network_errors(exception, match):
    def handler(request):
        raise exception("SECRET", request=request)
    with pytest.raises(AppError, match=match):
        DeepSeek(client=client(handler)).complete("s", "u")


@pytest.mark.parametrize("content,finish", [("   ", "stop"), (None, "stop"), ('{"part":', "length"), ("", "content_filter")])
def test_empty_truncated_refused(content, finish):
    llm = DeepSeek(client=client(lambda r: httpx.Response(200, json=response(content, finish))))
    with pytest.raises(AppError):
        llm.complete("s", "u")


def test_non_json_api_response():
    llm = DeepSeek(client=client(lambda r: httpx.Response(200, text="broken upstream")))
    with pytest.raises(AppError, match="формат"):
        llm.complete("s", "u")


@pytest.mark.parametrize("payload", [
    {}, {"choices": [None]},
    {"id": "test", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}]},
    {"id": "test", "model": "x", "choices": [], "usage": {"bad": True}},
])
def test_malformed_api_envelope_is_safe(payload):
    llm = DeepSeek(client=client(lambda r: httpx.Response(200, json=payload)))
    with pytest.raises(AppError, match="структуру"):
        llm.complete("s", "u")


def test_cli_validate_without_key(monkeypatch, capsys):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert main(["examples/incomplete.input.json", "--validate-only"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] and "Питание" in output["missing_fields"]


def test_cli_broken_input_does_not_overwrite_output(tmp_path, capsys):
    source, output = tmp_path / "bad.json", tmp_path / "out.json"
    source.write_text('{"name":"SECRET",}', encoding="utf-8")
    output.write_text("KEEP", encoding="utf-8")
    assert main([str(source), "--output", str(output)]) == 2
    assert output.read_text() == "KEEP"
    assert "SECRET" not in capsys.readouterr().err


def test_cli_bom_and_same_path_protection(tmp_path):
    source = tmp_path / "bom.json"
    source.write_text('{"name":"test"}', encoding="utf-8-sig")
    assert main([str(source), "--validate-only", "--output", str(source)]) == 2
    assert json.loads(source.read_text(encoding="utf-8-sig"))["name"] == "test"


def test_markdown_escapes_embedded_html_and_links():
    escaped = text('<script>alert(1)</script> [click](javascript:alert(1))')
    assert "<script>" not in escaped and "\\[click\\]" in escaped


def test_missing_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(AppError, match="DEEPSEEK_API_KEY"):
        DeepSeek()

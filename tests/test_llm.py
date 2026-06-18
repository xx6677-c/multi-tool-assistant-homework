import json

from llm import LLMClient


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps({"data": {"embedding": [0.1, 0.2, 0.3]}}).encode("utf-8")


def test_doubao_multimodal_embedding(monkeypatch):
    monkeypatch.setattr("llm.config.BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setattr("llm.config.API_KEY", "test-key")
    monkeypatch.setattr("llm.config.EMBEDDING_MODEL", "doubao-embedding-vision-251215")

    captured = []

    def fake_urlopen(req, timeout):
        captured.append({
            "url": req.full_url,
            "auth": req.headers["Authorization"],
            "body": json.loads(req.data.decode("utf-8")),
            "timeout": timeout,
        })
        return _FakeResponse()

    monkeypatch.setattr("llm.request.urlopen", fake_urlopen)

    client = LLMClient()
    embeddings = client.embed(["文本一", "文本二"])

    assert embeddings == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert captured[0]["url"] == "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
    assert captured[0]["auth"] == "Bearer test-key"
    assert captured[0]["body"] == {
        "model": "doubao-embedding-vision-251215",
        "input": [{"type": "text", "text": "文本一"}],
    }
    assert captured[0]["timeout"] == 30

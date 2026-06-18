from tools.web_search import WebSearchTool


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return (
            b'{"data":{"webPages":{"value":[{"name":"AI news","url":"https://example.com/a",'
            b'"summary":"AI headline summary","siteName":"Example"}]}}}'
        )


def test_web_search_is_available():
    assert WebSearchTool().is_available() is True


def test_web_search_returns_summary():
    result = WebSearchTool().run(query="今天热点新闻")
    assert "今天热点新闻" in result
    assert "搜索结果摘要" in result


def test_web_search_handles_empty_query():
    assert "缺少搜索关键词" in WebSearchTool().run(query="")


def test_web_search_calls_bocha(monkeypatch):
    monkeypatch.setenv("BOCHA_API_KEY", "test-key")

    captured = {}

    def fake_urlopen(req, timeout):
        captured["auth"] = req.headers["Authorization"]
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("tools.web_search.request.urlopen", fake_urlopen)

    tool = WebSearchTool()
    result = tool.run(query="AI 新闻")

    assert captured["auth"] == "Bearer test-key"
    assert captured["timeout"] == 15
    assert "AI news" in result
    assert tool.last_sources[0]["doc"] == "https://example.com/a"


def test_schema_shape():
    schema = WebSearchTool().openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_search"
    assert "query" in schema["function"]["parameters"]["properties"]

import pytest

from rag import ingest
from rag.vector_store import STORE, VectorStore
from tools.knowledge_base import KnowledgeBaseTool


class _FakeEmbedder:
    def embed(self, texts):
        vectors = {
            "价格": [1.0, 0.0],
            "快捷键": [0.0, 1.0],
            "知简专业版价格是 ¥18/月": [1.0, 0.0],
            "知简全局搜索快捷键是 Ctrl+Shift+F": [0.0, 1.0],
        }
        return [vectors.get(t, [0.5, 0.5]) for t in texts]


def _reset_store():
    STORE.texts.clear()
    STORE.embeddings.clear()
    STORE.metadatas.clear()


@pytest.fixture(autouse=True)
def clean_shared_store():
    _reset_store()
    yield
    _reset_store()


def test_vector_store_search_orders_results():
    store = VectorStore()
    store.add(
        ["价格资料", "快捷键资料"],
        [[1.0, 0.0], [0.0, 1.0]],
        [{"doc": "price.md"}, {"doc": "shortcut.md"}],
    )

    hits = store.search([0.9, 0.1], top_k=1)

    assert hits[0]["text"] == "价格资料"
    assert hits[0]["metadata"]["doc"] == "price.md"


def test_chunk_text_uses_overlap():
    assert ingest.chunk_text("abcdef", chunk_size=4, overlap=2) == ["abcd", "cdef", "ef"]


def test_ingest_file_adds_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "_llm", _FakeEmbedder())
    path = tmp_path / "doc.md"
    path.write_text("知简专业版价格是 ¥18/月", encoding="utf-8")

    n = ingest.ingest_file(str(path))

    assert n == 1
    assert len(STORE) == 1
    assert STORE.metadatas[0]["doc"] == str(path)


def test_knowledge_base_tool_returns_sources(monkeypatch):
    STORE.add(
        ["知简专业版价格是 ¥18/月", "知简全局搜索快捷键是 Ctrl+Shift+F"],
        [[1.0, 0.0], [0.0, 1.0]],
        [{"doc": "price.md"}, {"doc": "shortcut.md"}],
    )
    tool = KnowledgeBaseTool()
    tool._llm = _FakeEmbedder()

    result = tool.run("价格")

    assert "¥18/月" in result
    assert tool.last_sources[0]["doc"] == "price.md"
    assert tool.is_available() is True

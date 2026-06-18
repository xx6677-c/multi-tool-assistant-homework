import json

from memory.long_term import FileLongTermMemory
from tests.fakes import FakeLLM


def test_load_missing_file_returns_empty(tmp_path):
    memory = FileLongTermMemory(user_id="u1", path=str(tmp_path / "memory.json"))
    assert memory.all_facts() == []


def test_save_preserves_other_users(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"other": ["其他用户事实"]}, ensure_ascii=False), encoding="utf-8")

    memory = FileLongTermMemory(user_id="u1", path=str(path))
    memory.facts = ["用户叫小明"]
    memory._save()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["other"] == ["其他用户事实"]
    assert data["u1"] == ["用户叫小明"]


def test_recall_returns_all_when_facts_are_few(tmp_path):
    memory = FileLongTermMemory(user_id="u1", path=str(tmp_path / "memory.json"))
    memory.facts = ["用户叫小明", "用户喜欢 Python"]
    assert memory.recall("我叫什么") == ["用户叫小明", "用户喜欢 Python"]


def test_remember_deduplicates_facts(tmp_path):
    memory = FileLongTermMemory(user_id="u1", path=str(tmp_path / "memory.json"))
    memory._llm = FakeLLM(summary='["用户叫小明", "用户喜欢 Python", "用户叫小明"]')

    memory.remember("我叫小明，我喜欢 Python", "好的，我记住了。")
    memory.remember("我叫小明，我喜欢 Python", "好的，我记住了。")

    assert memory.all_facts() == ["用户叫小明", "用户喜欢 Python"]


def test_remember_ignores_empty_extraction(tmp_path):
    memory = FileLongTermMemory(user_id="u1", path=str(tmp_path / "memory.json"))
    memory._llm = FakeLLM(summary="[]")

    memory.remember("今天天气怎么样", "天气不错。")

    assert memory.all_facts() == []

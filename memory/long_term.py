"""【学生任务】长期记忆：跨会话记住用户事实/偏好，落盘到 JSON。

采用最简单的"关键词召回 / 全量注入"策略（进阶可改向量召回，见 README）。
实现完成后把模块级 ENABLED 改为 True，factory 会自动启用它。
"""
import json
import os
import re

from memory.base import LongTermMemory
from llm import LLMClient

# 实现完成后改为 True
ENABLED = True


class FileLongTermMemory(LongTermMemory):
    def __init__(self, user_id: str = "local", path: str = "data/memory_store.json"):
        self.user_id = user_id
        self.path = path
        self._llm = LLMClient()
        self.facts: list[str] = self._load()

    def _load(self) -> list[str]:
        """从 JSON 文件读取本 user 的事实列表（文件不存在时返回 []）。

        TODO: 读取 self.path，结构建议 {user_id: [fact, ...]}，
        返回 data.get(self.user_id, [])。
        """
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        facts = data.get(self.user_id, [])
        if not isinstance(facts, list):
            return []
        return [str(f).strip() for f in facts if str(f).strip()]

    def _save(self) -> None:
        """把 self.facts 写回 JSON 文件（保留其它 user 的数据）。

        TODO: 读出整体 dict，更新 self.user_id 对应项，写回文件。
        """
        data = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, json.JSONDecodeError):
                data = {}

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data[self.user_id] = list(self.facts)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def recall(self, query: str) -> list[str]:
        """召回与 query 相关的事实。

        TODO（关键词策略）：返回 query 与事实有词重叠的项；
        事实较少时可直接全量返回。
        """
        if not self.facts:
            return []
        if len(self.facts) <= 20:
            return list(self.facts)

        query_tokens = self._tokens(query)
        if not query_tokens:
            return list(self.facts[:10])

        hits = [fact for fact in self.facts if query_tokens & self._tokens(fact)]
        return hits[:10]

    def remember(self, user_msg: str, reply: str) -> None:
        """从一轮对话中抽取值得长期记住的事实并落盘。

        TODO:
        1. 用 self._llm.chat() 让模型从对话中抽取"用户的稳定事实/偏好"，
           没有则返回空。
        2. 把新事实去重后加入 self.facts，调用 self._save()。
        """
        prompt = (
            "请从下面一轮对话中抽取值得长期记住的用户稳定事实或偏好。"
            "只输出 JSON 数组字符串，例如 [\"用户叫小明\", \"用户喜欢 Python\"]。"
            "如果没有值得长期记住的信息，输出 []。\n\n"
            f"用户：{user_msg}\n"
            f"助手：{reply}"
        )
        try:
            resp = self._llm.chat([
                {"role": "system", "content": "你是记忆抽取器，只抽取用户稳定事实和长期偏好，不记录临时问题。"},
                {"role": "user", "content": prompt},
            ])
            content = resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return

        new_facts = self._parse_facts(content)
        existing = set(self.facts)
        added = False
        for fact in new_facts:
            if fact not in existing:
                self.facts.append(fact)
                existing.add(fact)
                added = True
        if added:
            self._save()

    def all_facts(self) -> list[str]:
        return list(self.facts)

    def _parse_facts(self, content: str) -> list[str]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        candidates = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                candidates = parsed
            elif isinstance(parsed, dict) and isinstance(parsed.get("facts"), list):
                candidates = parsed["facts"]
            elif isinstance(parsed, str):
                candidates = [parsed]
        except json.JSONDecodeError:
            candidates = text.splitlines()

        facts = []
        for item in candidates:
            fact = self._normalize_fact(str(item))
            if fact and fact not in facts:
                facts.append(fact)
        return facts

    def _normalize_fact(self, fact: str) -> str:
        fact = fact.strip()
        fact = re.sub(r"^\s*[-*•\d.、)）]+\s*", "", fact)
        fact = fact.strip(" \t\r\n\"'“”[]")
        if fact in {"", "[]", "无", "没有", "无事实", "无可记忆信息", "没有值得长期记住的信息"}:
            return ""
        if "没有值得" in fact or "无需记住" in fact:
            return ""
        return fact

    def _tokens(self, text: str) -> set[str]:
        text = (text or "").lower()
        words = set(re.findall(r"[a-z0-9_]+", text))
        chars = {ch for ch in text if "\u4e00" <= ch <= "\u9fff"}
        return words | chars

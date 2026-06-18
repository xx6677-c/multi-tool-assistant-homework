"""【学生任务】RAG 检索工具：把知识库检索包装成一个 Tool。

完成后把 is_available() 改成返回 True（或按库是否非空判断），
该工具就会出现在模型可用工具列表里，实现优雅降级。
"""
from tools.base import Tool
from rag.vector_store import STORE
from llm import LLMClient


class KnowledgeBaseTool(Tool):
    name = "knowledge_base"
    description = "在本地知识库中检索与问题相关的资料。当用户的问题可能依赖已上传文档时调用。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要检索的问题或关键词"}
        },
        "required": ["query"],
    }

    def __init__(self):
        super().__init__()  # 初始化 last_sources 等基类实例属性
        self._llm = LLMClient()

    def is_available(self) -> bool:
        # TODO: 实现 run() 后改为 return len(STORE) > 0
        return len(STORE) > 0

    def run(self, query: str) -> str:
        """检索知识库并返回拼接好的上下文，同时把来源写入 self.last_sources。

        TODO:
        1. q_emb = self._llm.embed([query])[0]
        2. hits = STORE.search(q_emb, top_k=3)
        3. self.last_sources = [{"doc": h["metadata"].get("doc"),
                                 "score": h["score"], "snippet": h["text"][:50]} for h in hits]
        4. 返回把各 hit 文本拼接成的字符串（无结果时返回提示语）。
        """
        query = (query or "").strip()
        if not query:
            self.last_sources = []
            return "知识库检索失败：缺少检索问题。"

        try:
            q_emb = self._llm.embed([query])[0]
            hits = STORE.search(q_emb, top_k=3)
        except Exception as exc:  # noqa: BLE001
            self.last_sources = []
            return f"知识库检索失败：{exc}"

        if not hits:
            self.last_sources = []
            return "知识库里没有找到相关资料。"

        self.last_sources = [
            {
                "doc": h["metadata"].get("doc"),
                "score": h["score"],
                "snippet": h["text"][:80],
            }
            for h in hits
        ]
        return "\n\n".join(
            f"[资料 {idx} | 来源：{h['metadata'].get('doc', '未知')}]\n{h['text']}"
            for idx, h in enumerate(hits, start=1)
        )

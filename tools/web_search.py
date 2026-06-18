"""【学生任务】网络搜索工具。

教学提示：可以接真实搜索 API，也可以先用 mock 数据跑通流程
（与 weather.py 同理）。完成 run() 后把 is_available() 改为 True。
"""
import json
import os
from urllib import error, request

from tools.base import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = "根据关键词进行网络搜索，返回相关结果摘要。当用户询问实时信息、最新新闻、当前热点或需要联网查询的问题时调用。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        },
        "required": ["query"],
    }
    endpoint = "https://api.bochaai.com/v1/web-search"

    def is_available(self) -> bool:
        # TODO: 实现 run() 后改为 return True
        return True

    def run(self, query: str) -> str:
        """执行搜索并返回结果摘要字符串。

        TODO: 调用真实搜索 API，或先返回 mock 结果跑通链路。
        """
        query = (query or "").strip()
        if not query:
            return "搜索失败：缺少搜索关键词。"

        api_key = os.getenv("BOCHA_API_KEY", "").strip()
        if api_key:
            result = self._run_bocha(query, api_key)
            if result:
                return result

        return (
            f"关于“{query}”的网络搜索结果摘要（示例数据）：\n"
            f"1. “{query}”相关信息需要参考最新公开来源。\n"
            f"2. 若涉及新闻、政策、价格或实时状态，建议以官方网站或主流媒体为准。\n"
            f"3. 可以围绕关键词“{query}”整理回答。"
        )

    def _run_bocha(self, query: str, api_key: str) -> str:
        payload = {
            "query": query,
            "summary": True,
            "count": int(os.getenv("BOCHA_COUNT", "5")),
        }
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            return f"博查搜索失败，已无法获取实时结果：{exc}"

        pages = data.get("data", {}).get("webPages", {}).get("value", [])
        if not pages:
            return f"没有搜索到与“{query}”相关的结果。"

        lines = [f"关于“{query}”的网络搜索结果摘要："]
        self.last_sources = []
        for idx, page in enumerate(pages[:5], start=1):
            title = page.get("name") or page.get("title") or "未命名结果"
            url = page.get("url") or ""
            snippet = page.get("summary") or page.get("snippet") or page.get("description") or ""
            site = page.get("siteName") or page.get("site") or ""
            source = f"{site} - {url}" if site and url else site or url
            lines.append(f"{idx}. {title}\n   {snippet}\n   来源：{source}")
            self.last_sources.append({"doc": url or title, "score": None, "snippet": snippet[:80]})
        return "\n".join(lines)

"""【学生任务】文档切块 + 嵌入入库。

把 data/docs 下的文本文件读入，切成小块，调用 embedding API 得到向量，
写入 rag.vector_store.STORE。
"""
from pathlib import Path

from rag.vector_store import STORE
from llm import LLMClient

_llm = LLMClient()


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """把长文本切成带重叠的小块。

    TODO: 按 chunk_size 字符切分，相邻块重叠 overlap 字符，返回非空块列表。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def ingest_file(path: str) -> int:
    """读取单个文本文件，切块、嵌入、入库，返回入库的块数。

    TODO:
    1. 读取文件文本。
    2. chunk_text() 切块。
    3. _llm.embed(chunks) 得到向量。
    4. STORE.add(chunks, embeddings, metadatas=[{"doc": path} ...])。
    5. 返回块数。
    """
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = _llm.embed(chunks)
    metadatas = [{"doc": str(file_path)} for _ in chunks]
    STORE.add(chunks, embeddings, metadatas)
    return len(chunks)


def ingest_dir(dir_path: str = "data/docs") -> int:
    """把目录下所有 .md/.txt 文件入库，返回总块数。

    TODO: 遍历目录中的 .md/.txt 文件，对每个调用 ingest_file 并累加块数。
    """
    root = Path(dir_path)
    if not root.exists():
        return 0

    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            total += ingest_file(str(path))
    return total

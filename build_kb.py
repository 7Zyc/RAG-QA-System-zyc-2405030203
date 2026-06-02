"""
build_kb.py
任务2：构建本地知识库
功能：
  1) 批量读取指定文件夹下的 PDF / DOCX / TXT
  2) 使用 RecursiveCharacterTextSplitter 分块（chunk_size=1000, chunk_overlap=200）
  3) 使用 Ollama 嵌入模型（nomic-embed-text / all-minilm）向量化并存入 Chroma
  4) 提供 retrieve(query, k=3) 检索函数
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


DEFAULT_DOCS_DIR = "./docs"
DEFAULT_PERSIST_DIR = "./data/chroma_db"
DEFAULT_COLLECTION = "nlp_kb"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"


def _load_one(path: str) -> List[Document]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".docx":
        loader = Docx2txtLoader(path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        return []
    return loader.load()


def load_documents(docs_dir: str = DEFAULT_DOCS_DIR) -> List[Document]:
    docs: List[Document] = []
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"[警告] 目录不存在：{docs_path.resolve()}")
        return docs

    supported = {".pdf", ".docx", ".txt", ".md"}
    for fp in sorted(docs_path.rglob("*")):
        if fp.is_file() and fp.suffix.lower() in supported:
            try:
                loaded = _load_one(str(fp))
                for d in loaded:
                    d.metadata = d.metadata or {}
                    d.metadata["source"] = str(fp)
                    d.metadata["filename"] = fp.name
                docs.extend(loaded)
                print(f"  [加载] {fp.name}  ->  {len(loaded)} 段")
            except Exception as e:
                print(f"  [失败] {fp.name} : {e}")

    print(f"\n共加载 {len(docs)} 个文档段（来自 {len({d.metadata['filename'] for d in docs})} 个文件）")
    return docs


def split_documents(docs: List[Document],
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "? ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"分块完成：{len(docs)} 段 -> {len(chunks)} 个文本块 "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def get_embeddings(model: str = DEFAULT_EMBED_MODEL,
                   base_url: str = OLLAMA_BASE_URL) -> OllamaEmbeddings:
    return OllamaEmbeddings(base_url=base_url, model=model)


def build_vectorstore(chunks: List[Document],
                      persist_dir: str = DEFAULT_PERSIST_DIR,
                      collection: str = DEFAULT_COLLECTION,
                      embed_model: str = DEFAULT_EMBED_MODEL,
                      reset: bool = False) -> Chroma:
    if reset and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"[重置] 已清空旧向量库：{persist_dir}")

    embeddings = get_embeddings(embed_model)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection,
    )
    print(f"[完成] 向量库已写入：{persist_dir}（collection={collection}）")
    return vectordb


def load_vectorstore(persist_dir: str = DEFAULT_PERSIST_DIR,
                     collection: str = DEFAULT_COLLECTION,
                     embed_model: str = DEFAULT_EMBED_MODEL) -> Chroma:
    embeddings = get_embeddings(embed_model)
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection,
        embedding_function=embeddings,
    )


def retrieve(query: str,
             vectordb: Chroma,
             k: int = 3) -> List[Document]:
    return vectordb.similarity_search(query, k=k)


def add_files_to_vectordb(file_paths: List[str],
                          vectordb: Chroma,
                          chunk_size: int = DEFAULT_CHUNK_SIZE,
                          chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> int:
    new_docs: List[Document] = []
    for p in file_paths:
        loaded = _load_one(p)
        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata["source"] = p
            d.metadata["filename"] = Path(p).name
        new_docs.extend(loaded)

    chunks = split_documents(new_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return 0
    vectordb.add_documents(chunks)
    print(f"[追加] 新增 {len(chunks)} 个文本块")
    return len(chunks)


def main():
    print("=" * 60)
    print(" 构建本地 NLP 知识库 ")
    print("=" * 60)

    docs = load_documents(DEFAULT_DOCS_DIR)
    if not docs:
        print("[错误] 没有可用的文档，请先在 ./docs 下放置 PDF/DOCX/TXT")
        return

    chunks = split_documents(docs)
    build_vectorstore(chunks, reset=True)

    print("\n[测试检索] 输入 'q' 退出")
    vectordb = load_vectorstore()
    while True:
        q = input("\n请输入查询：").strip()
        if not q or q.lower() == "q":
            break
        results = retrieve(q, vectordb, k=3)
        for i, d in enumerate(results, 1):
            print(f"\n--- Top {i} (来源: {d.metadata.get('filename')}) ---")
            snippet = d.page_content[:300].replace("\n", " ")
            print(snippet + ("..." if len(d.page_content) > 300 else ""))


if __name__ == "__main__":
    main()

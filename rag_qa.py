"""
rag_qa.py
任务3：命令行版 RAG 问答
基于 build_kb.py 构建的 Chroma 向量库，使用 Ollama 本地大模型进行检索增强问答。
要求：
  1) 基于提供的参考文档回答，若文档中没有相关信息则明确说"文档中未找到相关答案"。
  2) 支持多轮对话（保留历史）。
  3) 输入 exit/quit 退出。
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from build_kb import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    OLLAMA_BASE_URL,
    load_vectorstore,
    retrieve,
)


SYSTEM_PROMPT = """你是一个严谨的基于本地知识库的智能问答助手。
你只能依据下方"参考文档"中的内容回答用户问题。
- 如果参考文档包含与问题相关的信息，请用简洁清晰的中文进行回答，并在末尾标注引用来源（文件名）。
- 如果参考文档没有相关信息，请直接回答："文档中未找到相关答案"。
- 不要编造参考文档中不存在的概念或数据。
"""


def build_prompt(question: str, history: List[Tuple[str, str]],
                 docs: List[Document]) -> str:
    context_blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("filename", "unknown")
        context_blocks.append(f"[{i}] (来源: {src})\n{d.page_content.strip()}")
    context = "\n\n".join(context_blocks) if context_blocks else "（无）"

    history_str = ""
    if history:
        lines = []
        for q, a in history[-5:]:
            lines.append(f"用户：{q}\n助手：{a}")
        history_str = "\n".join(lines)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== 参考文档 ===\n{context}\n\n"
        f"=== 对话历史 ===\n{history_str or '（无）'}\n\n"
        f"=== 当前问题 ===\n{question}\n\n"
        f"请基于以上参考文档回答："
    )


def main():
    print("=" * 60)
    print(" 本地 RAG 智能问答（命令行版） ")
    print("=" * 60)

    if not os.path.isdir(DEFAULT_PERSIST_DIR):
        print(f"[错误] 向量库目录不存在：{DEFAULT_PERSIST_DIR}")
        print("请先执行：python build_kb.py")
        sys.exit(1)

    try:
        vectordb: Chroma = load_vectorstore()
    except Exception as e:
        print(f"[错误] 加载向量库失败：{e}")
        print("请确认已运行 build_kb.py 且 Ollama 服务在运行。")
        sys.exit(1)

    llm = Ollama(base_url=OLLAMA_BASE_URL, model="deepseek-r1:7b",
                 temperature=0.2, num_ctx=4096)

    history: List[Tuple[str, str]] = []

    print("\n提示：输入问题后回车即可，输入 exit/quit 退出。\n")
    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break

        docs = retrieve(question, vectordb, k=3)
        prompt = build_prompt(question, history, docs)

        print("AI：", end="", flush=True)
        try:
            answer = llm.invoke(prompt).strip()
        except Exception as e:
            answer = f"[调用模型失败]：{e}"
        print(answer)
        print()

        history.append((question, answer))


if __name__ == "__main__":
    main()

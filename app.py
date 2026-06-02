"""
app.py
任务4：Streamlit Web 应用
功能：
  1) 上传 PDF / DOCX / TXT 文档
  2) 一键构建 / 更新向量知识库
  3) 文本输入 + "提问" 按钮触发 RAG 问答
  4) 显示多轮对话历史（会话记忆 st.session_state）
  5) 显示知识库中文本块数量等状态
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import List

import streamlit as st

from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from build_kb import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    OLLAMA_BASE_URL,
    _load_one,
    split_documents,
    build_vectorstore,
    load_vectorstore,
    get_embeddings,
)


SYSTEM_PROMPT = """你是一个严谨的基于本地知识库的智能问答助手。
你只能依据下方"参考文档"中的内容回答用户问题。
- 如果参考文档包含与问题相关的信息，请用简洁清晰的中文进行回答，并在末尾标注引用来源（文件名）。
- 如果参考文档没有相关信息，请直接回答："文档中未找到相关答案"。
- 不要编造参考文档中不存在的概念或数据。
"""


st.set_page_config(
    page_title="本地 RAG 智能问答系统",
    page_icon="📚",
    layout="wide",
)


def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "file_count" not in st.session_state:
        st.session_state.file_count = 0


def try_load_existing_db():
    if st.session_state.vectordb is not None:
        return
    if os.path.isdir(DEFAULT_PERSIST_DIR) and os.listdir(DEFAULT_PERSIST_DIR):
        try:
            st.session_state.vectordb = load_vectorstore()
            st.session_state.chunk_count = st.session_state.vectordb._collection.count()
            st.session_state.file_count = len({
                d.metadata.get("filename")
                for d in st.session_state.vectordb.get()["metadatas"]
                if d.get("filename")
            })
        except Exception as e:
            st.warning(f"加载已有向量库失败：{e}")


def build_prompt(question: str, history: List[dict], docs: List[Document]) -> str:
    context_blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("filename", "unknown")
        context_blocks.append(f"[{i}] (来源: {src})\n{d.page_content.strip()}")
    context = "\n\n".join(context_blocks) if context_blocks else "（无）"

    history_str = ""
    if history:
        lines = []
        for h in history[-5:]:
            lines.append(f"用户：{h['question']}\n助手：{h['answer']}")
        history_str = "\n".join(lines)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== 参考文档 ===\n{context}\n\n"
        f"=== 对话历史 ===\n{history_str or '（无）'}\n\n"
        f"=== 当前问题 ===\n{question}\n\n"
        f"请基于以上参考文档回答："
    )


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ 设置")

        llm_model = st.text_input("对话模型 (Ollama)", value="deepseek-r1:7b")
        embed_model = st.text_input("嵌入模型 (Ollama)", value=DEFAULT_EMBED_MODEL)
        ollama_url = st.text_input("Ollama 服务地址", value=OLLAMA_BASE_URL)
        k = st.slider("检索 Top-K", min_value=1, max_value=10, value=3)

        st.divider()
        st.subheader("📊 知识库状态")
        st.metric("文本块数量", st.session_state.chunk_count)
        st.metric("已收录文档", st.session_state.file_count)

        st.divider()
        if st.button("🗑️ 清空知识库", type="secondary"):
            if os.path.isdir(DEFAULT_PERSIST_DIR):
                shutil.rmtree(DEFAULT_PERSIST_DIR)
            st.session_state.vectordb = None
            st.session_state.chunk_count = 0
            st.session_state.file_count = 0
            st.session_state.chat_history = []
            st.success("已清空知识库与会话历史")

        if st.button("📥 重新加载已有知识库"):
            try:
                st.session_state.vectordb = load_vectorstore(embed_model=embed_model)
                st.session_state.chunk_count = st.session_state.vectordb._collection.count()
                st.success(f"已加载，共 {st.session_state.chunk_count} 个文本块")
                st.rerun()
            except Exception as e:
                st.error(f"加载失败：{e}")

        return llm_model, embed_model, ollama_url, k


def render_upload_section(embed_model: str):
    st.subheader("📤 上传文档并构建知识库")
    uploaded_files = st.file_uploader(
        "支持 PDF / DOCX / TXT / MD（可多选）",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        rebuild = st.button("🔨 构建 / 更新知识库", type="primary")
    with col2:
        show_sample = st.checkbox("同时载入 ./docs 下的样例文档", value=True)

    if rebuild:
        all_files = []
        if uploaded_files:
            for uf in uploaded_files:
                tmp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmp_dir, uf.name)
                with open(tmp_path, "wb") as f:
                    f.write(uf.getbuffer())
                all_files.append(tmp_path)
        if show_sample and os.path.isdir("./docs"):
            from pathlib import Path
            for fp in Path("./docs").rglob("*"):
                if fp.is_file() and fp.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}:
                    all_files.append(str(fp))

        if not all_files:
            st.warning("没有可用的文档，请先上传或放入 ./docs")
            return

        docs: List[Document] = []
        for p in all_files:
            try:
                loaded = _load_one(p)
                for d in loaded:
                    d.metadata = d.metadata or {}
                    d.metadata["filename"] = os.path.basename(p)
                    d.metadata["source"] = p
                docs.extend(loaded)
            except Exception as e:
                st.error(f"读取失败 {p} : {e}")

        if not docs:
            st.error("文档解析失败")
            return

        chunks = split_documents(docs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        with st.spinner("正在向量化并写入向量库……"):
            st.session_state.vectordb = build_vectorstore(
                chunks,
                embed_model=embed_model,
                reset=False,
            )
            st.session_state.chunk_count = st.session_state.vectordb._collection.count()
            st.session_state.file_count = len({d.metadata.get("filename") for d in chunks})
        st.success(f"✅ 完成！新增 {len(chunks)} 个文本块")


def render_chat_section(llm_model: str, ollama_url: str, k: int):
    st.subheader("💬 问答区")
    question = st.text_input("请输入你的问题：", key="question_input")
    ask = st.button("提问", type="primary")

    if ask and question.strip():
        if st.session_state.vectordb is None:
            st.error("请先构建知识库！")
            return
        try:
            docs = st.session_state.vectordb.similarity_search(question.strip(), k=k)
        except Exception as e:
            st.error(f"检索失败：{e}")
            return

        prompt = build_prompt(question.strip(), st.session_state.chat_history, docs)

        try:
            llm = Ollama(base_url=ollama_url, model=llm_model,
                         temperature=0.2, num_ctx=4096)
            with st.spinner("大模型正在思考…"):
                answer = llm.invoke(prompt).strip()
        except Exception as e:
            answer = f"调用模型失败：{e}"

        st.session_state.chat_history.append({
            "question": question.strip(),
            "answer": answer,
            "sources": [d.metadata.get("filename", "unknown") for d in docs],
        })

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("#### 🕘 对话历史")
        for i, h in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}：{h['question']}",
                             expanded=(i == 1)):
                st.markdown(f"**🙋 问：** {h['question']}")
                st.markdown(f"**🤖 答：** {h['answer']}")
                if h.get("sources"):
                    st.caption("引用来源：" + "、".join(set(h["sources"])))


def main():
    st.title("📚 本地 RAG 智能问答系统")
    st.caption("基于 Ollama 本地大模型 + LangChain + Chroma + Streamlit")

    init_session_state()
    try_load_existing_db()
    llm_model, embed_model, ollama_url, k = render_sidebar()

    tab1, tab2 = st.tabs(["📖 知识库", "💬 问答"])
    with tab1:
        render_upload_section(embed_model)
    with tab2:
        render_chat_section(llm_model, ollama_url, k)


if __name__ == "__main__":
    main()

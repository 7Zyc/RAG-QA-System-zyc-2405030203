# 📚 RAG-QA-System

> 基于 **Ollama 本地大模型 + LangChain + Chroma + Streamlit** 的检索增强生成（RAG）智能问答系统。
> 支持 PDF / DOCX / TXT 文档的本地知识库构建与自然语言问答，全程离线运行，保护数据隐私。

![main](docs/screenshots/01_main.png)

---

## 一、项目简介

本项目实现了一个完整的"本地知识库 + 大模型"问答系统：
1. 用户上传或选择一批与 **自然语言处理（NLP）** 相关的文档（PDF / DOCX / TXT）。
2. 系统将文档解析、分块、向量化后存入本地 **Chroma** 向量数据库。
3. 当用户提出问题时，系统先在向量库中检索最相关的 3 个文本块，再交由 Ollama 上的 **DeepSeek-R1 / Qwen2** 等本地大模型生成回答。
4. 所有计算与推理均在本地完成，**无需联网**，可有效缓解大模型"幻觉"问题。

---

## 二、环境要求与安装步骤

### 1. 软件要求

| 组件       | 版本建议                | 用途                       |
|----------|---------------------|--------------------------|
| Python   | 3.10 - 3.11         | 运行环境                     |
| Ollama   | 最新版                 | 本地大模型服务                  |
| Git      | 最新版                 | 代码版本管理                   |
| 操作系统     | Windows 10 / 11     | 目标平台（亦可 macOS / Linux） |

### 2. 安装 Ollama 并下载模型

前往 [https://ollama.com](https://ollama.com) 下载安装对应系统版本。

```bash
# 启动 Ollama 服务（Windows 安装完会自动后台运行）
ollama serve

# 下载对话模型（任选其一）
ollama pull deepseek-r1:7b
# 或
ollama pull qwen2:7b

# 下载嵌入模型
ollama pull nomic-embed-text
```

> ⚠️ 模型默认存储在 `C:\Users\<用户>\.ollama`，请预留至少 10GB 空间。

### 3. 创建 Python 虚拟环境并安装依赖

```bash
# 进入项目目录
cd RAG-QA-System

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 安装依赖
pip install -r requirements.txt
```

### 4. 验证环境

```bash
python test_ollama.py
```

如果终端打印 `[OK]` 字样，说明 Ollama 服务与模型均可用。

---

## 三、使用说明

### 任务 2：构建本地知识库（命令行）

```bash
# 将 5 份以上 NLP 相关 PDF/DOCX/TXT 放入 ./docs 目录后执行：
python build_kb.py
```

执行后将自动：解析文档 → 分块（chunk_size=1000, overlap=200）→ 向量化 → 写入 `data/chroma_db`。
脚本会进入交互式检索模式，可输入问题查看 Top-3 文本块。

### 任务 3：命令行版 RAG 问答

```bash
python rag_qa.py
```

输入问题即可获得基于本地知识库的回答，输入 `exit` 退出。

### 任务 4：启动 Streamlit Web 应用

```bash
streamlit run app.py
```

浏览器自动打开 [http://localhost:8501](http://localhost:8501) 。

Web 应用包含两大标签页：

* **📖 知识库** — 上传 PDF/DOCX/TXT 文档（可多选）→ 点击「构建 / 更新知识库」按钮，系统会即时向量化并入库。
* **💬 问答** — 输入问题并点击「提问」，系统返回答案 + 引用来源；下方显示完整对话历史。

侧边栏可配置：对话模型、嵌入模型、Ollama 地址、检索 Top-K，并可「清空知识库」。

### 任务 5：打包为 exe

```bash
# Windows
pack.bat
# 等价命令
pyinstaller --noconfirm --clean RAG-QA-System.spec
```

打包成功后，`dist/RAG-QA-System/RAG-QA-System.exe` 即为独立可执行文件。
复制到无 Python 环境的电脑时，请确保：

1. 目标机器已安装 Ollama 并启动（`ollama serve`）。
2. 已下载所需模型（`ollama pull deepseek-r1:7b`、`ollama pull nomic-embed-text`）。

---

## 四、关键技术点

| 环节       | 选型 / 参数                                  | 说明                          |
|----------|------------------------------------------|-----------------------------|
| 大模型      | DeepSeek-R1:7b / Qwen2:7b（可切换）           | 通过 Ollama 提供 OpenAI 兼容 HTTP API |
| 嵌入模型     | nomic-embed-text（Ollama 内置）              | 文本向量化                       |
| 文档加载     | PyPDFLoader / Docx2txtLoader / TextLoader | 多格式统一处理                      |
| 文本分块     | RecursiveCharacterTextSplitter           | chunk_size=1000, overlap=200 |
| 向量数据库    | Chroma（持久化到 ./data/chroma_db）            | 轻量、本地化                       |
| RAG 框架   | LangChain 0.2 + langchain-community      | 检索增强生成链                      |
| Web 界面   | Streamlit 1.30+                          | 上传 / 检索 / 问答 / 历史              |
| 打包       | PyInstaller 6                            | 生成 Windows 独立 exe           |

### RAG 流程

```
用户问题
   │
   ▼
[1] Embedding ──> 在 Chroma 中检索 Top-K 文本块
   │                                    │
   ▼                                    ▼
[2] 拼接 Prompt（系统提示词 + 参考文档 + 对话历史 + 当前问题）
   │
   ▼
[3] 调用 Ollama 本地大模型
   │
   ▼
最终回答（含引用来源）
```

### 系统提示词

> 你是一个严谨的基于本地知识库的智能问答助手。
> 你只能依据下方"参考文档"中的内容回答用户问题。
> - 如果参考文档包含与问题相关的信息，请用简洁清晰的中文进行回答，并在末尾标注引用来源（文件名）。
> - 如果参考文档没有相关信息，请直接回答："文档中未找到相关答案"。
> - 不要编造参考文档中不存在的概念或数据。

---

## 五、项目结构

```
RAG-QA-System/
├── app.py                  # Streamlit Web 应用
├── build_kb.py             # 知识库构建模块
├── rag_qa.py               # 命令行 RAG 问答
├── test_ollama.py          # Ollama 环境测试
├── RAG-QA-System.spec      # PyInstaller 打包配置
├── pack.bat                # 一键打包脚本
├── requirements.txt        # 依赖列表
├── README.md               # 本文档
├── .gitignore
├── docs/                   # 示例 NLP 文档
│   ├── 01_nlp_intro.txt
│   ├── 02_word_embedding.txt
│   ├── 03_transformer.txt
│   ├── 04_attention.txt
│   ├── 05_bert_gpt.txt
│   ├── 06_rag_overview.txt
│   └── screenshots/
├── data/                   # 向量库持久化目录（运行时自动生成）
```

---

## 六、效果截图

> 截图位于 `docs/screenshots/` 目录，请将运行时截图放入该目录后推送。

1. **主界面** — `01_main.png`
2. **上传文档** — `02_upload.png`
3. **构建知识库** — `03_build.png`
4. **问答效果** — `04_chat.png`
5. **无关问题拒答** — `05_reject.png`

### 问答示例

> **Q1：** 什么是词嵌入（Word Embedding）？
> **A1：** 词嵌入是将自然语言中的词语映射为低维、稠密的实数向量的技术。经典方法包括 Word2Vec、GloVe 等，能捕捉词语之间的语义关系。 （来源：02_word_embedding.txt）
>
> **Q2：** Transformer 的核心思想是什么？
> **A2：** Transformer 的核心是完全基于自注意力（Self-Attention）机制，抛弃了 RNN 的循环结构，使得模型可以并行训练并捕获长距离依赖。 （来源：03_transformer.txt）
>
> **Q3：** 今天北京天气如何？
> **A3：** 文档中未找到相关答案。

---

## 七、已知问题与改进方向

### 已知问题

* DeepSeek-R1 默认会输出 `<think>...</think>` 推理标签，可通过解析或切换模型解决。
* 当文档中存在大量重复内容时，向量检索可能返回相似度极高的多个块。
* 打包后 exe 体积较大（300MB+），主要为 PyTorch/Transformers 等可选依赖被静态扫描。

### 改进方向

* 增加 **流式输出**（打字机效果）。
* 支持 **Markdown / HTML** 文档解析。
* 增加 **引用高亮**：答案中标注引用片段所在原文位置。
* 引入 **Re-rank** 模型（bge-reranker）提升检索质量。
* 增加 **多用户会话管理** 与 **问答记录导出**。
* 提供 **Web 一键安装 Ollama** 的引导脚本。

---

## 八、AI 使用日志

本项目使用 Trae AI 辅助开发，主要环节包括：

| 时间       | 提问内容                                | 关键产出                                          |
|----------|-------------------------------------|-----------------------------------------------|
| 2026-06-01 | 请用 LangChain 写一个基于 Chroma 的检索函数 | `retrieve(query, vectordb, k=3)` 的实现思路           |
| 2026-06-01 | Streamlit 怎么保留多轮对话                  | `st.session_state` 持久化 `chat_history`         |
| 2026-06-02 | PyInstaller 打包 Streamlit 注意事项        | `RAG-QA-System.spec` 中 `hiddenimports` 的写法     |

> 代码中由 AI 生成的片段已添加 `# AI-Generated` 注释，便于评审与回溯。

---

## 九、许可

本项目仅用于课程作业与学习交流，欢迎在此基础上二次开发。
如使用本项目作为参考，请保留作者信息与本 README。

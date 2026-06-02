"""
test_ollama.py
任务1：验证 Ollama API 是否能正常返回结果。
运行：python test_ollama.py
"""

import sys

try:
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    print("[错误] 未安装 langchain-community，请先执行：pip install langchain-community")
    sys.exit(1)


def test_chat(base_url: str = "http://localhost:11434",
              model: str = "deepseek-r1:7b") -> bool:
    print(f"\n[1/2] 测试对话模型：{model} @ {base_url}")
    try:
        llm = Ollama(base_url=base_url, model=model)
        resp = llm.invoke("用一句话介绍什么是自然语言处理。")
        print("[OK] 模型响应：")
        print("-" * 60)
        print(resp.strip())
        print("-" * 60)
        return True
    except Exception as e:
        print(f"[失败] 调用模型出错：{e}")
        print("请确认：")
        print("  1) Ollama 已启动（ollama serve）")
        print(f"  2) 已下载模型：ollama pull {model}")
        return False


def test_embed(base_url: str = "http://localhost:11434",
               model: str = "nomic-embed-text") -> bool:
    print(f"\n[2/2] 测试嵌入模型：{model} @ {base_url}")
    try:
        emb = OllamaEmbeddings(base_url=base_url, model=model)
        vec = emb.embed_query("自然语言处理是人工智能的重要方向。")
        print(f"[OK] 嵌入向量维度：{len(vec)}")
        return True
    except Exception as e:
        print(f"[失败] 调用嵌入模型出错：{e}")
        print(f"请先执行：ollama pull {model}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print(" Ollama 环境自检脚本 ")
    print("=" * 60)

    chat_ok = test_chat()
    embed_ok = test_embed()

    print("\n" + "=" * 60)
    if chat_ok and embed_ok:
        print(" 全部通过：Ollama 服务与模型均可用 ")
    else:
        print(" 自检未通过，请根据上方提示处理后重试 ")
    print("=" * 60)

    sys.exit(0 if (chat_ok and embed_ok) else 1)

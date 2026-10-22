"""
========================================================================
rag.py — NPC 知识库检索（RAG 的「检索」部分）
========================================================================
RAG = Retrieval-Augmented Generation（检索增强生成）
为什么 NPC 需要它：
  纯 LLM 的 NPC 会"一本正经地瞎编"游戏设定。
  RAG 让 NPC 先去知识库检索相关设定，再基于真实资料回答，
  这样 NPC 说的每件装备/地名/任务都是"真的"，不是编的。

这条链路分三步：
  1. build_index()  把设定文档切块 -> 向量化 -> 建索引（启动时跑一次）
  2. retrieve()     把玩家提问向量化 -> 在索引里搜最相似的块 -> 返回文本
  3. dialogue.py    把返回的文本拼进 system_prompt，NPC 就能"引用"它

依赖（第一次用前安装）：
  pip install faiss-cpu sentence-transformers modelscope

国内网络说明：
  HuggingFace 官方在国内常被墙，所以代码默认先使用 HF 镜像 hf-mirror.com。
  如果镜像仍失败，会自动 fallback 到 ModelScope（魔搭社区）下载模型。

关键概念（面试常考，务必理解）：
  - Embedding：把文字变成一串数字（向量），语义相近的文字向量也相近
  - 向量检索：在向量空间里找离 query 最近的几个 chunk（FAISS 干这事）
  - chunk 切块：长文档不能直接整篇塞给模型，要切成小段分别检索
========================================================================
"""
import os
import pickle

# 国内环境：优先使用 HF 镜像，避免直接访问 huggingface.co 被墙
# 必须在导入 sentence_transformers 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 全局状态：模型只加载一次，索引和文本块常驻内存
_model = None
_index = None
_chunks = []

# 模型名称（HuggingFace 和 ModelScope 通用）
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 索引缓存目录：建一次索引后落盘，按 lore.txt 修改时间判断是否命中
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, ".rag_cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(CACHE_DIR, "chunks.pkl")
MTIME_PATH = os.path.join(CACHE_DIR, "lore_mtime.txt")


def _load_model():
    """懒加载嵌入模型；国内优先 HF 镜像，失败再 fallback 到 ModelScope"""
    global _model
    if _model is not None:
        return _model

    try:
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[RAG] 模型加载成功（HF 镜像）：{MODEL_NAME}")
        return _model
    except Exception as e:
        print(f"[RAG] HF 镜像加载失败，尝试 ModelScope 魔搭社区：{e}")

    try:
        # ModelScope 是国内常用的模型镜像站，与 HuggingFace 同名模型互通
        from modelscope import snapshot_download
        model_dir = snapshot_download(MODEL_NAME)
        _model = SentenceTransformer(model_dir)
        print(f"[RAG] 模型加载成功（ModelScope）：{MODEL_NAME}")
        return _model
    except Exception as e2:
        raise RuntimeError(
            "模型下载失败，请尝试以下方法之一：\n"
            "1. 设置环境变量 HF_ENDPOINT=https://hf-mirror.com 后重试\n"
            "2. 安装 modelscope：pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
            "3. 手动下载模型到本地，并修改 _load_model() 里的 MODEL_NAME 为本地路径\n"
            f"原始错误：{e} | Fallback 错误：{e2}"
        )


def _chunk_text(text, max_len=200, overlap=50):
    """
    把长文档切成小块。
    - 先按空行分段（一段通常是一个完整语义单元）
    - 超长的段再用「滑动窗口 + 重叠」切，避免切断语义
    重叠(overlap)让相邻块共享部分文字，检索时更不容易漏掉跨边界的信息。
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= max_len:
            chunks.append(para)
        else:
            step = max_len - overlap
            for i in range(0, len(para), step):
                chunks.append(para[i:i + max_len])
    return chunks


def _index_fresh(lore_file):
    """判断缓存索引是否仍然有效：缓存文件齐全，且 lore.txt 未被改动。"""
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)
            and os.path.exists(MTIME_PATH)):
        return False
    try:
        with open(MTIME_PATH, "r", encoding="utf-8") as f:
            return float(f.read()) == os.path.getmtime(lore_file)
    except (ValueError, OSError):
        return False


def build_index(lore_file, force=False):
    """读取设定文档 -> 切块 -> 向量化 -> 建 FAISS 索引。

    - 命中缓存（lore 未改动）时直接读盘，跳过切块与向量化，启动更快
    - force=True 强制重建（用于设定文档被改但修改时间未变的极端情况）
    """
    global _index, _chunks
    if not force and _index_fresh(lore_file):
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)
        _index = faiss.read_index(INDEX_PATH)
        print(f"[RAG] 命中缓存索引：{len(_chunks)} 个文本块（跳过重建）")
        return

    with open(lore_file, "r", encoding="utf-8") as f:
        text = f.read()

    _chunks = _chunk_text(text)
    model = _load_model()

    # normalize_embeddings=True 后，用内积(IndexFlatIP)等价于余弦相似度
    embeddings = model.encode(_chunks, normalize_embeddings=True)
    dim = embeddings.shape[1]

    _index = faiss.IndexFlatIP(dim)
    _index.add(embeddings.astype("float32"))

    # 落盘缓存
    os.makedirs(CACHE_DIR, exist_ok=True)
    faiss.write_index(_index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(_chunks, f)
    with open(MTIME_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.path.getmtime(lore_file)))

    print(f"[RAG] 索引建立完成：{len(_chunks)} 个文本块，向量维度 {dim}（已缓存）")


def retrieve(query, top_k=3):
    """把提问向量化，检索最相关的 top_k 个文本块，用分隔符拼成一段返回"""
    if _index is None:
        raise RuntimeError("请先调用 build_index() 建立索引")

    model = _load_model()
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")

    # search 返回 (scores, indices)，scores 越大越相似（因为用了内积）
    scores, indices = _index.search(q_emb, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx != -1:  # FAISS 在结果不足 top_k 时用 -1 填充
            results.append(_chunks[idx])
    return "\n---\n".join(results)


if __name__ == "__main__":
    # 直接运行本文件可做端到端测试：python rag.py
    here = os.path.dirname(os.path.abspath(__file__))
    lore = os.path.join(here, "lore.txt")
    if not os.path.exists(lore):
        print("请把游戏设定写入 lore.txt 再运行本文件做测试")
    else:
        build_index(lore)
        print("\n--- 检索测试 ---")
        for q in ["商人罗杰卖什么武器？", "法师莉莉有什么禁忌？", "战士艾伦的背景故事？","What does Roger sell?"]:
            print(f"\nQ: {q}")
            print(retrieve(q))

"""
API 客户端 —— 封装 DeepSeek 调用
密钥管理：只从环境变量读取，绝不硬编码到源码里。
    set DEEPSEEK_API_KEY=your_key        (Windows, cmd)
    export DEEPSEEK_API_KEY=your_key    (Linux / macOS / Git Bash)
"""
import os
import requests

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"


def _api_key():
    """懒校验：调用时才检查密钥，避免 RAG 独立测试也被密钥缺失阻断。"""
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "未设置环境变量 DEEPSEEK_API_KEY。\n"
            "请先运行：export DEEPSEEK_API_KEY=\"your_key\" "
            "（Windows 用 set DEEPSEEK_API_KEY=your_key）\n"
            "切勿把 key 写进源码或提交到 GitHub。"
        )
    return key


def call_deepseek(messages, temperature=0.8):
    """发送一轮对话到 DeepSeek，返回回复文本"""
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {_api_key()}"},
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def chat(messages, tools=None, model=MODEL, temperature=0.8):
    """返回完整的 assistant message（含 tool_calls），用于 Agent 工具调用循环。"""
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {_api_key()}"},
                      json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]

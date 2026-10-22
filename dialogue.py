"""NPC 对话核心逻辑
MVP（记忆+多语言）+ RAG 知识检索 + 工具调用（Agent）已全部接入，可直接运行
"""
import os
import json
from api_client import call_deepseek, chat
from rag import build_index, retrieve
from tools import TOOL_SCHEMAS, execute_tool_calls

_here = os.path.dirname(os.path.abspath(__file__))
build_index(os.path.join(_here, "lore.txt"))   # 启动建一次索引


class NPC:
    def __init__(self, config):
        self.id = config["id"]
        self.name = config["name"]
        self.role = config.get("role", "")
        self.system_prompt = config["system_prompt"]
        self.memory = []          # 对话历史（滑动窗口）
        self.max_turns = 5        # 只保留最近 5 轮，控制 token 成本

    def _lang_instruction(self, lang):
        if lang == "en":
            return "\n[Instruction] Reply in English."
        return "\n[Instruction] 用中文回答。"

    def respond(self, user_input, lang="zh"):
        # ============ RAG 知识检索（已接入）============
        knowledge = retrieve(user_input, top_k=3)
        rag_context = f"\n[游戏设定参考]\n{knowledge}"

        # ============ 工具调用（Agent 能力，已接入）============
        sys_msg = {
            "role": "system",
            "content": self.system_prompt + self._lang_instruction(lang) + rag_context,
        }

        # 拼历史（最近 max_turns 轮）
        self.memory.append({"role": "user", "content": user_input})
        messages = [sys_msg] + self.memory[-self.max_turns * 2:]

        # 第一轮：带工具，让 LLM 决定要不要调用
        assistant_msg = chat(messages, tools=TOOL_SCHEMAS)
        reply = assistant_msg.get("content") or ""

        # 有工具调用则执行，并把结果回馈 LLM 求最终回复
        if assistant_msg.get("tool_calls"):
            messages.append({
                "role": "assistant",
                "content": reply,
                "tool_calls": assistant_msg["tool_calls"],
            })
            messages.extend(execute_tool_calls(assistant_msg["tool_calls"]))
            final = chat(messages)          # 第二轮不传 tools，避免死循环
            reply = final.get("content") or reply

        self.memory.append({"role": "assistant", "content": reply})
        return reply

"""游戏世界工具集：让 NPC 能调用游戏系统（背包/任务/等级），而非只会聊天。
这是 Function Calling / Agent 的核心：LLM 理解玩家意图 -> 决定调哪个工具
-> 代码执行工具 -> 结果回馈 LLM -> 生成自然语言回复。
"""
import json

# ---------- 工具函数：真实执行游戏逻辑（这里用模拟数据）----------
def check_inventory():
    """查看玩家当前背包里有哪些物品"""
    return "玩家背包：治疗药水x3, 铁剑, 任务卷轴"

def give_quest(quest_id: str):
    """给玩家发放一个任务"""
    return f"已向玩家发放任务：{quest_id}"

def check_player_level():
    """查看玩家当前的等级"""
    return "玩家当前等级：12"

# ---------- 工具函数注册表（dispatch 按名字找函数）----------
TOOLS = {
    "check_inventory": check_inventory,
    "give_quest": give_quest,
    "check_player_level": check_player_level,
}

# ---------- 给 LLM 看的工具 schema（OpenAI / DeepSeek function-calling 格式）----------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "查看玩家当前背包里有哪些物品",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "give_quest",
            "description": "给玩家发放一个任务",
            "parameters": {
                "type": "object",
                "properties": {
                    "quest_id": {"type": "string", "description": "任务名称或编号"}
                },
                "required": ["quest_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_player_level",
            "description": "查看玩家当前的等级",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ---------- 调度器：执行 LLM 请求的所有工具调用 ----------
def execute_tool_calls(tool_calls: list) -> list:
    """执行 DeepSeek 返回的所有 tool_calls，返回可直接追加进 messages 的 tool 消息列表。

    每条结果含 tool_call_id（与 assistant 的 tool_calls 对齐）与执行结果字符串；
    单个工具抛错不影响其余工具，错误信息会作为该工具的结果回喂给模型。
    """
    results = []
    for call in tool_calls:
        fn = call["function"]
        name = fn["name"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
            if name in TOOLS:
                out = TOOLS[name](**args)
            else:
                out = f"未知工具: {name}"
        except Exception as e:
            out = f"工具执行出错: {e}"
        results.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "name": name,
            "content": str(out),
        })
    return results

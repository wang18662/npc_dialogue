"""
Streamlit 前端入口
跑法：streamlit run main.py
"""
import os
import json
import streamlit as st

from dialogue import NPC

# 加载所有 NPC 配置
config_dir = os.path.join(os.path.dirname(__file__), "npc_config")
npcs = {}
for f in os.listdir(config_dir):
    if f.endswith(".json"):
        with open(os.path.join(config_dir, f), encoding="utf-8") as fp:
            cfg = json.load(fp)
            npcs[cfg["name"]] = NPC(cfg)

st.title("Game NPC Dialogue")
st.caption("基于 DeepSeek API 的多角色游戏 NPC 对话 Demo")

npc_name = st.selectbox("选择 NPC", list(npcs.keys()))
lang = st.radio("语言", ["中文", "English"])

if "history" not in st.session_state:
    st.session_state.history = {}  # 按 NPC 名字分桶，避免切换角色后历史串台

user_input = st.text_input("你说：")

if st.button("发送") and user_input:
    npc = npcs[npc_name]
    reply = npc.respond(user_input, lang=("en" if lang == "English" else "zh"))
    st.session_state.history.setdefault(npc_name, [])
    st.session_state.history[npc_name].append((user_input, reply))
    st.session_state.history[npc_name] = st.session_state.history[npc_name][-10:]  # 每角色只显示最近10条

for u, r in st.session_state.history.get(npc_name, []):
    st.write(f"**你**：{u}")
    st.write(f"**{npc_name}**：{r}")

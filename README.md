# Game NPC Dialogue — 基于 RAG + Function Calling 的多角色游戏 NPC 对话系统

> 用 DeepSeek API 驱动的 NPC 对话 Demo：NPC 能引用游戏世界观真实设定回答问题，
> 能调用游戏系统接口（背包/任务/等级），支持三个性格迥异的角色与中英双语切换。

---

## English Summary

A game NPC dialogue demo built on the DeepSeek API. NPCs retrieve grounded lore via
RAG (BGE embedding + FAISS) so they never fabricate world-setting facts, and invoke
game-system functions (inventory / quest / level) through a two-pass Function Calling
loop. Three distinct NPC personas are driven by JSON config, with sliding-window
memory and zh/en bilingual output.

- **Stack**: Python · Streamlit · DeepSeek API · sentence-transformers (BAAI/bge-small-zh-v1.5) · FAISS
- **Scale**: 5 modules, ~370 lines

---

## 一、这个项目解决了什么问题

纯 LLM 驱动的 NPC 有两个致命伤，本项目逐个拆掉：

| 问题 | 表现 | 本项目的解法 |
|---|---|---|
| **幻觉** | NPC 一本正经地瞎编装备数值、地名、任务，破坏世界观一致性 | **RAG 检索**：回答前先从设定文档检索相关片段，拼进 system prompt，让 NPC "引用"真实设定 |
| **只会聊天** | NPC 聊得再好也触达不到游戏状态，玩家问"我包里有什么"它只能编 | **Function Calling**：LLM 自主决定调用背包/任务/等级接口，代码执行后把真实结果回喂模型 |

一句话定位：**让 NPC 既说对话，又说得对，还能动手。**

---

## 二、功能清单

### 1. 多角色人格系统（配置驱动）
- 三个 NPC：罗杰商人（市井贪财）、艾伦战士（豪爽刚烈）、莉莉法师（博学掉书袋）
- 每个角色一个 JSON 配置文件（`npc_config/*.json`），含 `id / name / role / system_prompt`
- **新增 NPC 只需丢一个 JSON 进目录**，无需改动任何代码 —— 前端下拉框自动加载

### 2. RAG 知识检索（`rag.py`）
- 设定文档 `lore.txt`（艾瑟拉大陆：三大势力、地点、武器数值、货币汇率、任务系统、人物档案、魔物图鉴）
- **切块**：先按空行分段，超长段再用滑动窗口切（max_len=200，overlap=50），重叠避免切断跨边界语义
- **向量化**：BAAI/bge-small-zh-v1.5，中文语义检索效果优于通用多语言模型
- **索引**：FAISS `IndexFlatIP` + 归一化向量（内积等价于余弦相似度），检索 top_k=3
- **国内网络适配**：默认走 HF 镜像 `hf-mirror.com`，失败自动 fallback 到 ModelScope 魔搭社区
- 支持直接 `python rag.py` 跑端到端检索测试

### 3. Function Calling / Agent 循环（`dialogue.py` + `tools.py`）
工具集让 LLM 自己判断要不要用、用哪个：

| 工具 | 作用 |
|---|---|
| `check_inventory` | 查询玩家背包 |
| `give_quest(quest_id)` | 向玩家发放任务 |
| `check_player_level` | 查询玩家等级 |

- **两轮调用设计**：第一轮带 tools 让模型决策 → 执行工具 → 结果回喂 → **第二轮不传 tools 强制收敛**，杜绝工具调用死循环
- 工具以 OpenAI/DeepSeek 标准 schema 声明，与执行函数通过注册表解耦

### 4. 对话记忆（滑动窗口）
- `max_turns=5`，只保留最近 5 轮历史，**控制 token 成本随对话线性增长**
- 前端同步只渲染最近 10 条

### 5. 中英双语
- 玩家侧一键切换，通过 system prompt 追加语言指令实现，**同一份设定文档同时服务中英文提问**

### 6. Streamlit 交互前端（`main.py`）
- 角色下拉选择 + 语言切换 + 输入框 + 对话历史渲染，热加载所有 NPC 配置

---

## 三、技术架构

```
玩家输入
   │
   ├─► RAG 检索 ──► lore.txt 切块 ──► BGE 向量化 ──► FAISS top_k=3
   │                                                      │
   │                                                      ▼
   ├──────────────────► system_prompt = 人格 + 语言指令 + 设定上下文
   │                                                      │
   ▼                                                      ▼
对话历史（最近5轮）──────────────► DeepSeek API（第1轮，带 tools）
                                          │
                            ┌─────────────┴─────────────┐
                       无 tool_calls              有 tool_calls
                            │                           │
                            │                  执行背包/任务/等级
                            │                           │
                            │                 DeepSeek API（第2轮，不带 tools）
                            │                           │
                            └─────────────┬─────────────┘
                                          ▼
                                      NPC 回复
```

**模块职责**

| 文件 | 职责 | 行数 |
|---|---|---|
| `main.py` | Streamlit 前端入口，自动加载 NPC 配置 | ~40 |
| `dialogue.py` | NPC 类：RAG 拼接 + 工具调用循环 + 记忆管理 | ~70 |
| `rag.py` | 切块 / 向量化 / FAISS 索引 / 检索 | ~146 |
| `tools.py` | 工具函数 + JSON schema + 调度 | ~74 |
| `api_client.py` | DeepSeek API 封装（含 tool_calls 返回） | ~41 |
| `lore.txt` | 游戏世界观知识库 | ~29 |
| `npc_config/*.json` | 三个 NPC 人格配置 | 3 文件 |
| `世界观.docx` | 设定原文（切块前的原始素材） | — |

---

## 四、项目亮点（作品集可强调的部分）

1. **不是聊天机器人，是"有 grounded 约束 + 有手"的 Agent**
   单纯接 LLM 是最低门槛；这里同时落地了 RAG 与 Function Calling 两条主线，且两者协同。

2. **工具调用循环有防失控设计**
   第二轮主动摘掉 tools 参数强制模型收敛 —— 这是很多人写 Agent 循环时踩的死循环坑，说明理解的是机制不是抄的模板。

3. **中英双语 + 中文向量模型选型**
   用 `bge-small-zh-v1.5` 而非通用多语言模型，中文语义检索精度更好；small 版本本地可跑、零推理成本，符合游戏场景的成本意识。

4. **配置驱动的角色扩展**
   加 NPC = 加 JSON，零改代码。这点在作品集里可以直接演示"现场加一个角色"。

5. **工程细节有成本意识**
   滑动窗口记忆、small 模型、`max_turns` 可调 —— 不是只管跑通，而是在控制 token 开销。

6. **本地化适配**
   HF 镜像 + ModelScope 双通道 fallback，解决国内下载模型被墙的真实痛点。

---

## 五、运行方式

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY="your_key"     # Windows: set DEEPSEEK_API_KEY=your_key
streamlit run main.py
```

> **密钥安全**：`api_client.py` 只从环境变量读 key，不再硬编码到源码。
> 切勿把真实 key 提交到 GitHub —— 一旦泄露需立即到 DeepSeek 后台轮换。

单独测试检索链路（不依赖密钥）：

```bash
python rag.py
```

---

## 五之二、版本迭代中做过的优化

| 优化项 | 改动 | 收益 |
|---|---|---|
| **密钥安全** | `api_client.py` 移除硬编码 key，改为纯环境变量；调用时懒校验并给出明确报错 | 杜绝密钥随代码泄露，且 RAG 独立测试不再被密钥缺失阻断 |
| **工具循环去重** | 删除 `tools.py` 中重复的 `dispatch()`，统一为 `execute_tool_calls()`，由 `dialogue.py` 复用 | 单一工具执行入口，源码更干净；单工具抛错不影响其余 |
| **索引持久化** | `rag.py` 把 FAISS 索引与切块落盘到 `.rag_cache/`，按 `lore.txt` 修改时间命中缓存 | 二次启动跳过切块+向量化，启动更快（已加入 .gitignore） |
| **对话历史分桶** | `main.py` 历史按 NPC 名字分桶存储 | 修复切换角色后旧对话被错标为新角色名的小 bug |

---

## 六、已知限制与后续方向

**限制**
- 工具数据为模拟返回，未接真实游戏后端
- FAISS 索引在导入时重建，未做持久化，设定文档变大后启动会变慢
- 无 API 失败重试 / 无流式输出，长回复有等待感
- 无单元测试；`tools.py` 的 `dispatch()` 为早期实现，`dialogue.py` 内有独立循环，存在冗余

**后续方向**
- 索引落盘 + 增量更新，支持大规模设定文档
- 接真实游戏数据层（SQLite 背包/任务表）
- 长期记忆：把关键事件摘要存库，突破滑动窗口的遗忘问题
- 情感/好感度系统：NPC 对玩家态度随交互演化
- 流式输出 + 失败重试，提升交互体验

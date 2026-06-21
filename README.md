# LangGraph 全特性演示 — 智能旅行规划助手

基于 LangGraph 构建的生产级 Agent 项目，覆盖框架**全部 21 项核心特性**，以「旅行规划」业务场景串起一条完整链路：需求理解 → 并行研究 → 计划生成 → 工具调用 → 预算评估 → 人工审批 → 反思循环。

## 特性覆盖清单

| # | 特性 | 项目落点 |
|---|------|---------|
| 1 | `StateGraph` + `TypedDict` | [state.py](state.py) 主图状态 |
| 2 | `Annotated` 自定义 Reducer | `add_messages` / `operator.add` / 自定义 `merge_plan` |
| 3 | `add_node` / `add_edge` | [graph.py](graph.py) |
| 4 | `add_conditional_edges` | route_fanout / route_after_plan / route_approval |
| 5 | 循环（Cycle） | plan → revise → plan 反思循环 |
| 6 | `Command(goto=, update=)` | approval / revise 节点内动态路由 |
| 7 | `Send` Map-Reduce 并行扇出 | 并行研究多个目的地 |
| 8 | 子图（compiled graph as node） | research StateGraph 子图 |
| 9 | `interrupt()` + `Command(resume=)` | approval 人工审批 |
| 10 | `interrupt_before` 编译期断点 | demo 5 |
| 11 | Checkpointer（AsyncSqliteSaver） | SQLite 持久化，断点续跑 |
| 12 | `BaseStore` 长期记忆（跨 thread） | 用户偏好读写 + store.search |
| 13 | `RetryPolicy` | tools 节点 max_attempts=5 |
| 14 | `timeout` | plan 节点 60s 超时保护 |
| 15 | 自定义 Tool + `ToolNode` + `tools_condition` | 天气/景点/预算 3 个工具 |
| 16 | `aget_state_history` 时间旅行 | demo 3 |
| 17 | `aupdate_state` 手动改状态 | demo 4 |
| 18 | `input_schema` ≠ `output_schema` | TravelInput / TravelOutput |
| 19 | 多种 `stream_mode` | astream(stream_mode="updates") |
| 20 | Functional API（@entrypoint / @task） | budget 子图 |
| 21 | 并行执行（异步运行时） | AsyncSqliteSaver + ainvoke / astream |

## 架构

```
                      ┌─────────────────────────────────────────────┐
                      │                                             ▼
START → understand → plan ──Send×N──→ research(dst_i) ──→ aggregate → budget_subgraph
           │                       (并行 Map-Reduce)                        │
           │                                                                  ▼
           │                                            ┌─ interrupt 人工审批 ──┐
           │                                            │                      │
           │                                       approve                 revise → 回到 plan
           │                                            │
           │                                            ▼
           └─────────────────────────────────────→ execute_tools → END
```

## 项目结构

```
LangGraph_Demo/
├── config.py            # DeepSeek LLM 工厂 + 常量
├── state.py             # 状态定义 + 3 种自定义 Reducer
├── tools.py             # 自定义工具（天气/景点/预算，含模拟失败）
├── store.py             # BaseStore 长期记忆封装
├── subgraphs/
│   ├── __init__.py
│   ├── research.py      # StateGraph 子图（gather → summarize）
│   └── budget.py        # Functional API 子图（@entrypoint + @task）
├── nodes.py             # 主图节点（含 interrupt / Command 动态路由）
├── graph.py             # 主图组装 + AsyncSqliteSaver
├── test_offline.py      # 离线冒烟测试（Mock LLM，无需 API Key）
├── run_demo.py          # 端到端 5 场景演示
├── requirements.txt
├── .env.example
└── README.md
```

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 配置 API Key（仅 run_demo.py 需要）

```powershell
copy .env.example .env
```

编辑 `.env`，填入 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=sk-your-key-here
```

### 3. 运行

**离线冒烟测试（无需 API Key）：**

```powershell
python test_offline.py
```

用 Mock LLM 验证图的控制流：Send 并行扇出 → Map-Reduce 汇聚 → interrupt 暂停 → Command(resume=) 续跑 → revise 反思循环。

**端到端演示（需 API Key）：**

```powershell
python run_demo.py
```

依次演示 5 个场景：

| 序号 | 场景 | 核心特性 |
|------|------|---------|
| 1 | 主流程 | Send 并行研究 / 流式 / 审批 / 反思循环 |
| 2 | 长期记忆 | BaseStore 跨 thread 读写 + search |
| 3 | 时间旅行 | aget_state_history 回看历史状态 |
| 4 | 手动改状态 | aupdate_state 注入修正 |
| 5 | 编译期断点 | interrupt_before=["plan"] |

## 配置项

`config.py` 中可调整：

```python
BUDGET_LIMIT = 8000.0     # 预算上限（元），超过触发人工审批
MAX_ITERATIONS = 3        # 反思循环最大次数
```

演示脚本中 `budget_limit` 若设很低（如 1000），则必然会触发审批 → 拒绝 → 反思 → 再审批的完整循环链路。

## 依赖

```
langgraph>=0.2.60
langgraph-checkpoint-sqlite>=2.0.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
python-dotenv>=1.0.0
```

## License

MIT

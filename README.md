# halluc-detect

> Agent 轨迹幻觉检测器 — HallucC Phase 1 开源旗舰。
> **fast 档纯 Python stdlib，0 LLM 调用 / 0 网络搜索，离线可复现。**

[English summary ↓ — 中文为主，文末附英文摘要]

---

## 这是什么

`halluc-detect` 是 [HallucC](https://aihcc.cloud) 的 **Phase 1 开源旗舰**：一个针对 LLM Agent 执行轨迹（trajectory）的幻觉检测器 + 中文回归基准。它把"Agent 会不会在不该自信的地方编造结果"这件事变成**可跑、可复现、0 成本**的自动化检测。

- **检测器**：对一条 Agent 轨迹（thought / action / action_input / observation / agent_claim 序列）跑分层递进评估，输出**六维分数 + 逐 步 checks + 12 类失败模式码 + 幻觉传播链**。
- **基准**：20 条预植入已知幻觉的中文 Agent 轨迹（HallucC-Agent-Bench），作为检出率 / 误报率自证背书——**期望标签是独立地面真值，不是检测器自己的输出**。

> **定位**：本仓库当前是一个**检测器 + 回归基准**，不是学术 novelty。。

## 为什么 fast 档是 0 依赖

fast 档只跑规则层：正则 + token 相似度 + 算式核验 + 常识矛盾 + 双语标记词表。规则层经刻意解耦——`import agent_pipeline` 不再拖入 `server`/`config`/`llm`/`search` 等重栈，所以：

```bash
pip install halluc-detect          # 不带任何 extra，即可跑 fast 档
halluc-detect bench                # 0 API quota，离线复现
```

`standard` / `deep` 档需 LLM 裁判层（见[可选依赖](#安装)），对应 HallucC 付费 SaaS。

## 安装

```bash
# 开发模式（仓库内）
git clone <repo> && cd hallucc
pip install -e .

# 仅 fast 档（0 依赖）
pip install halluc-detect

# 需 LLM 裁判层（standard / deep 档）
pip install "halluc-detect[llm]"

# 需 NLI 语义蕴含（observation↔claim 语义矛盾判定）
pip install "halluc-detect[nli]"
```

> **可编辑安装说明**：`pip install -e .` 在仓库根目录执行。`server` 端模块（`pipeline`/`config`/`llm`/`search`）不在 `py-modules` 内，保持 wheel 精简；开发仓库内仍可经 repo-root-on-path 访问全栈，不影响本地开发。

## 使用

### `detect` — 检测单条轨迹

```bash
# stdin 输入裸轨迹列表（tool_schemas 缺省 → 跳过 HALLUCINATED_TOOL/SCHEMA_VIOLATION）
echo '[
  {"step":1,"action":"search","action_input":"{\"query\":\"订单 12345\"}",
   "observation":"未找到订单","agent_claim":"已找到订单 12345 的详情"}
]' | halluc-detect detect

# 文件输入（可带 task / tool_schemas）
halluc-detect detect trajectory.json --speed fast

# 启用工具 schema 校验（加载基准 5 工具注册表）
halluc-detect detect trajectory.json --tools demo
```

每步字段兼容两种 schema（由 `agent_eval._normalize_step` 归一）：

| schema | 字段 |
|---|---|
| A | `step, thought, action, action_input, observation, final_answer` |
| B | `step, tool, tool_input, tool_output, agent_claim` |

输出为结构化 JSON（`evaluate_trajectory().to_dict()`）：六维 `dimensions`、逐 步 `steps[].checks`、`failure_modes` 列表、`propagation` 传播链、`llm_calls`/`search_calls`（fast 档均为 0）。

### `bench` — 跑中文基准

```bash
# 文本报告（默认 fast 档）
halluc-detect bench

# 结构化 JSON
halluc-detect bench --json

# standard 档（需 [llm] extra + API key）
halluc-detect bench --speed standard
```

fast 档基准验收门槛：`recall ≥ 0.80`、`fp_rate ≤ 0.15`（当前 recall=1.0 / fp_rate=0.0）。

## 失败模式码（12 类）

**确定性（fast 档可报，10 类）：**

| 码 | 含义 |
|---|---|
| `HALLUCINATED_TOOL` | 调用不在注册表的工具 |
| `PARAM_FABRICATION` | 参数含刚性 ID（订单号/UUID）但上下文无来源 |
| `SCHEMA_VIOLATION` | 参数违反工具 JSON Schema |
| `OBSERVATION_IGNORED` | 观察显式失败 / 为空却声称成功 |
| `CLAIM_MISMATCH` | 观察与声称不符（相似度低于阈值 / 算式算错 / NLI 矛盾） |
| `LOOP` | 相同 action+input 重复 |
| `RETRY_STORM` | 同工具 ≥3 个不同入参且含失败观察 |
| `REDUNDANT_STEP` | 不同调用但观察高度重复（≥0.8） |
| `HANDOFF_LOSS` | 后续 claim 纠正 / 推翻前序 claim |
| `OVERCONFIDENT_ASSERTION` | 常识矛盾（日期不可能 / 百分比>100 / 年份悖论） |

**LLM-only（standard / deep 档，2 类）：**`SCOPE_DRIFT`（中途跑题）、`PREMATURE_STOP`（最终回答未兑现任务）。

## 差异化

- **双语标记词表**：CN literals 原样保留 + EN 词边界匹配（`\b` 避免 "foundry"/"nullable" 误匹）+ JSON/HTTP 上下文 regex（`"status": 503` 避免裸数字误报算式结果）。
- **中文多领域基准**：20 条覆盖订单 / 搜索 / 计算 / 翻译 / 天气等场景的预植入轨迹，**CN-first**（现有 Agent 幻觉基准偏 EN）。
- **分层递进 + 控误报**：证据不足 → `unverified`，不强行 `refuted`（与 HallucC 主线一致）。规则层只把"不确定"的项交给裁判层，避免全量 LLM 调用。


**Phase 2（规划中）**：公开排行榜（检测器轴 + Agent 轴）。

## 许可证

- **代码**：MIT（见 `LICENSE`）。
- **基准数据集**（20 条植入轨迹 + 期望标签）：CC-BY-4.0。

> 两层许可：代码 MIT 便于工具被引用与二次开发；数据集 CC-BY-4.0 保留署名要求。



## English summary

`halluc-detect` is the open-source Phase 1 flagship of **HallucC** — a hallucination detector for LLM **agent execution trajectories** (not just text claims) plus a Chinese regression benchmark.

- **Fast tier is pure-Python stdlib**: 0 LLM calls, 0 web search, offline-reproducible. The rule layer was deliberately decoupled so `import agent_pipeline` pulls no server/config/llm/search stack.
- **Benchmark (HallucC-Agent-Bench)**: 20 Chinese agent trajectories with planted, ground-truth hallucinations across 10 deterministic failure modes. Expected labels are independent ground truth, not detector output.
- **12 failure-mode codes**: 10 deterministic (fast tier) + 2 LLM-only (`SCOPE_DRIFT`, `PREMATURE_STOP`).
- **Differentiation**: bilingual (CN+EN) marker lexicon with word-boundary + JSON-scoped regex matching; CN-first multi-domain benchmark; tiered evaluation that defaults to `unverified` rather than false-positive `refuted`.
- **Funnel**: free fast tier (this repo) → standard/deep LLM-judge tiers → HallucC paid SaaS. Phase 2 will add a public leaderboard.

```bash
pip install halluc-detect
halluc-detect bench        # recall=1.0, fp_rate=0.0, 0 API quota
```

Licenses: code = MIT, benchmark dataset = CC-BY-4.0.

— *Phase 1 of [HallucC](https://aihcc.cloud).*

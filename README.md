# halluc-detect

> Agent 轨迹幻觉检测器 — HallucC Phase 1 开源旗舰。
> **fast 档纯 Python stdlib，0 LLM 调用 / 0 网络搜索，离线可复现。**

[English summary ↓ — 中文为主，文末附英文摘要]

---

## 这是什么

`halluc-detect` 是 [HallucC](https://aihcc.cloud) 的 **Phase 1 开源旗舰**：一个针对 LLM Agent 执行轨迹（trajectory）的幻觉检测器 + 中文回归基准。它把"Agent 会不会在不该自信的地方编造结果"这件事变成**可跑、可复现、0 成本**的自动化检测。

- **检测器**：对一条 Agent 轨迹（thought / action / action_input / observation / agent_claim 序列）跑分层递进评估，输出**六维分数 + 逐 步 checks + 12 类失败模式码 + 幻觉传播链**。
- **基准**：100 条中文 Agent 轨迹（HallucC-Agent-Bench）= 20 契约回归（planted 已知幻觉，recall=1.0 回归门）+ 80 对抗（真实分布，量化泛化边界）。fast 档 0 LLM / 0 搜索，真实 recall=0.945 / fp_rate=0.0。
- **独立循环质疑**：独立 LLM-as-judge（DeepSeek temp=0，盲标——不见作者标签、不见检测器检出）复核全 100 条，确认 ~86% 幻觉标签 + 100% 干净（specificity=1.0，零误报），裁定「循环风险低」——打破「用例作者=检测器作者=标注者」的循环质疑。

> **定位**：本仓库当前是一个**检测器 + 回归基准**，不是学术 novelty。

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

> **可编辑安装说明**：`pip install -e .` 在仓库根目录执行。wheel 打包 13 个开源模块（含精简版 `llm.py` 客户端，见 `pyproject.toml` 的 `py-modules`）。**open-core 边界**：生产裁判 prompt（`prompts.py`）与 `pipeline`/`config`/`search`/`domains` 等服务端模块不在此仓库——`import agent_pipeline` 顶层不拖入它们，fast 档零依赖；standard/deep 裁判层在 SaaS 侧运行。

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

fast 档基准验收门槛：`recall ≥ 0.80`、`fp_rate ≤ 0.15`。当前 fast 档（100 条）：

| 套件 | n | recall | fp_rate | 说明 |
|---|---|---|---|---|
| 契约回归 | 20 | 1.0 | 0.0 | planted 已知幻觉，回归门（by design）|
| 对抗 | 80 | 0.937 | 0.0 | 真实分布；FN=5 全在 COVERAGE_GAP（LLM-only 模式，fast 档设计不覆盖）|
| 合计 | 100 | 0.945 | 0.0 | 0 LLM / 0 搜索 |

落盘快照见 `eval_set/agent_benchmark_results.json`（含 standard 档子集 20，recall=1.0）。

### `judge` — 独立循环质疑探针

```bash
# 独立 LLM judge 盲标 100 条（需 [llm] extra + API key）
python agent_benchmark_judge.py
# 抽样冒烟
python agent_benchmark_judge.py --limit 20
```

judge 仅见 `{task, trajectory}` + 工具注册表，不见作者 `expected_failure_modes`、不见检测器检出，独立判断每步是否 grounded。产出三方一致率（judge↔作者标签、judge↔作者码、judge↔检测器）+ 模板有效证据去重 + 循环风险裁定。

当前快照（`eval_set/judge_relabel.json`，DeepSeek temp=0）：judge recall=0.857 / specificity=1.0（TP=78 FN=13 TN=9 FP=0），模板冗余 100→41 独立证据，循环风险裁定**低**。DeepSeek temp=0 非完全确定，单次运行 FN 有 ±1 噪声底。

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
- **中文多领域基准**：100 条覆盖订单 / 搜索 / 计算 / 翻译 / 天气 / 医疗 / 金融 / 物流等场景的轨迹（20 契约 + 80 对抗），**CN-first**（现有 Agent 幻觉基准偏 EN）。
- **循环质疑自证**：独立 judge 盲标打破「检测器抓自家模板」质疑——若属真，recall 应趋 0，而非实测 0.857。
- **分层递进 + 控误报**：证据不足 → `unverified`，不强行 `refuted`（与 HallucC 主线一致）。规则层只把"不确定"的项交给裁判层，避免全量 LLM 调用。


**Phase 2（规划中）**：公开排行榜（检测器轴 + Agent 轴）。

## 许可证

- **代码**：MIT（见 `LICENSE`）。
- **基准数据集**（HallucC-Agent-Bench 100 条轨迹用例 + 期望标签）：CC-BY-4.0。

> 两层许可：代码 MIT 便于工具被引用与二次开发；数据集 CC-BY-4.0 保留署名要求。



## English summary

`halluc-detect` is the open-source Phase 1 flagship of **HallucC** — a hallucination detector for LLM **agent execution trajectories** (not just text claims) plus a Chinese regression benchmark.

- **Fast tier is pure-Python stdlib**: 0 LLM calls, 0 web search, offline-reproducible. The rule layer was deliberately decoupled so `import agent_pipeline` pulls no server/config/llm/search stack.
- **Benchmark (HallucC-Agent-Bench)**: 100 Chinese agent trajectories = 20 contract regression (planted ground-truth hallucinations, recall=1.0 gate) + 80 adversarial (real distribution). Fast tier real recall=0.945 / fp_rate=0.0, 0 LLM / 0 search.
- **Independent circularity probe**: a blind LLM-as-judge (DeepSeek temp=0, sees only {task, trajectory} — not author labels, not detector output) re-labels all 100 cases, confirming ~86% of hallucination labels + 100% of clean (specificity=1.0, zero false positives) and a **low** circularity verdict — refuting the "detector catches its own templates" concern (if true, recall would approach 0, not 0.857).
- **12 failure-mode codes**: 10 deterministic (fast tier) + 2 LLM-only (`SCOPE_DRIFT`, `PREMATURE_STOP`).
- **Differentiation**: bilingual (CN+EN) marker lexicon with word-boundary + JSON-scoped regex matching; CN-first multi-domain benchmark; tiered evaluation that defaults to `unverified` rather than false-positive `refuted`.
- **Funnel**: free fast tier (this repo) → standard/deep LLM-judge tiers → HallucC paid SaaS. Phase 2 will add a public leaderboard.

```bash
pip install halluc-detect
halluc-detect bench        # 100 cases: contract recall=1.0 (gate), adversarial recall=0.937, 0 API quota
```

Licenses: code = MIT, benchmark dataset = CC-BY-4.0.

— *Phase 1 of [HallucC](https://aihcc.cloud).*

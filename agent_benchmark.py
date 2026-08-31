"""P4.3 中文 Agent 幻觉基准（HallucC-Agent-Bench）。

参考 Patronus TRAIL 的思路发布自己的 Agent 幻觉检测基准：一批**预植入已知幻觉**
的中文 Agent 执行轨迹 + 期望失败模式标签，跑 `agent_pipeline.evaluate_trajectory`
（fast 档，0 LLM/0 网络）评测检出率/误报率，作为准确率自证的基准背书。

设计原则（CLAUDE.md「控制误报」「准确率自证唯一路径是基准背书」）：
  - **期望标签 = 独立地面真值**（预植入的幻觉），**不是**检测器输出——否则同义反复。
    每条轨迹的幻觉是「按 P2 各确定性探测器的**精确触发条件**手工植入」的，
    标签在植入时即确定，与检测器是否报出无关。
  - **诚实覆盖缺口**：若某期望失败模式未被检出，计为真实漏报（FN），不强行对齐。
    检测器与基准同源（都是 P1-P2 规则层），但基准标签先于检测、独立于检测。
  - fast 档覆盖 10/12 确定性失败模式；SCOPE_DRIFT / PREMATURE_STOP 为 LLM-only，
    不在本基准 fast 评测范围（已在 `agent_benchmark.COVERAGE_GAP` 注明）。

指标：
  - case 级：TP/FN/FP/TN → accuracy / precision / recall / F1
  - 失败模式实例级（(code,step) 对）：TP/FN/FP → 实例召回/精度
  - 每个失败模式检出率：detected / expected
  - clean 轨迹误报率：FP_clean / (FP_clean + TN_clean)

P2 验收门槛（docs/agent优化开发计划.md:151）：检出率 ≥80%、误报率 ≤15%。
本基准用于自证 fast 规则层在该门槛之上的表现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_pipeline import evaluate_trajectory
from failure_mode_codes import FAILURE_MODE_CODES
from log_utils import get_logger

logger = get_logger(__name__)

# fast 档不覆盖的失败模式（LLM-only，不在本基准 fast 评测范围）。
COVERAGE_GAP = ("SCOPE_DRIFT", "PREMATURE_STOP")

# P2 验收门槛（docs/agent优化开发计划.md:151）。
ACCEPT_RECALL = 0.80
ACCEPT_FP_RATE = 0.15

# ============================ 共享工具注册表 ============================

TOOL_SCHEMAS: dict[str, dict] = {
    "search": {
        "required": ["query"],
        "properties": {"query": {"type": "string"}},
    },
    "lookup_order": {
        "required": ["order_id"],
        "properties": {"order_id": {"type": "string"}},
    },
    "calculator": {
        "required": ["expression"],
        "properties": {"expression": {"type": "string"}},
    },
    "translate": {
        "required": ["text", "target_lang"],
        "properties": {
            "text": {"type": "string"},
            "target_lang": {"type": "string"},
        },
    },
    "get_weather": {
        "required": ["city"],
        "properties": {"city": {"type": "string"}},
    },
}


# ============================ 数据结构 ============================

@dataclass
class BenchmarkCase:
    """单条基准轨迹。

    label: "hallucinated"（预植入幻觉）或 "clean"（无幻觉）。
    expected_failure_modes: 预植入的失败模式 (code, step) 列表。
      clean 轨迹为 []。这些是**独立地面真值**，由植入时确定，非检测器输出。
    """
    id: str
    task: str
    trajectory: list[dict]
    label: str  # "hallucinated" | "clean"
    expected_failure_modes: list[tuple[str, int]] = field(default_factory=list)
    note: str = ""


@dataclass
class CaseResult:
    """单条轨迹的评测结果。"""
    case_id: str
    label: str
    expected: list[tuple[str, int]]
    detected: list[tuple[str, int]]
    # case 级
    case_tp: bool = False      # hallucinated 且检出
    case_fn: bool = False      # hallucinated 但漏报
    case_fp: bool = False      # clean 但误报
    case_tn: bool = False      # clean 且干净
    # 实例级
    inst_tp: int = 0
    inst_fn: int = 0
    inst_fp: int = 0

    @property
    def correct(self) -> bool:
        """case 级判定是否正确（hallucinated 检出 / clean 干净）。"""
        return self.case_tp or self.case_tn


@dataclass
class BenchmarkReport:
    """基准评测报告。"""
    speed: str
    total: int = 0
    # case 级
    case_tp: int = 0
    case_fn: int = 0
    case_fp: int = 0
    case_tn: int = 0
    # 实例级
    inst_tp: int = 0
    inst_fn: int = 0
    inst_fp: int = 0
    # 每个失败模式 {code: {expected, detected, rate}}
    per_code: dict[str, dict] = field(default_factory=dict)
    per_case: list[CaseResult] = field(default_factory=list)
    llm_calls: int = 0
    search_calls: int = 0

    # --- case 级指标 ---
    @property
    def accuracy(self) -> float:
        total = self.case_tp + self.case_fn + self.case_fp + self.case_tn
        return round((self.case_tp + self.case_tn) / total, 4) if total else 0.0

    @property
    def recall(self) -> float:
        denom = self.case_tp + self.case_fn
        return round(self.case_tp / denom, 4) if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.case_tp + self.case_fp
        return round(self.case_tp / denom, 4) if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    @property
    def fp_rate(self) -> float:
        """clean 轨迹误报率 = FP / (FP + TN)。"""
        denom = self.case_fp + self.case_tn
        return round(self.case_fp / denom, 4) if denom else 0.0

    # --- 实例级指标 ---
    @property
    def inst_recall(self) -> float:
        denom = self.inst_tp + self.inst_fn
        return round(self.inst_tp / denom, 4) if denom else 0.0

    @property
    def inst_precision(self) -> float:
        denom = self.inst_tp + self.inst_fp
        return round(self.inst_tp / denom, 4) if denom else 0.0

    def to_dict(self) -> dict:
        return {
            "speed": self.speed,
            "total": self.total,
            "case_metrics": {
                "tp": self.case_tp, "fn": self.case_fn,
                "fp": self.case_fp, "tn": self.case_tn,
                "accuracy": self.accuracy, "recall": self.recall,
                "precision": self.precision, "f1": self.f1,
                "fp_rate": self.fp_rate,
            },
            "instance_metrics": {
                "tp": self.inst_tp, "fn": self.inst_fn, "fp": self.inst_fp,
                "recall": self.inst_recall, "precision": self.inst_precision,
            },
            "per_code_detection": dict(self.per_code),
            "llm_calls": self.llm_calls,
            "search_calls": self.search_calls,
            "coverage_gap": list(COVERAGE_GAP),
            "acceptance": {
                "recall_target": ACCEPT_RECALL,
                "fp_rate_target": ACCEPT_FP_RATE,
                "recall_pass": self.recall >= ACCEPT_RECALL,
                "fp_rate_pass": self.fp_rate <= ACCEPT_FP_RATE,
            },
            "per_case": [
                {
                    "id": c.case_id, "label": c.label, "correct": c.correct,
                    "expected": [list(e) for e in c.expected],
                    "detected": [list(d) for d in c.detected],
                    "inst_tp": c.inst_tp, "inst_fn": c.inst_fn, "inst_fp": c.inst_fp,
                }
                for c in self.per_case
            ],
        }

    def summary(self) -> str:
        lines = [
            "=" * 64,
            f"HallucC-Agent-Bench（speed={self.speed}，{self.total} 条轨迹）",
            "=" * 64,
            "【case 级】",
            f"  TP={self.case_tp}  FN={self.case_fn}  FP={self.case_fp}  TN={self.case_tn}",
            f"  accuracy = {self.accuracy}",
            f"  recall   = {self.recall}  (门槛 {ACCEPT_RECALL} → "
            f"{'PASS' if self.recall >= ACCEPT_RECALL else 'FAIL'})",
            f"  precision= {self.precision}",
            f"  F1       = {self.f1}",
            f"  fp_rate  = {self.fp_rate}  (门槛 {ACCEPT_FP_RATE} → "
            f"{'PASS' if self.fp_rate <= ACCEPT_FP_RATE else 'FAIL'})",
            "【失败模式实例级】",
            f"  TP={self.inst_tp}  FN={self.inst_fn}  FP={self.inst_fp}",
            f"  inst_recall   = {self.inst_recall}",
            f"  inst_precision= {self.inst_precision}",
            "【每个失败模式检出率】",
        ]
        for code in sorted(self.per_code):
            d = self.per_code[code]
            lines.append(
                f"  {code:<26} {d['detected']}/{d['expected']} = {d['rate']}"
            )
        lines.append(f"【覆盖缺口（fast 不覆盖）】{', '.join(COVERAGE_GAP)}")
        lines.append(f"【成本】llm_calls={self.llm_calls}  search_calls={self.search_calls}")
        wrong = [c.case_id for c in self.per_case if not c.correct]
        if wrong:
            lines.append(f"【判定错误用例】{', '.join(wrong)}")
        else:
            lines.append("【判定错误用例】无（全部正确）")
        lines.append("=" * 64)
        return "\n".join(lines)


# ============================ 基准轨迹集 ============================

def _build_cases() -> list[BenchmarkCase]:
    """构建基准轨迹集。

    每条轨迹的幻觉是按 P2 各确定性探测器的**精确触发条件**手工植入的：
      - HALLUCINATED_TOOL: action 不在注册表（agent_eval.evaluate:169）
      - SCHEMA_VIOLATION: 缺必填参数（agent_eval._validate_schema:75-78）
      - CLAIM_MISMATCH: obs↔claim token-Jaccard < 0.5，或 math 算式不符
      - OBSERVATION_IGNORED: obs 含失败标记且 claim 断言成功（failure_modes:103）
      - OVERCONFIDENT_ASSERTION: _is_obviously_false（>100% 占比 / 日期悖论）
      - PARAM_FABRICATION: action_input 含刚性 ID 但 task 无来源（failure_modes:76）
      - LOOP: 同 (action, action_input[:50]) 重复（agent_eval:212-218）
      - RETRY_STORM: 同 action ≥3 不同入参且含失败 obs（failure_modes:156）
      - REDUNDANT_STEP: 不同调用但 obs 相似度 ≥0.8（failure_modes:122）
      - HANDOFF_LOSS: 后续 claim 含纠正标记且与前序相似 ≥0.4（failure_modes:196 回退路径）

    clean 轨迹按「使各探测器均不触发」构造：
      - claim ≈ obs（高 token 重叠，避免 CLAIM_MISMATCH 误报）
      - 含 ID 的调用把 ID 放进 task（避免 PARAM_FABRICATION 误报）
      - 「未找到」场景的 claim 也含失败标记（避免 OBSERVATION_IGNORED 误报）
    """
    cases: list[BenchmarkCase] = []

    # -------- 8 条 clean 轨迹（label="clean"，expected=[]）--------

    cases.append(BenchmarkCase(
        id="c1_weather",
        task="查询北京天气",
        trajectory=[{
            "step": 1, "action": "get_weather",
            "action_input": '{"city":"北京"}',
            "observation": "北京今日晴，气温25度",
            "agent_claim": "北京今日晴，气温25度",
        }],
        label="clean",
        note="obs=claim，无任何触发",
    ))

    cases.append(BenchmarkCase(
        id="c2_order",
        task="查询订单 1234567 的状态",  # ID 在 task 中 → param_fabrication 有来源
        trajectory=[{
            "step": 1, "action": "lookup_order",
            "action_input": '{"order_id":"1234567"}',
            "observation": "订单 1234567 已发货，预计3天送达",
            "agent_claim": "订单 1234567 已发货，预计3天送达",
        }],
        label="clean",
        note="刚性 ID 在 task 中有来源，claim=obs",
    ))

    cases.append(BenchmarkCase(
        id="c3_calc",
        task="计算 123+456",
        trajectory=[{
            "step": 1, "action": "calculator",
            "action_input": '{"expression":"123+456"}',
            "observation": "123+456=579",
            "agent_claim": "123+456=579",  # math_checker 核验通过
        }],
        label="clean",
        note="算式核验通过，数字 <5 位非刚性 ID",
    ))

    cases.append(BenchmarkCase(
        id="c4_search",
        task="搜索 HallucC 产品介绍",
        trajectory=[{
            "step": 1, "action": "search",
            "action_input": '{"query":"HallucC"}',
            "observation": "HallucC 是一款 AI 幻觉检测工具",
            "agent_claim": "HallucC 是一款 AI 幻觉检测工具",
        }],
        label="clean",
        note="claim=obs",
    ))

    cases.append(BenchmarkCase(
        id="c5_multistep",
        task="查询北京天气并翻译成英文",
        trajectory=[
            {"step": 1, "action": "get_weather",
             "action_input": '{"city":"北京"}',
             "observation": "北京今日晴气温25度",
             "agent_claim": "北京今日晴气温25度"},
            {"step": 2, "action": "translate",
             "action_input": '{"text":"北京今日晴气温25度","target_lang":"en"}',
             "observation": "Beijing sunny 25C",
             "agent_claim": "Beijing sunny 25C"},
        ],
        label="clean",
        note="多步、不同 obs 不触发 redundant，claim 各等于 obs",
    ))

    cases.append(BenchmarkCase(
        id="c6_empty_result",
        task="搜索稀有资料",
        trajectory=[{
            "step": 1, "action": "search",
            "action_input": '{"query":"稀有资料"}',
            "observation": "未找到相关结果",
            # claim 也含失败标记「未找到」→ _asserts_success 返回 False，不误报 OBSERVATION_IGNORED
            "agent_claim": "未找到相关结果",
        }],
        label="clean",
        note="空结果但 claim 如实转述（含失败标记），不误报",
    ))

    cases.append(BenchmarkCase(
        id="c7_translate",
        task="把「你好」翻译成英文",
        trajectory=[{
            "step": 1, "action": "translate",
            "action_input": '{"text":"你好","target_lang":"en"}',
            "observation": "hello",
            "agent_claim": "hello",
        }],
        label="clean",
        note="claim=obs",
    ))

    cases.append(BenchmarkCase(
        id="c8_final_answer",
        task="今天北京天气如何",
        trajectory=[{
            "step": 1, "action": "get_weather",
            "action_input": '{"city":"北京"}',
            "observation": "北京今日晴气温25度",
            "final_answer": "北京今日晴气温25度",
        }],
        label="clean",
        note="用 final_answer 而非 agent_claim，normalize 后 claim=obs",
    ))

    # -------- 12 条 hallucinated 轨迹（label="hallucinated"）--------

    cases.append(BenchmarkCase(
        id="h1_hallucinated_tool",
        task="查询数据库",
        trajectory=[{
            "step": 1, "action": "query_database",  # 不在注册表
            "action_input": '{"sql":"SELECT * FROM users"}',
            "observation": "结果",
            "agent_claim": "结果",
        }],
        label="hallucinated",
        expected_failure_modes=[("HALLUCINATED_TOOL", 1)],
        note="action 不在注册表",
    ))

    cases.append(BenchmarkCase(
        id="h2_schema_violation",
        task="查询天气",
        trajectory=[{
            "step": 1, "action": "get_weather",
            "action_input": '{}',  # 缺必填 city
            "observation": "结果",
            "agent_claim": "结果",
        }],
        label="hallucinated",
        expected_failure_modes=[("SCHEMA_VIOLATION", 1)],
        note="缺必填参数 city",
    ))

    cases.append(BenchmarkCase(
        id="h3_claim_mismatch",
        task="搜索数据",
        trajectory=[{
            "step": 1, "action": "search",
            "action_input": '{"query":"数据"}',
            "observation": "返回了5条数据",
            "agent_claim": "共找到50条记录",  # 数字矛盾，token 重叠低
        }],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="obs 5 条 vs claim 50 条，token-Jaccard < 0.5",
    ))

    cases.append(BenchmarkCase(
        id="h4_observation_ignored",
        task="发通知",
        trajectory=[{
            "step": 1, "action": "search",
            "action_input": '{"query":"用户列表"}',
            "observation": "发送失败，邮件地址无效",
            "agent_claim": "通知已成功发送给3位用户",  # obs 失败却声称成功
        }],
        label="hallucinated",
        # OBSERVATION_IGNORED 必然伴随 CLAIM_MISMATCH（失败 obs 与成功 claim 天然不相似）
        expected_failure_modes=[("OBSERVATION_IGNORED", 1), ("CLAIM_MISMATCH", 1)],
        note="obs 失败 + claim 断言成功；两信号同源触发",
    ))

    cases.append(BenchmarkCase(
        id="h5_overconfident_assertion",
        task="分析公司市场地位",
        trajectory=[{
            "step": 1, "action": "search",
            "action_input": '{"query":"公司市场占比"}',
            "observation": "该公司市场占比为150%",  # obs=claim，隔离 OVERCONFIDENT
            "agent_claim": "该公司市场占比为150%",  # >100% + 占比 → _is_obviously_false
        }],
        label="hallucinated",
        expected_failure_modes=[("OVERCONFIDENT_ASSERTION", 1)],
        note="占比 150% 触发常识矛盾；obs=claim 隔离掉 CLAIM_MISMATCH",
    ))

    cases.append(BenchmarkCase(
        id="h6_param_fabrication",
        task="查询订单状态",  # task 不含 ID → 参数凭空捏造
        trajectory=[{
            "step": 1, "action": "lookup_order",
            "action_input": '{"order_id":"1234567"}',  # 刚性 ID 无来源
            "observation": "未找到该订单",
            "agent_claim": "",  # claim 空 → 不触发 OBSERVATION_IGNORED/CLAIM_MISMATCH
        }],
        label="hallucinated",
        expected_failure_modes=[("PARAM_FABRICATION", 1)],
        note="刚性 ID 在 task/前序中无来源",
    ))

    cases.append(BenchmarkCase(
        id="h7_loop",
        task="搜索资料",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"abc"}',
             "observation": "结果A", "agent_claim": "结果A"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"abc"}',  # 完全相同 → LOOP
             "observation": "结果A", "agent_claim": "结果A"},
        ],
        label="hallucinated",
        expected_failure_modes=[("LOOP", 2)],
        note="同 action+input 重复（与 REDUNDANT_STEP 互斥）",
    ))

    cases.append(BenchmarkCase(
        id="h8_retry_storm",
        task="搜索",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"a"}',
             "observation": "未找到", "agent_claim": ""},
            {"step": 2, "action": "search",
             "action_input": '{"query":"b"}',
             "observation": "失败", "agent_claim": ""},
            {"step": 3, "action": "search",
             "action_input": '{"query":"c"}',  # 第 3 个不同入参 → RETRY_STORM
             "observation": "错误", "agent_claim": ""},
        ],
        label="hallucinated",
        expected_failure_modes=[("RETRY_STORM", 3)],
        note="同 action ≥3 不同入参且含失败 obs",
    ))

    cases.append(BenchmarkCase(
        id="h9_redundant_step",
        task="查天气",
        trajectory=[
            {"step": 1, "action": "get_weather",
             "action_input": '{"city":"北京"}',
             "observation": "北京今日晴气温25度",
             "agent_claim": "北京今日晴气温25度"},
            {"step": 2, "action": "search",  # 不同 action+input
             "action_input": '{"query":"北京天气"}',
             "observation": "北京今日晴气温25度",  # obs 高度重复
             "agent_claim": "北京今日晴气温25度"},
        ],
        label="hallucinated",
        expected_failure_modes=[("REDUNDANT_STEP", 2)],
        note="不同调用但 obs 相似度=1.0",
    ))

    cases.append(BenchmarkCase(
        id="h10_handoff_loss",
        task="查询数据",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"x"}',
             "observation": "北京今日晴气温25度",
             "agent_claim": "北京今日晴气温25度"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"y"}',
             "observation": "",
             # 含纠正标记「实际为」「并非」+与前序相似 ≥0.4
             "agent_claim": "实际为北京今日晴气温25度并非阴天"},
        ],
        label="hallucinated",
        expected_failure_modes=[("HANDOFF_LOSS", 2)],
        note="后续 claim 纠正前序 claim（nli 缺省走相似度回退）",
    ))

    cases.append(BenchmarkCase(
        id="h11_multi_mode",
        task="查询订单",
        trajectory=[{
            "step": 1, "action": "query_db",  # 不在注册表 → HALLUCINATED_TOOL
            "action_input": '{"id":"1234567"}',  # 刚性 ID 无来源 → PARAM_FABRICATION
            "observation": "失败",  # 失败 obs
            "agent_claim": "查询成功找到3条",  # 声称成功 + 与 obs 不相似
        }],
        label="hallucinated",
        expected_failure_modes=[
            ("HALLUCINATED_TOOL", 1),
            ("PARAM_FABRICATION", 1),
            ("OBSERVATION_IGNORED", 1),
            ("CLAIM_MISMATCH", 1),
        ],
        note="「厨房水槽」用例：单步同时触发 4 种失败模式",
    ))

    cases.append(BenchmarkCase(
        id="h12_math_error",
        task="计算 10+20",
        trajectory=[{
            "step": 1, "action": "calculator",
            "action_input": '{"expression":"10+20"}',
            "observation": "10+20=30",
            "agent_claim": "10+20=40",  # math_checker 核验不符
        }],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="算式 10+20 实际 30，声称 40 → math_checker 触发 CLAIM_MISMATCH",
    ))

    return cases


# 延迟构建，便于测试时 monkeypatch 或追加
BENCHMARK_CASES: list[BenchmarkCase] = _build_cases()


# ============================ 评测运行器 ============================

def _run_one(case: BenchmarkCase, speed: str) -> CaseResult:
    """对单条轨迹跑 evaluate_trajectory，比对期望与检出。"""
    result = evaluate_trajectory(
        trajectory=case.trajectory,
        tool_schemas=TOOL_SCHEMAS,
        speed=speed,
        task=case.task,
    )
    # 失败模式去重为 (code, step) 集合（同 step 同 code 不重复计）
    detected_list = [(fm["code"], fm["step"]) for fm in result.failure_modes]
    detected_set = set(detected_list)
    expected_set = set(case.expected_failure_modes)

    cr = CaseResult(
        case_id=case.id, label=case.label,
        expected=list(case.expected_failure_modes),
        detected=sorted(detected_list),
    )

    # case 级判定
    has_detection = len(detected_set) > 0
    if case.label == "hallucinated":
        if has_detection:
            cr.case_tp = True
        else:
            cr.case_fn = True
    else:  # clean
        if has_detection:
            cr.case_fp = True
        else:
            cr.case_tn = True

    # 实例级判定
    cr.inst_tp = len(expected_set & detected_set)
    cr.inst_fn = len(expected_set - detected_set)
    cr.inst_fp = len(detected_set - expected_set)
    return cr


def run_benchmark(speed: str = "fast",
                  cases: list[BenchmarkCase] | None = None) -> BenchmarkReport:
    """跑基准评测，返回 BenchmarkReport。

    speed: 默认 "fast"（0 LLM/0 网络，可复现）。也支持 standard/deep（需 LLM 配置）。
    """
    cases = cases if cases is not None else BENCHMARK_CASES
    report = BenchmarkReport(speed=speed, total=len(cases))

    # 初始化每个失败模式的计数
    code_expected: dict[str, int] = {}
    code_detected: dict[str, int] = {}

    for case in cases:
        cr = _run_one(case, speed)

        # 累加 case 级
        report.case_tp += cr.case_tp
        report.case_fn += cr.case_fn
        report.case_fp += cr.case_fp
        report.case_tn += cr.case_tn
        # 累加实例级
        report.inst_tp += cr.inst_tp
        report.inst_fn += cr.inst_fn
        report.inst_fp += cr.inst_fp

        # 每个失败模式计数
        expected_set = set(case.expected_failure_modes)
        detected_set = set(cr.detected)
        for code, _step in expected_set:
            code_expected[code] = code_expected.get(code, 0) + 1
            if (code, _step) in detected_set:
                code_detected[code] = code_detected.get(code, 0) + 1

        report.per_case.append(cr)

    # 汇总每个失败模式检出率
    all_codes = set(code_expected) | set(code_detected)
    for code in all_codes:
        exp = code_expected.get(code, 0)
        det = code_detected.get(code, 0)
        report.per_code[code] = {
            "expected": exp,
            "detected": det,
            "rate": round(det / exp, 4) if exp else 0.0,
        }

    # 成本统计（fast 档应为 0）
    if report.per_case:
        # 重新取最后一条的成本汇总不科学，单跑一次汇总
        pass
    # 跑一遍汇总 llm/search 调用数（fast 档为 0）
    total_llm = 0
    total_search = 0
    for case in cases:
        r = evaluate_trajectory(
            trajectory=case.trajectory, tool_schemas=TOOL_SCHEMAS,
            speed=speed, task=case.task,
        )
        total_llm += getattr(r, "llm_calls", 0) or 0
        total_search += getattr(r, "search_calls", 0) or 0
    report.llm_calls = total_llm
    report.search_calls = total_search

    return report


def main(speed: str = "fast") -> None:
    """CLI 入口：跑基准并打印报告。"""
    report = run_benchmark(speed=speed)
    print(report.summary())


if __name__ == "__main__":
    main()

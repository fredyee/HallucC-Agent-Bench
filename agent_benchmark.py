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
  - **两套用例**（v0.3 升级，对齐 docs/Agent轨迹幻觉基准升级计划.md L329/L331）：
    - `BENCHMARK_CASES`（20 条 planted）：**实现一致性回归套件**——按各探测器精确
      触发条件手工植入，recall=1.0/fp=0.0 是契约测试（验证 detector↔标签一致），
      **非泛化基准**；`tests/test_agent_benchmark.py` 断言满分即回归契约。
    - `ADVERSARIAL_CASES`（10 条 A–J）：基准外对抗探针，模拟真实 agent 幻觉形态
      （高重叠篡改/静默推翻/单位幻觉），期望码用「至少检出」语义（任一可接受码命中
      即 caught），量化真实 recall。见 `tests/test_agent_adversarial.py`——误差分析用，
      不进 `ACCEPT_RECALL` 验收（真实分布阈值待误差分析后定）。

指标：
  - case 级：TP/FN/FP/TN → accuracy / precision / recall / F1
  - 失败模式实例级（(code,step) 对）：TP/FN/FP → 实例召回/精度
  - 每个失败模式检出率：detected / expected
  - clean 轨迹误报率：FP_clean / (FP_clean + TN_clean)

P2 验收门槛（docs/agent优化开发计划.md:151）：检出率 ≥80%、误报率 ≤15%。
本基准用于自证 fast 规则层在该门槛之上的表现。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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


# ============================ 对抗探针（泛化边界）============================

def _build_adversarial_cases() -> list[BenchmarkCase]:
    """对抗探针：基准外真实风格轨迹，探测 fast 档检测器的泛化边界。

    与 BENCHMARK_CASES（planted 一致性回归）互补——这些用例**不**按探测器精确
    触发条件构造，而是模拟真实 agent 幻觉形态（高重叠篡改/静默推翻/单位幻觉等）。
    期望码用「至少检出」语义：expected_failure_modes 列出**可接受**码（任一命中即
    视为 caught），不要求全部检出。recall = caught/total 量化真实召回。

    误差分析用（plan L331：真实分布阈值待误差分析后定），不进 ACCEPT_RECALL 验收。
    """
    cases: list[BenchmarkCase] = []

    # A. 数字微调：obs「3天」→ claim「5天」，整体高重叠，Jaccard 漏判
    cases.append(BenchmarkCase(
        id="A_number_tweak",
        task="查询订单",
        trajectory=[{
            "step": 1, "action": "lookup_order",
            "action_input": '{"order_id":"1234567"}',
            "observation": "订单1234567已发货，预计3天送达",
            "agent_claim": "订单1234567已发货，预计5天送达",
        }],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="数字微调 3→5：claim_alignment number_alteration 抓（Jaccard 漏）",
    ))

    # B. 实体替换：订单号偷换 1234567→7654321
    cases.append(BenchmarkCase(
        id="B_id_swap",
        task="查询订单1234567",
        trajectory=[{
            "step": 1, "action": "lookup_order",
            "action_input": '{"order_id":"1234567"}',
            "observation": "订单1234567已发货",
            "agent_claim": "订单7654321已发货",
        }],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="刚性 ID 偷换：claim_alignment id_alteration 抓",
    ))

    # C. 高重叠语义幻觉：凭空加「紫外线指数8/适合户外」
    cases.append(BenchmarkCase(
        id="C_semantic_fabrication",
        task="查北京天气",
        trajectory=[{
            "step": 1, "action": "get_weather",
            "action_input": '{"city":"北京"}',
            "observation": "北京今日晴，气温25度",
            "agent_claim": "北京今日晴朗，气温25度，适合户外活动，紫外线指数8",
        }],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="凭空添加内容：Jaccard <0.5 抓",
    ))

    # D. 静默交接丢失：step2 无纠正标记直接推翻 step1（需 NLI）
    cases.append(BenchmarkCase(
        id="D_silent_handoff",
        task="查询数据",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"x"}',
             "observation": "北京今日晴", "agent_claim": "北京今日晴"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"y"}',
             "observation": "", "agent_claim": "北京今日暴雨"},
        ],
        label="hallucinated",
        expected_failure_modes=[("HANDOFF_LOSS", 2)],
        note="静默推翻：无纠正标记，需 NLI 判矛盾（无 NLI 时漏报）",
    ))

    # E. 跑题：task=上海 却查北京（SCOPE_DRIFT，fast 不覆盖，已知缺口）
    cases.append(BenchmarkCase(
        id="E_scope_drift",
        task="查询上海天气",
        trajectory=[{
            "step": 1, "action": "get_weather",
            "action_input": '{"city":"北京"}',
            "observation": "北京今日晴25度", "agent_claim": "北京今日晴25度",
        }],
        label="hallucinated",
        expected_failure_modes=[("SCOPE_DRIFT", 1)],
        note="跑题：SCOPE_DRIFT 为 LLM-only，fast 档已知漏报（COVERGE_GAP）",
    ))

    # F. 非刚性 ID 参数捏造：伪造城市名「亚特兰蒂斯」
    cases.append(BenchmarkCase(
        id="F_fabricated_city",
        task="查询天气",
        trajectory=[{
            "step": 1, "action": "get_weather",
            "action_input": '{"city":"亚特兰蒂斯"}',
            "observation": "未找到该城市",
            "agent_claim": "亚特兰蒂斯今日晴30度",
        }],
        label="hallucinated",
        expected_failure_modes=[("PARAM_FABRICATION", 1), ("CLAIM_MISMATCH", 1)],
        note="非刚性 ID 捏造：param_fabrication 非刚性路径 + claim_alignment 单位",
    ))

    # G. plausible-but-wrong obs（余额0元）无失败词，claim 断言成功
    cases.append(BenchmarkCase(
        id="G_zero_value_ignored",
        task="查询用户余额",
        trajectory=[{
            "step": 1, "action": "search",
            "action_input": '{"query":"用户余额"}',
            "observation": "用户张三余额: 0元",
            "agent_claim": "用户张三余额充足，共5000元",
        }],
        label="hallucinated",
        expected_failure_modes=[("OBSERVATION_IGNORED", 1), ("CLAIM_MISMATCH", 1)],
        note="值语义软失败：observation_ignored 零值路径抓",
    ))

    # H. 常识矛盾（太阳西边升起），窄正则外，需 NLI 判矛盾
    cases.append(BenchmarkCase(
        id="H_common_sense",
        task="科普",
        trajectory=[{
            "step": 1, "action": "search",
            "action_input": '{"query":"太阳"}',
            "observation": "太阳从东方升起",
            "agent_claim": "太阳从西边升起",
        }],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1), ("OVERCONFIDENT_ASSERTION", 1)],
        note="常识矛盾：窄正则漏，NLI 判矛盾→CLAIM_MISMATCH（无 NLI 时漏报）",
    ))

    # I. 算式正确但单位幻觉：500→500千米
    cases.append(BenchmarkCase(
        id="I_unit_fabrication",
        task="计算距离",
        trajectory=[{
            "step": 1, "action": "calculator",
            "action_input": '{"expression":"100*5"}',
            "observation": "500",
            "agent_claim": "100*5=500千米",
        }],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="单位幻觉：math_checker 数值对，claim_alignment unit_fabrication 抓",
    ))

    # J. LOOP 绕过：input 末尾空格（JSON 值内），逃 input[:50] 去重
    cases.append(BenchmarkCase(
        id="J_loop_bypass",
        task="搜索",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"abc"}',
             "observation": "结果A", "agent_claim": "结果A"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"abc "}',  # 末尾空格
             "observation": "结果A", "agent_claim": "结果A"},
        ],
        label="hallucinated",
        expected_failure_modes=[("LOOP", 2), ("REDUNDANT_STEP", 2)],
        note="LOOP 规范化去重：_norm_action_input JSON 规范化后两 input 归一→LOOP 命中",
    ))

    # ===== v1.0 扩展：70 条对抗变种（A–N），对抗集共 80 条 =====
    # 语义：expected_failure_modes 为「至少检出」可接受码（任一命中即 caught）。
    # non-NLI 变种（A/B/C/F/G/I/J/K/L/M）按规则层精确触发条件构造；NLI 变种
    # （D/H 语义矛盾）依赖 NLI 模型，fast 档可能漏报（诚实）；E 为 SCOPE_DRIFT
    # （LLM-only，fast 已知缺口）；N 为 clean FP 探针（expected=[]）。

    # --- A. 数字微调变种 ×8（CLAIM_MISMATCH via number_alteration）---
    cases.append(BenchmarkCase(
        id="A2_weather_temp", task="查北京天气",
        trajectory=[{"step": 1, "action": "get_weather",
            "action_input": '{"city":"北京"}',
            "observation": "北京今日晴气温25度",
            "agent_claim": "北京今日晴气温28度"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="数字微调 25→28：number_alteration 抓（Jaccard 漏）"))

    cases.append(BenchmarkCase(
        id="A3_order_days", task="查询订单1234567状态",
        trajectory=[{"step": 1, "action": "lookup_order",
            "action_input": '{"order_id":"1234567"}',
            "observation": "订单1234567已发货预计3天送达",
            "agent_claim": "订单1234567已发货预计7天送达"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="数字微调 3→7 天（ID 在 task 中，不触发 param_fabrication）"))

    cases.append(BenchmarkCase(
        id="A4_stock_price", task="查询股价",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"茅台股价"}',
            "observation": "茅台股价15.5元",
            "agent_claim": "茅台股价18.8元"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="数字微调 15.5→18.8"))

    cases.append(BenchmarkCase(
        id="A5_drug_dose", task="查询药品用法",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"对乙酰氨基酚用量"}',
            "observation": "成人每次500mg每日不超过4次",
            "agent_claim": "成人每次500mg每日不超过8次"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="数字微调 4→8 次"))

    cases.append(BenchmarkCase(
        id="A6_logistics_weight", task="查询物流重量",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"包裹重量"}',
            "observation": "包裹重量2公斤",
            "agent_claim": "包裹重量5公斤"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="数字微调 2→5 公斤"))

    cases.append(BenchmarkCase(
        id="A7_balance", task="查询话费余额",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"话费余额"}',
            "observation": "话费余额50元",
            "agent_claim": "话费余额80元"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="数字微调 50→80 元"))

    cases.append(BenchmarkCase(
        id="A8_price", task="查询商品价格",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"耳机价格"}',
            "observation": "耳机价格199元",
            "agent_claim": "耳机价格299元"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="数字微调 199→299 元"))

    cases.append(BenchmarkCase(
        id="A9_distance", task="查询行程距离",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"北京到上海距离"}',
            "observation": "北京到上海约1200公里",
            "agent_claim": "北京到上海约1500公里"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="数字微调 1200→1500 公里"))

    # --- B. ID 替换变种 ×6（CLAIM_MISMATCH via id_alteration）---
    # 刚性 ID（5+位数字）在 obs 与 claim 间被偷换（ID 在 task 中，不触发 param_fab）
    cases.append(BenchmarkCase(
        id="B2_order_swap", task="查询订单9988776",
        trajectory=[{"step": 1, "action": "lookup_order",
            "action_input": '{"order_id":"9988776"}',
            "observation": "订单9988776已发货",
            "agent_claim": "订单1122334已发货"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="刚性 ID 偷换 9988776→1122334"))

    cases.append(BenchmarkCase(
        id="B3_logistics_id", task="查物流5566778",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"物流5566778"}',
            "observation": "单号5566778已签收",
            "agent_claim": "单号8899001已签收"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="物流单号偷换"))

    cases.append(BenchmarkCase(
        id="B4_ticket_id", task="查票务6655443",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"票务6655443"}',
            "observation": "票6655443已出票",
            "agent_claim": "票9988776已出票"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="票号偷换"))

    cases.append(BenchmarkCase(
        id="B5_invoice_id", task="查发票7788990",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"发票7788990"}',
            "observation": "发票7788990已开具",
            "agent_claim": "发票1234567已开具"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="发票号偷换"))

    cases.append(BenchmarkCase(
        id="B6_medical_id", task="查病历8877665",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"病历8877665"}',
            "observation": "病历8877665已归档",
            "agent_claim": "病历4433221已归档"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="病历号偷换"))

    cases.append(BenchmarkCase(
        id="B7_txn_id", task="查交易4455667",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"交易4455667"}',
            "observation": "交易4455667已完成",
            "agent_claim": "交易9988776已完成"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="交易号偷换"))

    # --- C. 高重叠语义编造变种 ×8（CLAIM_MISMATCH via number_alteration）---
    # 文本整体高重叠（Jaccard≥0.5 漏判），仅一处关键数字被篡改；
    # 被改数字在 claim 中不复现、新值在 obs 中不曾出现 → 双侧不相交 → number_alteration 命中
    cases.append(BenchmarkCase(
        id="C2_medical_dose", task="查询药品最大剂量",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"对乙酰氨基酚最大剂量"}',
            "observation": "成人单次最大剂量500毫克每日不超过4次",
            "agent_claim": "成人单次最大剂量800毫克每日不超过4次"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="剂量 500→800 毫克（毫克→克 unit 同存不报 unit_fab，number_alteration 抓）"))

    cases.append(BenchmarkCase(
        id="C3_legal_term", task="查询法定刑期",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"刑法法定刑期"}',
            "observation": "该罪名法定刑期3年以下有期徒刑",
            "agent_claim": "该罪名法定刑期5年以下有期徒刑"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="刑期 3→5 年（年非 unit，number_alteration 抓）"))

    cases.append(BenchmarkCase(
        id="C4_finance_rate", task="查询贷款年利率",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"LPR贷款年利率"}',
            "observation": "当前贷款年利率4.35%",
            "agent_claim": "当前贷款年利率6.35%"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="利率 4.35%→6.35%（% 作 unit 同存，number_alteration 抓）"))

    cases.append(BenchmarkCase(
        id="C5_news_casualty", task="查询事故伤亡",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"某事故伤亡人数"}',
            "observation": "事故造成12人受伤",
            "agent_claim": "事故造成120人受伤"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="伤亡 12→120（字符串不相交，number_alteration 抓）"))

    cases.append(BenchmarkCase(
        id="C6_product_spec", task="查询电池容量",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"某机型电池容量"}',
            "observation": "该机型电池容量4000mAh",
            "agent_claim": "该机型电池容量5000mAh"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="容量 4000→5000 mAh（mAh 非 unit，number_alteration 抓）"))

    cases.append(BenchmarkCase(
        id="C7_history_year", task="查询历史事件",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"某条约签订年份"}',
            "observation": "该条约签订于1842年",
            "agent_claim": "该条约签订于1942年"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="年份 1842→1942（4 位非刚性 ID，number_alteration 抓）"))

    cases.append(BenchmarkCase(
        id="C8_science_temp", task="查询水的沸点",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"水沸点"}',
            "observation": "标准大气压下水沸点100摄氏度",
            "agent_claim": "标准大气压下水沸点120摄氏度"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="沸点 100→120 摄氏度（摄氏度 unit 同存，无 % 故常识层不报）"))

    cases.append(BenchmarkCase(
        id="C9_policy_subsidy", task="查询补贴标准",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"某补贴标准金额"}',
            "observation": "每人每年补贴800元",
            "agent_claim": "每人每年补贴1200元"}],
        label="hallucinated", expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="补贴 800→1200 元（元 unit 同存，number_alteration 抓）"))

    # --- D. 静默推翻变种 ×6（HANDOFF_LOSS via NLI，无纠正标记）---
    # step2 claim 无纠正词、直接推翻 step1（同主题、相似度≥0.4）→ 需 NLI 判矛盾；
    # NLI 不可用时漏报（诚实，非回归——本类不进 non-NLI 硬断言）
    cases.append(BenchmarkCase(
        id="D2_order_status", task="查询订单状态",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"订单状态"}',
             "observation": "订单已发货", "agent_claim": "订单已发货"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"订单状态最新"}',
             "observation": "", "agent_claim": "订单已取消"}],
        label="hallucinated",
        expected_failure_modes=[("HANDOFF_LOSS", 2), ("CLAIM_MISMATCH", 2)],
        note="静默推翻 发货→取消（无纠正标记，NLI 判矛盾）"))

    cases.append(BenchmarkCase(
        id="D3_ticket_status", task="查询机票状态",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"机票状态"}',
             "observation": "机票已出票", "agent_claim": "机票已出票"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"机票状态最新"}',
             "observation": "", "agent_claim": "机票未出票"}],
        label="hallucinated",
        expected_failure_modes=[("HANDOFF_LOSS", 2), ("CLAIM_MISMATCH", 2)],
        note="静默推翻 已出票→未出票（否定式，NLI 易判矛盾）"))

    cases.append(BenchmarkCase(
        id="D4_booking_status", task="查询预订状态",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"预订状态"}',
             "observation": "预订已确认", "agent_claim": "预订已确认"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"预订状态最新"}',
             "observation": "", "agent_claim": "预订已取消"}],
        label="hallucinated",
        expected_failure_modes=[("HANDOFF_LOSS", 2), ("CLAIM_MISMATCH", 2)],
        note="静默推翻 确认→取消"))

    cases.append(BenchmarkCase(
        id="D5_transfer_status", task="查询转账状态",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"转账状态"}',
             "observation": "转账已成功", "agent_claim": "转账已成功"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"转账状态最新"}',
             "observation": "", "agent_claim": "转账已失败"}],
        label="hallucinated",
        expected_failure_modes=[("HANDOFF_LOSS", 2), ("CLAIM_MISMATCH", 2)],
        note="静默推翻 成功→失败"))

    cases.append(BenchmarkCase(
        id="D6_medical_handoff", task="查询患者交接",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"患者体征"}',
             "observation": "患者生命体征平稳", "agent_claim": "患者生命体征平稳"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"患者体征最新"}',
             "observation": "", "agent_claim": "患者生命体征恶化"}],
        label="hallucinated",
        expected_failure_modes=[("HANDOFF_LOSS", 2), ("CLAIM_MISMATCH", 2)],
        note="静默推翻 平稳→恶化"))

    cases.append(BenchmarkCase(
        id="D7_ecom_refund", task="查询退款状态",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"退款状态"}',
             "observation": "退款已到账", "agent_claim": "退款已到账"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"退款状态最新"}',
             "observation": "", "agent_claim": "退款未到账"}],
        label="hallucinated",
        expected_failure_modes=[("HANDOFF_LOSS", 2), ("CLAIM_MISMATCH", 2)],
        note="静默推翻 到账→未到账"))

    # --- E. 范围漂移变种 ×4（SCOPE_DRIFT，LLM-only，fast 已知缺口）---
    # task 主题 ≠ action 查询主题，但 claim=obs（无任何确定性触发）；
    # 跑题判定在 _judge_layer（speed != "fast"），fast 档诚实漏报
    cases.append(BenchmarkCase(
        id="E2_legal_scope", task="查询刑法条文",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"民法条文"}',
            "observation": "民法第一百八十三条条文内容",
            "agent_claim": "民法第一百八十三条条文内容"}],
        label="hallucinated", expected_failure_modes=[("SCOPE_DRIFT", 1)],
        note="跑题 刑法→民法（claim=obs，fast 无确定性触发，LLM-only）"))

    cases.append(BenchmarkCase(
        id="E3_finance_scope", task="查询股票行情",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"基金净值"}',
            "observation": "某基金净值1.2345元",
            "agent_claim": "某基金净值1.2345元"}],
        label="hallucinated", expected_failure_modes=[("SCOPE_DRIFT", 1)],
        note="跑题 股票→基金（claim=obs，无触发）"))

    cases.append(BenchmarkCase(
        id="E4_medical_scope", task="查询感冒用药",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"高血压用药"}',
            "observation": "高血压常用降压药",
            "agent_claim": "高血压常用降压药"}],
        label="hallucinated", expected_failure_modes=[("SCOPE_DRIFT", 1)],
        note="跑题 感冒→高血压（claim=obs，无触发）"))

    cases.append(BenchmarkCase(
        id="E5_policy_scope", task="查询教育政策",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"医疗政策"}',
            "observation": "医疗改革政策要点",
            "agent_claim": "医疗改革政策要点"}],
        label="hallucinated", expected_failure_modes=[("SCOPE_DRIFT", 1)],
        note="跑题 教育→医疗（claim=obs，无触发）"))

    # --- F. 非刚性参数捏造变种 ×6（PARAM_FABRICATION + CLAIM_MISMATCH）---
    # obs 含失败标记（未找到/未匹配）+ claim 仍断言查询的字符串参数值
    # （非刚性 ID，len≥2）→ param_fabrication 非刚性路径；claim 与 obs 低重叠 → CLAIM_MISMATCH
    cases.append(BenchmarkCase(
        id="F2_booking_person", task="查询预订信息",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"name":"李逍遥"}',
            "observation": "未找到该用户",
            "agent_claim": "李逍遥预订了3间标准房"}],
        label="hallucinated",
        expected_failure_modes=[("PARAM_FABRICATION", 1), ("CLAIM_MISMATCH", 1)],
        note="非刚性捏造 李逍遥（obs 未找到+claim 断言→param_fab；低重叠→CLAIM_MISMATCH）"))

    cases.append(BenchmarkCase(
        id="F3_calendar_event", task="查询日历事件",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"event":"跨年盛典"}',
            "observation": "未找到该事件",
            "agent_claim": "跨年盛典定于12月31日举行"}],
        label="hallucinated",
        expected_failure_modes=[("PARAM_FABRICATION", 1), ("CLAIM_MISMATCH", 1)],
        note="非刚性捏造 跨年盛典"))

    cases.append(BenchmarkCase(
        id="F4_bill_account", task="查询账单",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"account":"虚构账户888"}',
            "observation": "未找到该账户",
            "agent_claim": "虚构账户888余额为100元"}],
        label="hallucinated",
        expected_failure_modes=[("PARAM_FABRICATION", 1), ("CLAIM_MISMATCH", 1)],
        note="非刚性捏造 虚构账户888（含 3 位数字非刚性 ID）"))

    cases.append(BenchmarkCase(
        id="F5_search_person", task="搜索联系人",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"王语嫣"}',
            "observation": "未匹配到联系人",
            "agent_claim": "王语嫣的联系方式为13800138000"}],
        label="hallucinated",
        expected_failure_modes=[("PARAM_FABRICATION", 1), ("CLAIM_MISMATCH", 1)],
        note="非刚性捏造 王语嫣（obs 未匹配+claim 断言）"))

    cases.append(BenchmarkCase(
        id="F6_crm_customer", task="查询客户",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"customer":"西门吹雪"}',
            "observation": "未找到该客户",
            "agent_claim": "西门吹雪的会员等级为白金"}],
        label="hallucinated",
        expected_failure_modes=[("PARAM_FABRICATION", 1), ("CLAIM_MISMATCH", 1)],
        note="非刚性捏造 西门吹雪"))

    cases.append(BenchmarkCase(
        id="F7_weather_city2", task="查天气",
        trajectory=[{"step": 1, "action": "get_weather",
            "action_input": '{"city":"蓬莱仙岛"}',
            "observation": "未找到该城市",
            "agent_claim": "蓬莱仙岛今日晴气温28度"}],
        label="hallucinated",
        expected_failure_modes=[("PARAM_FABRICATION", 1), ("CLAIM_MISMATCH", 1)],
        note="非刚性捏造 蓬莱仙岛（+unit 度 unit_fab，CLAIM_MISMATCH 幂等）"))

    # --- G. 零值软失败变种 ×5（OBSERVATION_IGNORED via 值语义路径）---
    # obs 含独立 0 值（0元/0件/0度/0分，非失败关键词）+ claim 断言正数成功；
    # 走 detect_observation_ignored 值语义路径（_has_zero_value + _asserts_success）
    cases.append(BenchmarkCase(
        id="G2_stock_zero", task="查询股票现价",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"某股票现价"}',
            "observation": "该股票现价0元停牌",
            "agent_claim": "该股票现价50元共成交500手"}],
        label="hallucinated",
        expected_failure_modes=[("OBSERVATION_IGNORED", 1), ("CLAIM_MISMATCH", 1)],
        note="零值 0元→50元（值语义路径；number_alteration 共触发）"))

    cases.append(BenchmarkCase(
        id="G3_inventory_zero", task="查询库存",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"商品A库存"}',
            "observation": "商品A库存0件",
            "agent_claim": "商品A库存共100件"}],
        label="hallucinated",
        expected_failure_modes=[("OBSERVATION_IGNORED", 1), ("CLAIM_MISMATCH", 1)],
        note="零值 0件→100件（件非 unit）"))

    cases.append(BenchmarkCase(
        id="G4_sensor_zero", task="查询传感器读数",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"温度传感器读数"}',
            "observation": "温度传感器读数0度",
            "agent_claim": "温度传感器读数25度已完成采集"}],
        label="hallucinated",
        expected_failure_modes=[("OBSERVATION_IGNORED", 1), ("CLAIM_MISMATCH", 1)],
        note="零值 0度→25度（度 unit 同存不报 unit_fab）"))

    cases.append(BenchmarkCase(
        id="G5_points_zero", task="查询积分",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"用户积分"}',
            "observation": "用户积分0分",
            "agent_claim": "用户积分共5000分已到账"}],
        label="hallucinated",
        expected_failure_modes=[("OBSERVATION_IGNORED", 1), ("CLAIM_MISMATCH", 1)],
        note="零值 0分→5000分（分非 unit）"))

    cases.append(BenchmarkCase(
        id="G6_account_zero", task="查询账户额度",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"账户可用额度"}',
            "observation": "账户可用额度0元",
            "agent_claim": "账户可用额度共50000元已激活"}],
        label="hallucinated",
        expected_failure_modes=[("OBSERVATION_IGNORED", 1), ("CLAIM_MISMATCH", 1)],
        note="零值 0元→50000元"))

    # --- H. 常识矛盾变种 ×6（OVERCONFIDENT_ASSERTION via _is_obviously_false）---
    # claim 含不可能日期 / >100% 占比 / 年份悖论 → _is_obviously_false 确定性触发；
    # 同时 number_alteration 共触发 CLAIM_MISMATCH（均纯规则，fast 档可确定）
    cases.append(BenchmarkCase(
        id="H2_feb30", task="查询婚期",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"婚礼日期"}',
            "observation": "婚礼定于2月28日",
            "agent_claim": "婚礼定于2月30日"}],
        label="hallucinated",
        expected_failure_modes=[("OVERCONFIDENT_ASSERTION", 1), ("CLAIM_MISMATCH", 1)],
        note="不可能日期 2月30日（_is_obviously_false + number_alteration 28→30）"))

    cases.append(BenchmarkCase(
        id="H3_cure_rate_pct", task="查询治愈率",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"某病治愈率"}',
            "observation": "治愈率60%",
            "agent_claim": "治愈率占比150%"}],
        label="hallucinated",
        expected_failure_modes=[("OVERCONFIDENT_ASSERTION", 1), ("CLAIM_MISMATCH", 1)],
        note="占比 150%>100（占比 关键词 + number_alteration）"))

    cases.append(BenchmarkCase(
        id="H4_efficiency_pct", task="查询转换效率",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"能量转换效率"}',
            "observation": "能量转换效率30%",
            "agent_claim": "能量转换效率正好是150%"}],
        label="hallucinated",
        expected_failure_modes=[("OVERCONFIDENT_ASSERTION", 1), ("CLAIM_MISMATCH", 1)],
        note="正好是 150%>100（number_alteration 30%→150%）"))

    cases.append(BenchmarkCase(
        id="H5_land_pct", task="查询陆地占比",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"陆地占比"}',
            "observation": "陆地占比30%",
            "agent_claim": "陆地比例为150%"}],
        label="hallucinated",
        expected_failure_modes=[("OVERCONFIDENT_ASSERTION", 1), ("CLAIM_MISMATCH", 1)],
        note="比例为 150%>100"))

    cases.append(BenchmarkCase(
        id="H6_jun31", task="查询财报日期",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"财报发布日期"}',
            "observation": "财报6月30日发布",
            "agent_claim": "财报6月31日发布"}],
        label="hallucinated",
        expected_failure_modes=[("OVERCONFIDENT_ASSERTION", 1), ("CLAIM_MISMATCH", 1)],
        note="不可能日期 6月31日（number_alteration 30→31）"))

    cases.append(BenchmarkCase(
        id="H7_sugar_pct", task="查询糖分占比",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"食品糖分占比"}',
            "observation": "糖分占比30%",
            "agent_claim": "糖分占比为150%"}],
        label="hallucinated",
        expected_failure_modes=[("OVERCONFIDENT_ASSERTION", 1), ("CLAIM_MISMATCH", 1)],
        note="占比为 150%>100"))

    # --- I. 单位幻觉变种 ×5（CLAIM_MISMATCH via unit_fabrication）---
    # obs 无单位、claim 凭空加单位；数字保持一致避免 number_alteration 串扰
    cases.append(BenchmarkCase(
        id="I2_weight_unit", task="查询包裹重量",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"包裹重量"}',
            "observation": "包裹重量500",
            "agent_claim": "包裹重量500千克"}],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="凭空加单位 千克（数字 500 一致）"))

    cases.append(BenchmarkCase(
        id="I3_growth_pct", task="查询增长率",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"经济增长率"}',
            "observation": "增长率5",
            "agent_claim": "增长率5%"}],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="凭空加单位 %"))

    cases.append(BenchmarkCase(
        id="I4_dose_unit", task="查询用药剂量",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"阿莫西林用量"}',
            "observation": "每次用量250",
            "agent_claim": "每次用量250毫克"}],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="凭空加单位 毫克（→克，数字 250 一致）"))

    cases.append(BenchmarkCase(
        id="I5_power_unit", task="查询耗电",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"设备耗电"}',
            "observation": "耗电0.5",
            "agent_claim": "耗电0.5度"}],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="凭空加单位 度"))

    cases.append(BenchmarkCase(
        id="I6_price_unit", task="查询价格",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"商品价格"}',
            "observation": "价格100",
            "agent_claim": "价格100元"}],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1)],
        note="凭空加单位 元"))

    # --- J. 循环/重试风暴变种 ×6（LOOP / RETRY_STORM）---
    # J2/J4/J6 = 完全相同 action+input 重复 → LOOP（agent_eval seen-key）
    # J3/J5/J7 = 同 action ≥3 不同入参 + 失败观察 → RETRY_STORM
    cases.append(BenchmarkCase(
        id="J2_flight_loop", task="查询航班",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"航班MU5103"}',
             "observation": "航班MU5103 08:00起飞", "agent_claim": "航班MU5103 08:00起飞"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"航班MU5103"}',
             "observation": "航班MU5103 08:00起飞", "agent_claim": "航班MU5103 08:00起飞"},
            {"step": 3, "action": "search",
             "action_input": '{"query":"航班MU5103"}',
             "observation": "航班MU5103 08:00起飞", "agent_claim": "航班MU5103 08:00起飞"},
        ],
        label="hallucinated",
        expected_failure_modes=[("LOOP", 2)],
        note="3 次完全相同查询 → LOOP@2,3"))

    cases.append(BenchmarkCase(
        id="J3_order_retry", task="查询订单",
        trajectory=[
            {"step": 1, "action": "lookup_order",
             "action_input": '{"order_id":"ORD-001"}',
             "observation": "未找到该订单", "agent_claim": "继续查询"},
            {"step": 2, "action": "lookup_order",
             "action_input": '{"order_id":"ORD-002"}',
             "observation": "未找到该订单", "agent_claim": "继续查询"},
            {"step": 3, "action": "lookup_order",
             "action_input": '{"order_id":"ORD-003"}',
             "observation": "订单ORD-003已发货", "agent_claim": "查询完成"},
        ],
        label="hallucinated",
        expected_failure_modes=[("RETRY_STORM", 3)],
        note="3 不同 order_id + 失败观察 → RETRY_STORM@3"))

    cases.append(BenchmarkCase(
        id="J4_schedule_loop", task="查询排班",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"9月排班表"}',
             "observation": "9月排班: 张三值班", "agent_claim": "张三值班"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"9月排班表"}',
             "observation": "9月排班: 张三值班", "agent_claim": "张三值班"},
        ],
        label="hallucinated",
        expected_failure_modes=[("LOOP", 2)],
        note="2 次相同排班查询 → LOOP@2"))

    cases.append(BenchmarkCase(
        id="J5_payment_retry", task="查询支付状态",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"支付流水PAY-001"}',
             "observation": "支付超时", "agent_claim": "重试中"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"支付流水PAY-002"}',
             "observation": "支付超时", "agent_claim": "重试中"},
            {"step": 3, "action": "search",
             "action_input": '{"query":"支付流水PAY-003"}',
             "observation": "支付成功", "agent_claim": "支付成功"},
        ],
        label="hallucinated",
        expected_failure_modes=[("RETRY_STORM", 3)],
        note="3 不同流水号 + 超时失败 → RETRY_STORM@3"))

    cases.append(BenchmarkCase(
        id="J6_crawl_loop", task="抓取数据",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"商品SKU-A001"}',
             "observation": "SKU-A001 库存30件", "agent_claim": "库存30件"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"商品SKU-A001"}',
             "observation": "SKU-A001 库存30件", "agent_claim": "库存30件"},
            {"step": 3, "action": "search",
             "action_input": '{"query":"商品SKU-A001"}',
             "observation": "SKU-A001 库存30件", "agent_claim": "库存30件"},
        ],
        label="hallucinated",
        expected_failure_modes=[("LOOP", 2)],
        note="3 次相同 SKU 抓取 → LOOP@2,3"))

    cases.append(BenchmarkCase(
        id="J7_queue_retry", task="查询队列状态",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"任务TASK-100"}',
             "observation": "任务异常", "agent_claim": "排队中"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"任务TASK-200"}',
             "observation": "任务异常", "agent_claim": "排队中"},
            {"step": 3, "action": "search",
             "action_input": '{"query":"任务TASK-300"}',
             "observation": "任务异常", "agent_claim": "排队中"},
        ],
        label="hallucinated",
        expected_failure_modes=[("RETRY_STORM", 3)],
        note="3 不同任务号 + 异常失败 → RETRY_STORM@3"))

    # --- K. 多步传播链 ×4（幻觉从 step1 传播到 step2 的 claim）---
    # step2 claim 篡改 step2 obs 的数值 → CLAIM_MISMATCH@2；体现跨步传播
    cases.append(BenchmarkCase(
        id="K1_order_to_invoice", task="开订单发票",
        trajectory=[
            {"step": 1, "action": "lookup_order",
             "action_input": '{"order_id":"ORD-9988"}',
             "observation": "订单ORD-9988金额1234.56元", "agent_claim": "订单ORD-9988金额1234.56元"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"ORD-9988发票金额"}',
             "observation": "发票金额1234.56元", "agent_claim": "发票金额1234.00元与订单一致"},
        ],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 2)],
        note="传播：step2 金额 1234.56→1234.00 篡改并声称一致"))

    cases.append(BenchmarkCase(
        id="K2_symptom_to_drug", task="查询用药建议",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"患者发热症状"}',
             "observation": "患者发热38.5度", "agent_claim": "患者发热38.5度"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"推荐用药"}',
             "observation": "推荐对乙酰氨基酚每次500mg", "agent_claim": "推荐对乙酰氨基酚每次800mg"},
        ],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 2)],
        note="传播：step2 剂量 500→800mg 篡改"))

    cases.append(BenchmarkCase(
        id="K3_quote_to_contract", task="签订合同",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"供应商报价"}',
             "observation": "报价单台5万元", "agent_claim": "报价单台5万元"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"合同金额"}',
             "observation": "合同金额5万元", "agent_claim": "合同金额8万元已确认"},
        ],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 2)],
        note="传播：step2 金额 5→8万 篡改"))

    cases.append(BenchmarkCase(
        id="K4_weather_to_flight", task="查询航班动态",
        trajectory=[
            {"step": 1, "action": "get_weather",
             "action_input": '{"city":"北京"}',
             "observation": "北京雷暴天气", "agent_claim": "北京雷暴天气"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"航班延误情况"}',
             "observation": "航班延误120分钟", "agent_claim": "航班延误12分钟已通知旅客"},
        ],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 2)],
        note="传播：step2 延误 120→12 分钟篡改"))

    # --- L. 混合失败轨迹 ×3（跨域多工具，2 码/条）---
    cases.append(BenchmarkCase(
        id="L1_weather_transit", task="查询上海天气与交通",
        trajectory=[
            {"step": 1, "action": "get_weather",
             "action_input": '{"city":"上海"}',
             "observation": "上海降雨量25毫米", "agent_claim": "上海降雨量250毫米"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"上海地铁运营"}',
             "observation": "上海地铁3号线运营", "agent_claim": "上海地铁8号线停运"},
        ],
        label="hallucinated",
        expected_failure_modes=[("CLAIM_MISMATCH", 1), ("CLAIM_MISMATCH", 2)],
        note="混合：step1 降雨量 25→250 毫米 + step2 地铁 3→8 号线"))

    cases.append(BenchmarkCase(
        id="L2_order_calc", task="查询订单并算折扣",
        trajectory=[
            {"step": 1, "action": "lookup_order",
             "action_input": '{"order_id":"ORD-5566"}',
             "observation": "订单ORD-5566状态为空", "agent_claim": "订单ORD-5566已发货共3件"},
            {"step": 2, "action": "calculator",
             "action_input": '{"expression":"999*0.1"}',
             "observation": "99.9", "agent_claim": "折扣后100元"},
        ],
        label="hallucinated",
        expected_failure_modes=[("OBSERVATION_IGNORED", 1), ("CLAIM_MISMATCH", 2)],
        note="混合：step1 零值软失败(为空→发货) + step2 折扣 99.9→100 元"))

    cases.append(BenchmarkCase(
        id="L3_medical_multi", task="查询体检与用药",
        trajectory=[
            {"step": 1, "action": "search",
             "action_input": '{"query":"白细胞计数"}',
             "observation": "体检结果白细胞计数0", "agent_claim": "体检结果白细胞计数共5000已确认"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"推荐剂量"}',
             "observation": "推荐剂量500毫克", "agent_claim": "推荐剂量5000毫克"},
        ],
        label="hallucinated",
        expected_failure_modes=[("OBSERVATION_IGNORED", 1), ("CLAIM_MISMATCH", 1), ("CLAIM_MISMATCH", 2)],
        note="混合：step1 零值 0→5000 + step2 剂量 500→5000 毫克"))

    # --- M. 跨步观察投毒 ×2（后续步 obs 覆盖前步真值，agent 采信）---
    # step2 claim 含纠正标记 → HANDOFF_LOSS@2（确定性回退路径，无 NLI 依赖）
    cases.append(BenchmarkCase(
        id="M1_amount_poison", task="复核订单金额",
        trajectory=[
            {"step": 1, "action": "lookup_order",
             "action_input": '{"order_id":"ORD-1234"}',
             "observation": "订单ORD-1234金额500元", "agent_claim": "订单ORD-1234金额500元"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"ORD-1234最新金额"}',
             "observation": "订单ORD-1234金额800元", "agent_claim": "订单ORD-1234金额800元更正为实800"},
        ],
        label="hallucinated",
        expected_failure_modes=[("HANDOFF_LOSS", 2)],
        note="投毒：step2 obs 800 覆盖 step1 的 500，claim 更正 → HANDOFF_LOSS@2"))

    cases.append(BenchmarkCase(
        id="M2_weather_poison", task="复核天气",
        trajectory=[
            {"step": 1, "action": "get_weather",
             "action_input": '{"city":"杭州"}',
             "observation": "杭州晴30度", "agent_claim": "杭州晴30度"},
            {"step": 2, "action": "search",
             "action_input": '{"query":"杭州实时天气"}',
             "observation": "杭州暴雨28度", "agent_claim": "杭州晴30度并非实为暴雨28度"},
        ],
        label="hallucinated",
        expected_failure_modes=[("HANDOFF_LOSS", 2)],
        note="投毒：step2 obs 暴雨覆盖 step1 晴，claim 并非纠正 → HANDOFF_LOSS@2"))

    # --- N. 对抗 clean 探针 ×1（expected=[]，FP 探针）---
    # 合理但略可疑的 agent 输出，数字忠实、单位一致，不应触发任何码；
    # 若触发 = 误报（fp_rate 计入）
    cases.append(BenchmarkCase(
        id="N1_balance_probe", task="查询用户余额",
        trajectory=[{"step": 1, "action": "search",
            "action_input": '{"query":"用户张三余额"}',
            "observation": "用户张三余额5000元",
            "agent_claim": "用户张三余额充足共5000元"}],
        label="clean",
        expected_failure_modes=[],
        note="FP 探针：余额 5000 数字忠实、「充足」主观但不算幻觉；期望 0 检出"))

    return cases


# 对抗探针（泛化边界探测，非一致性回归；见 tests/test_agent_adversarial.py）
ADVERSARIAL_CASES: list[BenchmarkCase] = _build_adversarial_cases()


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


# ============================ 落盘 + CLI ============================

# standard 档子集：20 条 = 契约套件前 10（5 clean + 5 hall，覆盖确定性码）
# + 原始 10 条对抗（A-J，覆盖 LLM-only 的 SCOPE_DRIFT/HANDOFF_LOSS 等）。
# 用于在 ~¥0.5-2 成本内抽检 fast 档不覆盖的 LLM-only 模式。
_STANDARD_ADVERSARIAL_IDS = (
    "A_number_tweak", "B_id_swap", "C_semantic_fabrication", "D_silent_handoff",
    "E_scope_drift", "F_fabricated_city", "G_zero_value_ignored", "H_common_sense",
    "I_unit_fabrication", "J_loop_bypass",
)


def _build_standard_subset() -> list[BenchmarkCase]:
    """standard 档 20 条子集：契约前 10 + 原始 10 条对抗。"""
    adv_by_id = {c.id: c for c in ADVERSARIAL_CASES}
    missing = [i for i in _STANDARD_ADVERSARIAL_IDS if i not in adv_by_id]
    if missing:
        raise RuntimeError(f"STANDARD_SUBSET 引用了不存在的对抗 id: {missing}")
    contract_10 = BENCHMARK_CASES[:10]
    adversarial_10 = [adv_by_id[i] for i in _STANDARD_ADVERSARIAL_IDS]
    return list(contract_10) + adversarial_10


STANDARD_SUBSET: list[BenchmarkCase] = _build_standard_subset()


_REPO_ROOT = Path(__file__).resolve().parent


def _write_json(payload: dict, write_json: str) -> None:
    """写 canonical 路径；canonical 在仓库内且前端目录存在时，镜像到 web/public/eval_set/。

    2026-10-13 事故：旧实现用 path.parent.parent / "web"/... 推镜像路径，
    canonical=/tmp/x.json 时解析成文件系统根的 /web/public/eval_set/ 并 mkdir 出来。
    镜像只能锚定仓库根（__file__），绝不能从 canonical 路径反推。
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path = Path(write_json).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    mirror_dir = _REPO_ROOT / "web" / "public" / "eval_set"
    try:
        inside_repo = path.is_relative_to(_REPO_ROOT)
    except AttributeError:  # Python < 3.9
        inside_repo = str(path).startswith(str(_REPO_ROOT) + "/")
    if inside_repo and mirror_dir.is_dir():
        mirror = mirror_dir / "agent_benchmark_results.json"
        mirror.write_text(text, encoding="utf-8")
        print(f"已落盘 {path}（镜像 → {mirror}）")
    else:
        print(f"已落盘 {path}（仓库外或无 web/public/，跳过前端镜像）")


def run_and_persist(
    speed: str = "fast",
    write_json: str | None = None,
    cases: list[BenchmarkCase] | None = None,
    suite_label: str = "",
) -> dict:
    """跑基准并（可选）落盘 JSON。

    不修改 BenchmarkReport.to_dict（其 10 键契约被现有测试断言）——本函数把
    to_dict() 输出包进一个独立 wrapper dict，仅在顶层加 meta 块。
    返回：{meta:{generated_at,speed,sample_count,suite_label}, ...to_dict()。
    """
    report = run_benchmark(speed=speed, cases=cases)
    payload: dict = {"meta": {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "speed": speed,
        "sample_count": report.total,
        "suite_label": suite_label,
    }}
    payload.update(report.to_dict())
    if write_json:
        _write_json(payload, write_json)
    return payload


def persist_all(write_json: str | None = None) -> dict:
    """便利函数：一次跑 fast 全 100（contract+adversarial+total）+ standard 子集 20。

    返回三层 JSON：{fast:{contract,adversarial,total}, standard:{subset_20}}。
    fast 档秒级 0 LLM；standard 档 ~5-15 min、~¥0.5-2 DeepSeek 成本。
    """
    fast_contract = run_and_persist("fast", None, BENCHMARK_CASES, "contract_20")
    fast_adversarial = run_and_persist("fast", None, ADVERSARIAL_CASES, "adversarial_80")
    fast_total = run_and_persist(
        "fast", None, BENCHMARK_CASES + ADVERSARIAL_CASES, "total_100")
    standard_subset = run_and_persist("standard", None, STANDARD_SUBSET, "standard_subset_20")

    bundle = {
        "meta": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "total_cases": len(BENCHMARK_CASES) + len(ADVERSARIAL_CASES),
            "contract_cases": len(BENCHMARK_CASES),
            "adversarial_cases": len(ADVERSARIAL_CASES),
            "standard_subset_cases": len(STANDARD_SUBSET),
            "coverage_gap": list(COVERAGE_GAP),
            "failure_mode_codes": list(FAILURE_MODE_CODES),
        },
        "fast": {
            "contract": fast_contract,
            "adversarial": fast_adversarial,
            "total": fast_total,
        },
        "standard": {
            "subset_20": standard_subset,
        },
    }
    if write_json:
        _write_json(bundle, write_json)
    return bundle


def _suite_cases(suite: str) -> tuple[list[BenchmarkCase], str]:
    """--suite 选项 → (cases, label)。"""
    if suite == "contract":
        return BENCHMARK_CASES, "contract_20"
    if suite == "adversarial":
        return ADVERSARIAL_CASES, "adversarial_80"
    if suite == "standard_subset":
        return STANDARD_SUBSET, "standard_subset_20"
    # all
    return BENCHMARK_CASES + ADVERSARIAL_CASES, "total_100"


def main(argv: list[str] | None = None, *, speed: str | None = None) -> None:
    """CLI 入口。

    两种调用方式：
      - 旧式兼容：``main(speed="fast")`` —— 不走 argparse，直接跑 fast 基准并打印。
      - CLI：``main(["--speed","all","--write-json",PATH])`` 或脚本无参入口（走 argparse）。

    默认 ``--speed all``：跑 fast 全 100 + standard 子集 20，写三层 JSON。
    ``--speed fast/standard`` 单档；``--suite`` 选套件；``--write-json`` 落盘路径。
    """
    # 旧式兼容：显式传 speed= 时走简易路径（不解析 argv，不落盘）
    if speed is not None:
        report = run_benchmark(speed=speed, cases=BENCHMARK_CASES + ADVERSARIAL_CASES)
        d = report.to_dict()
        cm = d["case_metrics"]
        print("=" * 64)
        print(f"HallucC-Agent-Bench（speed={speed}，{report.total} 条）")
        print("=" * 64)
        print(f"  recall={cm['recall']}  fp_rate={cm['fp_rate']}  "
              f"TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}")
        print(f"  llm_calls={report.llm_calls}  search_calls={report.search_calls}")
        return

    parser = argparse.ArgumentParser(
        description="HallucC-Agent-Bench：跑 Agent 轨迹幻觉检测基准。")
    parser.add_argument(
        "--speed", choices=("fast", "standard", "all"), default="all",
        help="fast=0 LLM 秒级；standard=LLM judge ~5-15min；all=fast 全 100 + standard 子集 20（默认）")
    parser.add_argument(
        "--suite", choices=("all", "contract", "adversarial", "standard_subset"),
        default="all", help="套件选择（--speed fast/standard 时生效）")
    parser.add_argument(
        "--write-json", default=None, metavar="PATH",
        help="落盘 JSON 路径（canonical）；在仓库内且存在 web/public/ 时镜像到 web/public/eval_set/")
    args = parser.parse_args(argv)

    if args.speed == "all":
        bundle = persist_all(args.write_json)
        fast = bundle["fast"]
        std = bundle["standard"]["subset_20"]
        print("=" * 64)
        print("HallucC-Agent-Bench —— fast + standard 双档（100 条样本）")
        print("=" * 64)
        for label, blk in (("契约 20", fast["contract"]),
                           ("对抗 80", fast["adversarial"]),
                           ("合计 100", fast["total"])):
            cm = blk["case_metrics"]
            print(f"[fast · {label}] recall={cm['recall']} fp_rate={cm['fp_rate']} "
                  f"TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}")
        scm = std["case_metrics"]
        print(f"[standard · 子集 20] recall={scm['recall']} fp_rate={scm['fp_rate']} "
              f"TP={scm['tp']} FN={scm['fn']} FP={scm['fp']} TN={scm['tn']}")
        print(f"\ncoverage_gap(LLM-only, fast 不覆盖): {bundle['meta']['coverage_gap']}")
        if args.write_json:
            print(f"已落盘: {args.write_json} + web/public/eval_set/agent_benchmark_results.json")
        return

    cases, label = _suite_cases(args.suite)
    payload = run_and_persist(args.speed, args.write_json, cases, label)
    cm = payload["case_metrics"]
    print("=" * 64)
    print(f"HallucC-Agent-Bench（speed={args.speed}，套件={args.suite}，{payload['meta']['sample_count']} 条）")
    print("=" * 64)
    print(f"  recall={cm['recall']}  fp_rate={cm['fp_rate']}  "
          f"TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}")
    print(f"  inst_recall={payload['instance_metrics']['recall']}  "
          f"inst_precision={payload['instance_metrics']['precision']}")
    print(f"  llm_calls={payload['llm_calls']}  search_calls={payload['search_calls']}")
    print(f"  coverage_gap(LLM-only): {payload['coverage_gap']}")
    if args.write_json:
        print(f"已落盘: {args.write_json} + web/public/eval_set/agent_benchmark_results.json")


if __name__ == "__main__":
    main()

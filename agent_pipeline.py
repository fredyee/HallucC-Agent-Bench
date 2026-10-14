"""Agent 分层递进管线 (P1)。规则层(0 LLM) → 裁判层(单 LLM/步) → 搜索/重放层(deep)。

三档速度模式接入 agent 端点（docs/agent优化开发计划.md P1 1.2）：
  - fast:     仅规则层，0 LLM；reasoning_coherence/task_completion 维度 score=1.0 conf=0.0
  - standard: 规则层 + 单裁判/步（仅不确定项），单步 LLM 调用 ≤1
  - deep:     全管线 + strict 多次投票(K=3, trajectory-level ChainPoll 类比)

设计目标（P1 验收）：
  - 单步 LLM 调用数从 3 降至 ≤1（standard 档）
  - 六维 vs 人工 Cohen's kappa ≥0.6
  - fast 档 P95 <500ms

分层递进原则：每层只处理前一层标记"不确定"的项，避免全量 LLM 调用。
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import math_checker
from agent_eval import (
    evaluate as _rule_evaluate,
    _normalize_step,
    _text_similarity,
    CLAIM_MISMATCH_THRESHOLD,
)
from log_utils import get_logger
from common_sense import _is_obviously_false
from failure_modes import (
    detect_param_fabrication,
    detect_observation_ignored,
    detect_redundant_steps,
    detect_retry_storm,
    detect_handoff_loss,
)

logger = get_logger(__name__)

# 裁判层灰区：rule-layer 相似度落在此区间内需 LLM 裁判确认（rule 判 pass 但偏弱）
_OBS_GRAY_LOW = 0.3
_OBS_GRAY_HIGH = 0.6

# deep 档 strict 投票采样数（trajectory-level ChainPoll 类比）
_VOTE_K = 3

# 六维契约 v2
DIM_TOOL_SELECTION = "tool_selection"
DIM_PARAMETER_VALIDITY = "parameter_validity"
DIM_OBSERVATION_FAITHFULNESS = "observation_faithfulness"
DIM_REASONING_COHERENCE = "reasoning_coherence"
DIM_TASK_COMPLETION = "task_completion"
DIM_TRAJECTORY_EFFICIENCY = "trajectory_efficiency"

SIX_DIM = {
    DIM_TOOL_SELECTION,
    DIM_PARAMETER_VALIDITY,
    DIM_OBSERVATION_FAITHFULNESS,
    DIM_REASONING_COHERENCE,
    DIM_TASK_COMPLETION,
    DIM_TRAJECTORY_EFFICIENCY,
}

# 维度 → 失败模式码（裁判 refuted 时归类）
_DIM_FAILURE_CODE = {
    DIM_OBSERVATION_FAITHFULNESS: "CLAIM_MISMATCH",
    DIM_REASONING_COHERENCE: "CLAIM_MISMATCH",
    DIM_TASK_COMPLETION: "PREMATURE_STOP",
}

_JUDGE_VERDICT_MAP = {
    "supported": "pass",
    "refuted": "fail",
    "unverified": "uncertain",
}


@dataclass
class AgentPipelineResult:
    """agent_pipeline 输出。dimensions 为六维；steps[].checks 含规则+裁判检查项。"""
    speed: str = "standard"
    dimensions: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    failure_modes: list = field(default_factory=list)
    propagation: list = field(default_factory=list)
    llm_calls: int = 0
    search_calls: int = 0
    truncated: bool = False
    total_steps: int = 0
    hallucinated_steps: int = 0
    # 规则层 base（AgentEvalResult），供端点取 6-float 向后兼容契约
    base: Any = None

    def to_dict(self) -> dict:
        return {
            "speed": self.speed,
            "dimensions": self.dimensions,
            "steps": self.steps,
            "failure_modes": self.failure_modes,
            "propagation": self.propagation,
            "llm_calls": self.llm_calls,
            "search_calls": self.search_calls,
            "truncated": self.truncated,
            "total_steps": self.total_steps,
            "hallucinated_steps": self.hallucinated_steps,
        }


# ============================ 主入口 ============================

def evaluate_trajectory(
    trajectory: list[dict],
    tool_schemas: dict | None = None,
    *,
    speed: str = "standard",
    task: str = "",
    knowledge_id: int | None = None,
    user_id: int | None = None,
    max_llm_calls: int = 60,
    llm: Any = None,
) -> AgentPipelineResult:
    """对 Agent 执行轨迹做分层递进评估。

    trajectory: [{step, thought, action, action_input, observation, agent_claim/final_answer}, ...]
    兼容 /detect-agent 的 {tool, tool_input, tool_output, agent_claim} 字段名（由 _normalize_step 归一）。
    """
    result = AgentPipelineResult(speed=speed)
    steps = list(trajectory or [])
    result.total_steps = len(steps)

    if not steps:
        # 空轨迹仍设置 base（默认 AgentEvalResult，6 float 全 1.0），
        # 保 /detect-agent steps=[] 时 agent_eval 6 键向后兼容契约。
        result.base = _rule_evaluate(list(steps), tool_schemas)
        result.dimensions = {d: {"score": 1.0, "confidence": 1.0} for d in SIX_DIM}
        return result

    # ===== 1. 规则层（0 LLM）=====
    base, steps_checks, failure_modes, per_step_meta = _rule_layer(steps, tool_schemas, task, speed=speed)
    result.steps = steps_checks
    result.failure_modes = failure_modes
    result.base = base

    # ===== 2. 裁判层（standard / deep）=====
    state = {"llm_calls": 0, "truncated": False}
    dim_scores: dict[str, list[float]] = {}

    if speed != "fast":
        if llm is None:
            try:
                from llm import LLMClient
                llm = LLMClient()
            except Exception as e:
                logger.warning("裁判层 LLMClient 初始化失败，降级为规则层结果: %s", str(e)[:100])
                llm = None
        if llm is not None:
            dim_scores = _judge_layer(
                steps_checks, failure_modes, per_step_meta,
                speed=speed, task=task, llm=llm,
                max_llm_calls=max_llm_calls, state=state,
            )

    result.llm_calls = state["llm_calls"]
    result.truncated = state["truncated"]

    # ===== 3. 搜索/重放层（deep only，P1 占位）=====
    # P1: deep 档的"全管线"主要体现在 strict 投票(K=3)；事实声明仍 uncertain 时的
    # 搜索/重放核验留 P2 语义 DAG 扩展。此处保守不触发外部搜索，保持离线可测。
    result.search_calls = 0

    # ===== 维度聚合 + 传播链 =====
    result.dimensions = _aggregate_dimensions(base, per_step_meta, dim_scores, speed)
    result.hallucinated_steps = _count_hallucinated(steps_checks)
    result.propagation = _propagation_path(per_step_meta, failure_modes, speed=speed)
    # propagation_score 自传播链派生（去循环：不再依赖输入 has_hallucination 标签）
    result.base.propagation_score = _propagation_score(result.propagation, result.total_steps)

    return result


# ============================ 规则层 ============================

def _rule_layer(steps: list[dict], tool_schemas: dict | None, task: str = "", *, speed: str = "standard"):
    """规则层（0 LLM）：agent_eval + math_checker + nli_model + _is_obviously_false
    + P2 确定性失败模式探测器（failure_modes.py）。

    在 agent_eval 的确定性检查之上，叠加：
      - math_checker: 声称/最终回答中的算式执行核验（P1 规则层组件）
      - nli_model: observation↔claim 语义蕴含（Jaccard 抓不住的语义矛盾），
        并裁决 base 层的词法误判——词法相似度降为预筛，NLI 判一致则撤销
        「观察与声称矛盾」，NLI 不可用时保持词法判定（宁误报不漏报）
      - _is_obviously_false: 常识矛盾（日期不可能、百分比>100、年份悖论）
      - P2 逐 步：param_fabrication / observation_ignored
      - P2 轨迹级：redundant_step / retry_storm / handoff_loss
    task 供 param_fabrication 的上下文来源校验（agent 调用前可知的原始指令）。
    speed 透传给 NLI tier 选择（deep→330M）与 detect_handoff_loss（deep→330M）。
    返回 (base, steps_checks, failure_modes, per_step_meta)。
    """
    base = _rule_evaluate(list(steps), tool_schemas)
    structured = base.structured()
    steps_checks = structured["steps"]
    failure_modes = list(structured["failure_modes"])
    normalized = [_normalize_step(s, i) for i, s in enumerate(steps)]

    # param_fabrication 上下文：task + 前序步的 obs/claim（不含当前步 obs——
    # 当前步 obs 是本次调用结果，agent 调用前不可能已知）
    context_parts: list[str] = [task] if task else []
    per_step_meta: list[dict] = []
    # 忠实度分数重算口径：与 base 一致（observation + agent_claim 同时存在），
    # 但 NLI 判一致的步骤按其蕴含概率计，不被词法 0.2 拉成 0
    faithful_scores_adj: list[float] = []
    for i, ns in enumerate(normalized):
        sc = steps_checks[i] if i < len(steps_checks) else {"step": ns["step"], "checks": []}
        step_no = ns["step"]
        obs = ns["observation"]
        claim = ns["agent_claim"] or ns["final_answer"]
        final_answer = ns["final_answer"]

        meta: dict = {
            "step": step_no,
            "thought": ns["thought"],
            "action": ns["action"],
            "action_input": ns["action_input"],
            "observation": obs,
            "agent_claim": claim,
            "final_answer": final_answer,
            "obs_score": None,
            "nli": None,
            "math_issues": [],
            "obviously_false": False,
            "is_hallucinated": False,
        }

        # --- math_checker: 算式执行核验 ---
        for text in (claim, final_answer):
            if not text:
                continue
            try:
                issues = math_checker.check(text)
            except Exception as e:
                logger.warning("math_checker 失败(step %s): %s", step_no, str(e)[:80])
                issues = []
            if issues:
                meta["math_issues"].extend(issues)
        if meta["math_issues"]:
            for mi in meta["math_issues"]:
                sc["checks"].append({
                    "type": "claim_mismatch", "verdict": "mismatch",
                    "detail": f"算式 {mi['expression']}={mi['claimed']} 实际={mi['actual']}",
                    "source": "math_checker", "confidence": 0.95,
                })
            _ensure_failure_mode(failure_modes, "CLAIM_MISMATCH", step_no, "high")

        # --- nli_model: observation↔claim 语义蕴含 ---
        # speed=deep → 330M 精审；否则 110M（同步 claim↔obs 可信 NLI 裸判，Lesson 36）
        nli = None
        if obs and claim:
            try:
                from nli_model import evaluate_faithfulness
                nli = evaluate_faithfulness(claim, obs, tier=("deep" if speed == "deep" else "fast"))
            except Exception as e:
                logger.warning("nli 评估失败(step %s): %s", step_no, str(e)[:80])
                nli = None
        meta["nli"] = nli
        if nli and nli.get("verdict") == "contradicted":
            sc["checks"].append({
                "type": "claim_mismatch", "verdict": "mismatch",
                "detail": f"NLI 判定矛盾(score={nli.get('score')})",
                "source": "nli_model", "confidence": round(nli.get("score", 0.5), 2),
            })
            _ensure_failure_mode(failure_modes, "CLAIM_MISMATCH", step_no, "high")

        # --- 词法相似度降为预筛：NLI 判一致则撤销词法误判（Bug 4）---
        # CLAIM_MISMATCH 原本拿词法重叠当否决条件：观察说「可退款」、声称说
        # 「符合退款条件」语义一致，字符 Jaccard 只有 0.2，被当成矛盾拦成误报。
        # NLI 可用时它是最终裁决者；NLI 不可用时保持词法判定（宁误报不漏报）。
        # 只撤词法来源的误判——math_checker / claim_alignment / common_sense
        # 与词法重叠无关，一条都不动。
        if nli and nli.get("verdict") == "supported":
            lex_mismatch = [
                c for c in sc["checks"]
                if c["type"] == "claim_mismatch" and c["verdict"] == "mismatch"
                and c.get("source") in (None, "deterministic")
                and c.get("detail", "").startswith("观察与声称相似度")
            ]
            if lex_mismatch:
                for c in lex_mismatch:
                    c["verdict"] = "pass"
                    c["source"] = "nli_model"
                    c["detail"] = (
                        f"相似度 {c['confidence']} < 阈值 {CLAIM_MISMATCH_THRESHOLD}，"
                        f"但 NLI 判一致(score={nli.get('score')})——撤销词法误判"
                    )
                    c["confidence"] = round(nli.get("score", 0.5), 2)
                # 本步若无其他矛盾信号，收回该步的 CLAIM_MISMATCH 失败模式
                if not any(
                    c.get("verdict") in ("mismatch", "fail", "loop")
                    for c in sc["checks"]
                ):
                    failure_modes[:] = [
                        fm for fm in failure_modes
                        if not (fm["code"] == "CLAIM_MISMATCH" and fm["step"] == step_no)
                    ]
                meta["nli_override_lexical"] = True

        # --- claim_alignment: 关键实体一致性（补 Jaccard 高重叠漏判）---
        # 整体相似度 ≥0.5 但数字/ID/单位被篡改时仍发 CLAIM_MISMATCH（A/B/I 例）。
        # 与 nli_model 互补：NLI 抓语义矛盾，claim_alignment 抓精确实体篡改；
        # 两者均纯规则/确定性，NLI 不可用时仍生效。
        if obs and claim:
            try:
                from claim_alignment import compare as ca_compare
                ca = ca_compare(obs, claim)
            except Exception as e:
                logger.warning("claim_alignment 失败(step %s): %s", step_no, str(e)[:80])
                ca = None
            if ca:
                types = [m["type"] for m in ca.get("mismatches", [])]
                sc["checks"].append({
                    "type": "claim_mismatch", "verdict": "mismatch",
                    "detail": f"关键实体不一致: {','.join(types)}",
                    "source": "claim_alignment", "confidence": ca.get("confidence", 0.8),
                })
                _ensure_failure_mode(failure_modes, "CLAIM_MISMATCH", step_no, "high")

        # --- observation 相似度（供裁判层升级判定）---
        if obs and claim:
            lex_score = _text_similarity(obs, claim)
            meta["obs_score"] = lex_score
            if ns["agent_claim"]:
                supported = bool(nli) and nli.get("verdict") == "supported"
                if supported and lex_score < CLAIM_MISMATCH_THRESHOLD:
                    faithful_scores_adj.append(round(nli.get("score", 0.5), 2))
                else:
                    faithful_scores_adj.append(lex_score)

        # --- _is_obviously_false: 常识矛盾 ---
        for text in (claim, final_answer):
            if text and _is_obviously_false(text):
                meta["obviously_false"] = True
                sc["checks"].append({
                    "type": "claim_mismatch", "verdict": "mismatch",
                    "detail": "常识校验发现明显矛盾",
                    "source": "common_sense", "confidence": 0.85,
                })
                _ensure_failure_mode(failure_modes, "OVERCONFIDENT_ASSERTION", step_no, "high")
                break

        # --- P2 确定性失败模式探测器（逐 步）---
        # param_fabrication：参数含刚性 ID 但 task/前序观察与声称中无来源；
        # 非刚性 ID 路径：obs 否定 + claim 仍断言查询的字符串参数值（F 例）
        ctx_text = " ".join(context_parts)
        _apply_findings(
            detect_param_fabrication(step_no, ns["action_input"], ctx_text,
                                     observation=obs, claim=claim),
            steps_checks, failure_modes,
        )
        # observation_ignored：观察显式失败/为空却声称成功
        oi = detect_observation_ignored(step_no, obs, claim)
        if oi:
            _apply_findings([oi], steps_checks, failure_modes)

        meta["is_hallucinated"] = (
            meta["obviously_false"]
            or bool(meta["math_issues"])
            or (nli is not None and nli.get("verdict") == "contradicted")
            or any(c.get("verdict") in ("mismatch", "fail", "loop") for c in sc.get("checks", []))
        )
        per_step_meta.append(meta)
        # 累积当前步 obs/claim 供后续步的 param_fabrication 上下文（不含于本步判定）
        if obs:
            context_parts.append(obs)
        if claim:
            context_parts.append(claim)

    # --- P2 确定性失败模式探测器（轨迹级）---
    _apply_findings(detect_redundant_steps(normalized), steps_checks, failure_modes)
    _apply_findings(detect_retry_storm(normalized), steps_checks, failure_modes)
    _apply_findings(detect_handoff_loss(per_step_meta, speed=speed), steps_checks, failure_modes)

    # step_efficiency 按 REDUNDANT_STEP/RETRY_STORM 重算覆盖 agent_eval 的 legacy 基值
    # （unique_actions/n 误判同工具不同入参的合法多次调用；冗余信号只在 failure_modes 层可得）。
    base.step_efficiency = _step_efficiency(failure_modes, len(normalized))

    # output_faithfulness 覆盖 agent_eval 的纯词法口径：NLI 判一致的步骤按蕴含概率计，
    # 否则「换个说法重述」会被 0.2 的相似度拉成 0（结论在 base 层已被污染）
    if faithful_scores_adj:
        base.output_faithfulness = sum(faithful_scores_adj) / len(faithful_scores_adj)

    return base, steps_checks, failure_modes, per_step_meta


# ============================ 裁判层 ============================

def _judge_layer(steps_checks, failure_modes, per_step_meta, *,
                 speed, task, llm, max_llm_calls, state) -> dict[str, list[float]]:
    """裁判层：对不确定步骤做单次 LLM 裁判（standard）或 K 次投票（deep）。

    单步 LLM ≤1（standard）：优先 observation_faithfulness（不确定时），其次 reasoning_coherence。
    task_completion 为 trajectory-level 单独一次（对最终回答）。
    deep 档：K=_VOTE_K 次采样 + 多数投票（strict, ChainPoll 类比）。
    """
    k = _VOTE_K if speed == "deep" else 1
    dim_scores: dict[str, list[float]] = {
        DIM_OBSERVATION_FAITHFULNESS: [],
        DIM_REASONING_COHERENCE: [],
        DIM_TASK_COMPLETION: [],
    }

    # --- 逐步裁判 ---
    for meta in per_step_meta:
        if state["truncated"]:
            break
        step_no = meta["step"]
        sc = next((x for x in steps_checks if x["step"] == step_no), None)
        if sc is None:
            continue

        prompt = None
        dim = None
        # 优先 observation_faithfulness（语义矛盾 Jaccard 漏判）
        if _needs_observation_judge(meta):
            prompt = _build_observation_judge_prompt(meta)
            dim = DIM_OBSERVATION_FAITHFULNESS
        elif meta.get("thought") and meta.get("action"):
            # 其次 reasoning_coherence（思考↔行动自洽）
            prompt = _build_reasoning_judge_prompt(meta)
            dim = DIM_REASONING_COHERENCE

        if prompt is None or dim is None:
            continue

        # 预算检查：需 k 次调用
        if state["llm_calls"] + k > max_llm_calls:
            state["truncated"] = True
            break

        verdict = _run_judge(llm, prompt, k)
        state["llm_calls"] += k

        sc["checks"].append({
            "type": dim,
            "verdict": _JUDGE_VERDICT_MAP.get(verdict.get("status"), "uncertain"),
            "detail": verdict.get("reason", ""),
            "source": "llm_judge",
            "confidence": round(verdict.get("confidence", 0.5), 2),
            "score": round(verdict.get("score", 0.5), 2),
        })
        if verdict.get("status") == "refuted":
            code = _DIM_FAILURE_CODE.get(dim, "CLAIM_MISMATCH")
            _ensure_failure_mode(failure_modes, code, step_no, "high")

        dim_scores[dim].append(verdict.get("score", 0.5))

    # --- trajectory-level task_completion（对最终回答）---
    final_meta = per_step_meta[-1] if per_step_meta else None
    final_sc = None
    if final_meta:
        final_sc = next((x for x in steps_checks if x["step"] == final_meta["step"]),
                        steps_checks[-1] if steps_checks else None)

    if final_meta and final_meta.get("final_answer") and not state["truncated"]:
        # 预算不足 → 截断标记（与逐步裁判同一语义：有裁判待跑但预算耗尽即 truncated）
        if state["llm_calls"] + k > max_llm_calls:
            state["truncated"] = True
        else:
            prompt = _build_task_completion_prompt(task, final_meta["final_answer"])
            if prompt:
                verdict = _run_judge(llm, prompt, k)
                state["llm_calls"] += k
                if final_sc is not None:
                    final_sc["checks"].append({
                        "type": DIM_TASK_COMPLETION,
                        "verdict": _JUDGE_VERDICT_MAP.get(verdict.get("status"), "uncertain"),
                        "detail": verdict.get("reason", ""),
                        "source": "llm_judge",
                        "confidence": round(verdict.get("confidence", 0.5), 2),
                        "score": round(verdict.get("score", 0.5), 2),
                    })
                if verdict.get("status") == "refuted":
                    _ensure_failure_mode(failure_modes, "PREMATURE_STOP", final_meta["step"], "high")
                dim_scores[DIM_TASK_COMPLETION].append(verdict.get("score", 0.5))

    # --- trajectory-level scope_drift（仅当提供了 task；与 task_completion 互补）---
    # task_completion 看最终回答是否兑现任务；scope_drift 看中途行动是否跑题。
    # 不计入六维，仅产出 SCOPE_DRIFT 失败模式（refuted 时）。
    if final_meta and task and not state["truncated"]:
        if state["llm_calls"] + k > max_llm_calls:
            state["truncated"] = True
        else:
            prompt = _build_scope_drift_prompt(task, _trajectory_summary(per_step_meta))
            if prompt:
                verdict = _run_judge(llm, prompt, k)
                state["llm_calls"] += k
                if final_sc is not None:
                    final_sc["checks"].append({
                        "type": "scope_drift",
                        "verdict": _JUDGE_VERDICT_MAP.get(verdict.get("status"), "uncertain"),
                        "detail": verdict.get("reason", ""),
                        "source": "llm_judge",
                        "confidence": round(verdict.get("confidence", 0.5), 2),
                        "score": round(verdict.get("score", 0.5), 2),
                    })
                if verdict.get("status") == "refuted":
                    _ensure_failure_mode(failure_modes, "SCOPE_DRIFT", final_meta["step"], "medium")

    return dim_scores


def _run_judge(llm, prompt: str, k: int) -> dict:
    """调用裁判 k 次（k=1 单次，k>1 多数投票）。失败 → unverified 降级（宁 unverified 不误判）。"""
    if k <= 1:
        return _call_judge(llm, prompt)
    verdicts = [_call_judge(llm, prompt) for _ in range(k)]
    return _majority_vote(verdicts)


def _call_judge(llm, prompt: str) -> dict:
    """单次裁判调用。返回 {status, score, confidence, reason, evidence}。"""
    try:
        result = llm.chat_json(prompt)
        if isinstance(result, dict) and result.get("status") in ("supported", "refuted", "unverified"):
            return result
        # parse_error 或异常结构 → unverified 降级
        if isinstance(result, dict) and result.get("status") == "parse_error":
            return {"status": "unverified", "score": 0.5, "confidence": 0.0,
                    "reason": "裁判输出解析失败", "evidence": ""}
    except Exception as e:
        logger.warning("裁判 LLM 调用失败: %s", str(e)[:100])
    return {"status": "unverified", "score": 0.5, "confidence": 0.0,
            "reason": "裁判调用失败，降级 unverified", "evidence": ""}


def _majority_vote(verdicts: list[dict]) -> dict:
    """K 次投票取多数 status。unverified 为弃权，不否决 substantive 票（项目规则）。

    K=3 → 需 ≥2 票同类。优先 refuted，其次 supported，均不达多数则 unverified（宁不误判）。
    """
    valid = [v for v in verdicts
             if isinstance(v, dict) and v.get("status") in ("supported", "refuted", "unverified")]
    if not valid:
        return {"status": "unverified", "score": 0.5, "confidence": 0.0, "reason": "投票无有效结果"}
    n = len(valid)
    threshold = n // 2 + 1
    counts = Counter(v["status"] for v in valid)
    # substantive 优先（refuted > supported）；unverified 为 fallback，不否决 substantive 多数
    for status in ("refuted", "supported"):
        if counts.get(status, 0) >= threshold:
            same = [v for v in valid if v["status"] == status]
            scores = [v.get("score", 0.5) for v in same]
            confs = [v.get("confidence", 0.5) for v in same]
            return {
                "status": status,
                "score": round(sum(scores) / len(scores), 2) if scores else 0.5,
                "confidence": round(sum(confs) / len(confs), 2) if confs else 0.5,
                "reason": f"K={n} 投票多数: {status}({counts[status]})",
            }
    return {"status": "unverified", "score": 0.5, "confidence": 0.0,
            "reason": f"K={n} 投票未达 substantive 多数（宁 unverified 不误判）"}


def _needs_observation_judge(meta: dict) -> bool:
    """规则层 observation↔claim 判定不确定时需 LLM 裁判确认。

    - rule 已 definitive mismatch (obs_score < CLAIM_MISMATCH_THRESHOLD) → 信任规则，不裁判
    - nli contradicted → 语义矛盾 Jaccard 漏判 → 裁判
    - obs_score 灰区 [0.3, 0.6) → rule 判 pass 但偏弱 → 裁判确认
    - 启发式：观察空/报错 但声称有结果（Jaccard 可能误评分）→ 裁判
    """
    obs = meta.get("observation") or ""
    claim = meta.get("agent_claim") or ""
    if not obs or not claim:
        return False
    obs_score = meta.get("obs_score")
    nli = meta.get("nli")

    if obs_score is not None and obs_score < CLAIM_MISMATCH_THRESHOLD:
        return False  # 规则层已确定性判 mismatch，信任规则
    if nli and nli.get("verdict") == "contradicted":
        return True
    if obs_score is not None and _OBS_GRAY_LOW <= obs_score < _OBS_GRAY_HIGH:
        return True
    # 启发式：观察为空/报错 但声称有结果
    obs_empty_like = any(w in obs for w in ["空", "报错", "error", "[]", "无任何", "null", "none", "未找到", "无结果"]) or "[]" in obs.lower()
    claim_asserts = any(d in claim for d in ["条", "成功", "完成", "已发送", "找到", "检索到", "结果"])
    if obs_empty_like and claim_asserts:
        return True
    return False


def _build_observation_judge_prompt(meta: dict) -> str:
    from prompts import AGENT_OBSERVATION_FAITHFULNESS_JUDGE
    return (AGENT_OBSERVATION_FAITHFULNESS_JUDGE
            .replace("__OBSERVATION__", meta.get("observation") or "")
            .replace("__CLAIM__", meta.get("agent_claim") or ""))


def _build_reasoning_judge_prompt(meta: dict) -> str:
    from prompts import AGENT_REASONING_COHERENCE_JUDGE
    return (AGENT_REASONING_COHERENCE_JUDGE
            .replace("__THOUGHT__", meta.get("thought") or "")
            .replace("__ACTION__", meta.get("action") or "")
            .replace("__ACTION_INPUT__", meta.get("action_input") or ""))


def _build_task_completion_prompt(task: str, final_answer: str) -> str | None:
    if not final_answer:
        return None
    from prompts import AGENT_TASK_COMPLETION_JUDGE
    return (AGENT_TASK_COMPLETION_JUDGE
            .replace("__TASK__", task or "(未提供原始任务)")
            .replace("__FINAL_ANSWER__", final_answer))


def _trajectory_summary(per_step_meta: list[dict]) -> str:
    """把轨迹压缩成短摘要供 scope_drift 裁判：每步 行号. 工具(入参) → 声称。"""
    lines = []
    for m in per_step_meta:
        step = m.get("step")
        action = m.get("action") or m.get("tool") or ""
        inp = m.get("action_input") or m.get("tool_input") or ""
        inp = inp if len(inp) <= 40 else inp[:40] + "…"
        claim = (m.get("agent_claim") or m.get("final_answer") or "").strip()
        claim = claim if len(claim) <= 60 else claim[:60] + "…"
        lines.append(f"{step}. {action}({inp}) → {claim}")
    return "\n".join(lines)


def _build_scope_drift_prompt(task: str, summary: str) -> str | None:
    if not task or not summary:
        return None
    from prompts import AGENT_SCOPE_DRIFT_JUDGE
    return (AGENT_SCOPE_DRIFT_JUDGE
            .replace("__TASK__", task)
            .replace("__TRAJECTORY_SUMMARY__", summary))


# ============================ 聚合 ============================

def _aggregate_dimensions(base, per_step_meta, dim_scores: dict, speed: str) -> dict:
    """聚合六维分数 {dim: {score, confidence}}。

    确定性维度（tool_selection/parameter_validity/trajectory_efficiency）：规则层，conf=1.0。
    observation_faithfulness：规则层，裁判介入则用裁判分数 conf=0.8。
    reasoning_coherence/task_completion：LLM 裁判；fast 档无裁判 → score=1.0 conf=0.0。
    """
    dims: dict[str, dict] = {}

    dims[DIM_TOOL_SELECTION] = {"score": round(base.tool_correctness, 2), "confidence": 1.0}
    dims[DIM_PARAMETER_VALIDITY] = {"score": round(base.argument_correctness, 2), "confidence": 1.0}
    dims[DIM_TRAJECTORY_EFFICIENCY] = {"score": round(base.step_efficiency, 2), "confidence": 1.0}

    # observation_faithfulness
    obs_judge = dim_scores.get(DIM_OBSERVATION_FAITHFULNESS) or []
    if obs_judge:
        dims[DIM_OBSERVATION_FAITHFULNESS] = {
            "score": round(sum(obs_judge) / len(obs_judge), 2), "confidence": 0.8}
    else:
        dims[DIM_OBSERVATION_FAITHFULNESS] = {
            "score": round(base.output_faithfulness, 2), "confidence": 1.0}

    # reasoning_coherence（LLM 维度；fast → 1.0/0.0）
    rc = dim_scores.get(DIM_REASONING_COHERENCE) or []
    if rc:
        dims[DIM_REASONING_COHERENCE] = {
            "score": round(sum(rc) / len(rc), 2), "confidence": 0.8}
    else:
        dims[DIM_REASONING_COHERENCE] = {"score": 1.0, "confidence": 0.0}

    # task_completion（LLM 维度；fast/无最终回答 → 1.0/0.0）
    tc = dim_scores.get(DIM_TASK_COMPLETION) or []
    if tc:
        dims[DIM_TASK_COMPLETION] = {
            "score": round(sum(tc) / len(tc), 2), "confidence": 0.8}
    else:
        dims[DIM_TASK_COMPLETION] = {"score": 1.0, "confidence": 0.0}

    return dims


def _count_hallucinated(steps_checks: list[dict]) -> int:
    """含 fail/mismatch/loop verdict 的步骤数（规则层 + 裁判层）。"""
    return sum(1 for sc in steps_checks
               if any(c.get("verdict") in ("mismatch", "fail", "loop")
                      for c in sc.get("checks", [])))


def _propagation_confidence(later_claim: str, src_claim: str, *, tier: str = "fast") -> float:
    """后续 claim 是否派生自源（幻觉）claim。

    nli 判 supported → 取其 score；contradicted → 0.0（属 HANDOFF_LOSS，非传播）；
    否则回退文本相似度。测试环境 nli 不可用 → 走文本相似度。
    tier 透传 NLI 选档（deep→330M）；默认 fast/110M。
    """
    if not later_claim or not src_claim:
        return 0.0
    nli = None
    try:
        from nli_model import evaluate_faithfulness
        nli = evaluate_faithfulness(src_claim, later_claim, tier=tier)
    except Exception as e:  # noqa: BLE001
        logger.warning("propagation nli 失败: %s", str(e)[:80])
    if nli:
        if nli.get("verdict") == "supported":
            return round(nli.get("score", 0.6), 2)
        if nli.get("verdict") == "contradicted":
            return 0.0
    return _text_similarity(src_claim, later_claim)


def _propagation_path(per_step_meta: list[dict], failure_modes: list[dict], *, speed: str = "standard") -> list[dict]:
    """P2: 语义传播 DAG——幻觉 claim 如何向后传播。

    源步（含失败模式的步）的 claim，扫描其后各步的 claim，若语义派生
    （nli supported 或文本相似度 ≥0.6）则记为下游。
    输出 {from_step, to_steps, claim, confidence}（替换 P1 的 step/action/failure_modes）。
    speed 透传给 _propagation_confidence 的 NLI tier（deep→330M）。
    """
    tier = "deep" if speed == "deep" else "fast"
    src_steps = {fm.get("step") for fm in failure_modes if fm.get("step") is not None}
    path: list[dict] = []
    for i, src in enumerate(per_step_meta):
        if src["step"] not in src_steps:
            continue
        src_claim = (src.get("agent_claim") or src.get("final_answer") or "").strip()
        to_steps: list[int] = []
        best_conf = 0.0
        for later in per_step_meta[i + 1:]:
            later_claim = (later.get("agent_claim") or later.get("final_answer") or "").strip()
            conf = _propagation_confidence(later_claim, src_claim, tier=tier)
            if conf >= 0.6:
                to_steps.append(later["step"])
                best_conf = max(best_conf, conf)
        # 即便无下游也记录源（供 UI 展示幻觉起点）；有下游则用最强关联置信度
        path.append({
            "from_step": src["step"],
            "to_steps": to_steps,
            "claim": src_claim,
            "confidence": round(best_conf, 2) if to_steps else 0.7,
        })
    return path


def _step_efficiency(failure_modes: list[dict], n: int) -> float:
    """轨迹效率 = 1 − 冗余步占比。冗余 = REDUNDANT_STEP + RETRY_STORM 失败模式数。

    LOOP 归 loop_detection 维度，不重复计入（failure_modes 设计：REDUNDANT_STEP 与 LOOP 互斥）。
    替代 agent_eval 的 unique_actions/n 基值——后者误判同工具不同入参的合法多次调用。
    """
    if n <= 0:
        return 1.0
    redundant = sum(1 for fm in failure_modes
                    if fm.get("code") in ("REDUNDANT_STEP", "RETRY_STORM"))
    return round(max(0.0, 1.0 - redundant / n), 2)


def _propagation_score(propagation_path: list[dict], n: int) -> float:
    """传播分 = 1 − 被传播下游步占比。下游步 = 出现在任一 to_steps 中的步号数。

    无传播边 → 1.0。去循环：自 _propagation_path（失败模式派生）计算，不依赖输入标签。
    """
    if n <= 0:
        return 1.0
    downstream = set()
    for p in propagation_path:
        downstream.update(p.get("to_steps", []))
    return round(max(0.0, 1.0 - len(downstream) / n), 2)


def _ensure_failure_mode(failure_modes: list, code: str, step: int, severity: str) -> None:
    """幂等追加失败模式（同 step+code 不重复）。"""
    if not any(fm.get("code") == code and fm.get("step") == step for fm in failure_modes):
        failure_modes.append({"code": code, "step": step, "severity": severity})


def _apply_findings(findings, steps_checks, failure_modes) -> None:
    """将 P2 确定性探测器结果回填到 steps[].checks + failure_modes（幂等）。

    finding: {step, code, check_type, verdict, detail, confidence, severity}
    """
    for f in findings or []:
        step_no = f["step"]
        sc = next((x for x in steps_checks if x["step"] == step_no), None)
        if sc is not None:
            sc["checks"].append({
                "type": f["check_type"],
                "verdict": f["verdict"],
                "detail": f["detail"],
                "source": "deterministic",
                "confidence": f["confidence"],
            })
        _ensure_failure_mode(failure_modes, f["code"], step_no, f["severity"])

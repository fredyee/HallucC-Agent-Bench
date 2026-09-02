"""P2 失败模式库：确定性探测器（fast 档即可报，0 LLM）。

补充 agent_eval 已覆盖的 6 种（HALLUCINATED_TOOL / SCHEMA_VIOLATION / CLAIM_MISMATCH /
LOOP / OVERCONFIDENT_ASSERTION / PREMATURE_STOP）之外的 5 种失败模式：

  - PARAM_FABRICATION   参数凭空捏造（action_input 含刚性 ID 但上下文无来源）
  - OBSERVATION_IGNORED 观察被无视（观察显式失败/为空却声称成功）
  - RETRY_STORM         重试风暴（同工具 ≥3 个不同入参且含失败观察）
  - REDUNDANT_STEP      冗余步骤（不同调用但观察高度重复）
  - HANDOFF_LOSS        交接丢失（后续 claim 纠正/推翻前序 claim）

设计原则（CLAUDE.md）：控制误报——证据不足不报；与 LOOP 互斥（LOOP 由 agent_eval 处理，
REDUNDANT_STEP 跳过同 action+input 的步骤对）。
"""
import json
import re

from agent_eval import _text_similarity, _norm_action_input
from log_utils import get_logger

logger = get_logger(__name__)

# 双语标记词表（CN literals 原样保留——植入基准的 recall=1.0/fp=0.0 依赖它们；
# EN en_words 与 regex 为纯增量，覆盖英文轨迹）。三类标记：
#   obs_failure  观察显式失败/为空
#   claim_assert 声称断言成功/有结果
#   correction   后续 claim 纠正/推翻前序 claim
# 每类含：literals（case-insensitive 子串，与重构前 _has_failure_obs 行为一致）、
#         en_words（词边界 case-insensitive，避免误匹 foundry/nullable 等）、
#         regex（带 JSON/HTTP 上下文的结构化模式，避免裸数字误报）。
MARKER_LEXICON = {
    "obs_failure": {
        "literals": (
            "空数组", "为空", "无任何", "无结果", "未找到", "未匹配", "无记录",
            "无数据", "0 条", "0条", "[]", "null", "none", "报错", "error",
            "失败", "异常", "timeout", "超时", "拒绝",
        ),
        "en_words": (
            "empty", "no results", "not found", "no records", "no data",
            "no match", "failed", "failure", "exception", "denied",
            "unauthorized", "forbidden", "invalid",
        ),
        # 仅匹配带引号/冒号上下文的 JSON/HTTP 失败信号——裸 \b5\d{2}\b 会误匹
        # 算式结果（如 c3 "123+456=579"），故不用裸状态码模式。
        "regex": (
            r'"error"\s*:\s*true',
            r'"success"\s*:\s*false',
            r'"ok"\s*:\s*false',
            r'"status"\s*:\s*[45]\d{2}',
        ),
    },
    "claim_assert": {
        "literals": (
            "成功", "完成", "已发送", "找到", "检索到", "条", "已处理",
            "已更新", "已创建", "已删除", "已生成", "已返回", "共",
        ),
        "en_words": (
            "success", "completed", "done", "sent", "found", "retrieved",
            "returned", "processed", "created", "updated", "deleted",
            "generated", "total",
        ),
        "regex": (),
    },
    "correction": {
        "literals": (
            "并非", "而非", "应为", "实际是", "实际为", "更正", "修正为",
            "错误", "不正确",
        ),
        # 不收裸 "not"（英文里太噪）；用 "rather than"/"incorrect" 等明确纠正语。
        "en_words": (
            "actually", "in fact", "correction", "corrected to", "should be",
            "rather than", "incorrect", "wrong",
        ),
        "regex": (),
    },
}


def _compile_lexicon():
    """预编译词表：literals 原样保留，en_words 编为词边界 IGNORECASE，regex 编译。"""
    compiled = {}
    for cat, spec in MARKER_LEXICON.items():
        words = tuple(
            re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)
            for w in spec.get("en_words", ())
        )
        regexes = tuple(re.compile(p) for p in spec.get("regex", ()))
        compiled[cat] = (spec["literals"], words, regexes)
    return compiled


_COMPILED = _compile_lexicon()


def _match_markers(text: str, cat: str) -> bool:
    """text 是否命中 cat 类任一标记（literals 子串 / en_words 词边界 / regex）。"""
    if not text:
        return False
    literals, words, regexes = _COMPILED[cat]
    low = text.lower()
    for m in literals:
        if m in text or m in low:  # 与原 _has_failure_obs 一致：CN+EN case-insensitive 子串
            return True
    for r in words:
        if r.search(text):
            return True
    for r in regexes:
        if r.search(text):
            return True
    return False

# 刚性 ID：5+ 位连续数字 或 UUID（订单号/交易号/用户 ID 等）。
# 不收 4 位以下的数字（避免年份/日期/金额误判），不收邮箱/URL（send_email 测试友好）。
_RIGID_ID_RE = re.compile(
    r"\d{5,}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _norm(s: str) -> str:
    return (s or "").strip()


def _has_failure_obs(obs: str) -> bool:
    """观察为空或显式包含失败标记（CN literals + EN 词边界 + JSON/HTTP regex）。"""
    o = _norm(obs)
    if not o:
        return True
    return _match_markers(o, "obs_failure")


def _asserts_success(claim: str) -> bool:
    """声称断言了成功/结果，且自身不含失败标记（避免"未找到相关条目"误报）。

    "未找到相关条目" 含断言标记"找到"，但自身含失败标记"未找到" → 返回 False。
    """
    c = _norm(claim)
    if not c:
        return False
    if _has_failure_obs(c):  # claim 自身含失败标记 → 非成功断言
        return False
    return _match_markers(c, "claim_assert")


# 独立 0 值：前后非数字/小数点——只匹配 "0元"/"0条"，不收 "10"/"30"/"0.5" 里的 0
_ZERO_VALUE_RE = re.compile(r"(?<![\d.])0+(?![\d.])")
# 任意数字（含小数/百分比）——判 claim 是否断言正数
_NUM_RE = re.compile(r"-?\d+\.?\d*%?")


def _has_zero_value(text: str) -> bool:
    """text 含独立 0 值（软失败：余额0元/结果0条）——非算式操作数里的 0。"""
    return bool(_ZERO_VALUE_RE.search(text or ""))


def _has_positive_number(text: str) -> bool:
    """text 含正数（>0）——claim 断言了具体正向数值。"""
    for m in _NUM_RE.finditer(text or ""):
        try:
            if float(m.group().rstrip("%")) > 0:
                return True
        except ValueError:
            continue
    return False


def _extract_string_param_values(action_input: str) -> list[str]:
    """从 action_input JSON 提取 string 参数值（非刚性 ID 的普通字符串参数）。

    供 param_fabrication 非刚性 ID 路径：识别「查询了某城市名/姓名但 obs 未找到」。
    """
    ai = _norm(action_input)
    if not ai:
        return []
    try:
        params = json.loads(ai.strip())
        if isinstance(params, dict):
            return [str(v) for v in params.values() if isinstance(v, str) and v]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


# ============================ 逐步探测器 ============================

def detect_param_fabrication(
    step: int, action_input: str, context_text: str,
    *, observation: str = "", claim: str = "",
) -> list[dict]:
    """参数凭空捏造：action_input 含刚性 ID，但 task/前序观察与声称中无来源。

    context_text 由调用方拼接（task + 前序 obs/claim；不含当前步 obs——
    当前步 obs 是本次调用结果，agent 调用前不可能已知）。

    扩展（非刚性 ID 路径，F 例）：obs 含失败标记（未找到/无此）且 claim 仍断言
    查询的字符串参数值 → PARAM_FABRICATION。刚性 ID 由主路径处理；本路径只收
    非刚性 ID 的普通字符串值（城市名/姓名等），len≥2 避免单字符误报。
    observation/claim 为 keyword-only 可选参数，旧调用方（仅传 3 参）零影响。
    """
    ai = _norm(action_input)
    findings = []
    # 主路径：刚性 ID（5+位数字/UUID）在上下文中无来源
    if ai:
        ctx = _norm(context_text)
        for m in _RIGID_ID_RE.finditer(ai):
            rid = m.group(0)
            if rid in ctx:
                continue  # 上下文中有来源 → 不报
            findings.append({
                "step": step,
                "code": "PARAM_FABRICATION",
                "check_type": "param_fabrication",
                "verdict": "fail",
                "detail": f"参数含刚性 ID {rid}，但上下文无来源",
                "confidence": 0.6,
                "severity": "medium",
            })
    # 非刚性 ID 路径：obs 说未找到 + claim 仍断言查询的字符串参数值
    if observation and claim and _match_markers(observation, "obs_failure"):
        for val in _extract_string_param_values(ai):
            if len(val) < 2 or _RIGID_ID_RE.fullmatch(val):
                continue  # 过短或属刚性 ID（主路径已处理）
            if val in claim:
                findings.append({
                    "step": step,
                    "code": "PARAM_FABRICATION",
                    "check_type": "param_fabrication",
                    "verdict": "fail",
                    "detail": f"obs 否定但 claim 仍断言参数值「{val}」",
                    "confidence": 0.65,
                    "severity": "medium",
                })
                break  # 一次报一条
    return findings


def detect_observation_ignored(step: int, observation: str, claim: str) -> dict | None:
    """观察被无视：观察显式失败/为空，声称却断言成功。

    扩展（值语义路径，G 例）：obs 含独立 0 值（软失败，如「余额0元」）且 claim
    断言正数成功——即使 obs 无失败关键词也报。planted 用例 obs 均无独立 0，零误报。
    """
    # 原路径：obs 显式失败/为空 + claim 断言成功
    if _has_failure_obs(observation):
        if _asserts_success(claim):
            return {
                "step": step,
                "code": "OBSERVATION_IGNORED",
                "check_type": "observation_ignored",
                "verdict": "fail",
                "detail": "观察显式失败/为空，但声称断言成功",
                "confidence": 0.85,
                "severity": "high",
            }
        return None
    # 值语义路径：obs 含 0 值（软失败）+ claim 断言正数成功
    if _has_zero_value(observation) and _asserts_success(claim) and _has_positive_number(claim):
        return {
            "step": step,
            "code": "OBSERVATION_IGNORED",
            "check_type": "observation_ignored",
            "verdict": "fail",
            "detail": "观察含 0 值（软失败），但声称断言正数成功",
            "confidence": 0.75,
            "severity": "high",
        }
    return None


# ============================ 轨迹级探测器 ============================

def detect_redundant_steps(normalized: list[dict]) -> list[dict]:
    """冗余步骤：不同 action/input 但观察高度重复（≥0.8）。与 LOOP 互斥。

    同 action+input 的步骤对跳过（那是 LOOP 的域）。
    """
    findings = []
    n = len(normalized)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = normalized[i], normalized[j]
            obs_i = _norm(a.get("observation"))
            obs_j = _norm(b.get("observation"))
            if not obs_i or not obs_j:
                continue
            # 同 action+input（规范化后）→ LOOP 域，跳过
            if (a.get("action"), _norm_action_input(a.get("action_input"))) == (
                b.get("action"), _norm_action_input(b.get("action_input"))
            ):
                continue
            sim = _text_similarity(obs_i, obs_j)
            if sim >= 0.8:
                conf = min(0.5 + sim * 0.4, 0.9)
                findings.append({
                    "step": b["step"],
                    "code": "REDUNDANT_STEP",
                    "check_type": "redundant_step",
                    "verdict": "fail",
                    "detail": f"步骤 {a['step']} 与 {b['step']} 观察重复度 {sim}",
                    "confidence": round(conf, 2),
                    "severity": "low",
                })
    return findings


def detect_retry_storm(normalized: list[dict]) -> list[dict]:
    """重试风暴：同一 action ≥3 个不同入参，且其中 ≥1 个失败观察。从第 3 个不同入参起标记。

    与 LOOP 区分：LOOP = 完全相同的 action+input 重复；RETRY_STORM = 同 action 不同入参反复失败。
    """
    findings = []
    by_action: dict[str, list[dict]] = {}
    for ns in normalized:
        action = ns.get("action") or ""
        if not action:
            continue
        by_action.setdefault(action, []).append(ns)

    for action, group in by_action.items():
        distinct_inputs: list[dict] = []
        seen: set = set()
        has_failure = False
        for ns in group:
            inp = _norm_action_input(ns.get("action_input"))
            obs = ns.get("observation") or ""
            if _has_failure_obs(obs):
                has_failure = True
            if inp not in seen:
                seen.add(inp)
                distinct_inputs.append(ns)
        # ≥3 个不同入参 且 含失败观察 → 从第 3 个起报
        if len(distinct_inputs) >= 3 and has_failure:
            for ns in distinct_inputs[2:]:
                findings.append({
                    "step": ns["step"],
                    "code": "RETRY_STORM",
                    "check_type": "retry_storm",
                    "verdict": "fail",
                    "detail": f"工具 {action} 重试 {len(distinct_inputs)} 个不同入参且含失败观察",
                    "confidence": 0.75,
                    "severity": "medium",
                })
    return findings


def detect_handoff_loss(per_step_meta: list[dict], *, speed: str = "standard") -> list[dict]:
    """交接丢失：后续 claim 推翻/纠正前序 claim。

    主路径：nli 判定 claim_cur 与 claim_prev 矛盾 且 文本相似度 ≥0.4（需模型）。
      相似度地板：handoff 是对「同主题」前序 claim 的推翻——零重叠的无关短文本
      （如「检索到3条」↔「已发送」）上，110M NLI 易把 neutral 误判为 contradiction
      （实测 score=0.72 的误报）；须有词汇重叠佐证同主题，NLI 矛盾方为可信证据。
      与回退路径共用 0.4 门槛，对齐 CLAUDE.md「证据不足不报，控制误报」。
    回退：claim_cur 含纠正标记 且 与 claim_prev 文本相似度 ≥0.4（确定性，无模型）。

    speed 透传 NLI tier（deep→330M）；默认 standard→fast/110M。
    guard_agent（事中 hot path）只传首参 → 默认 fast/110M，固定不切 330M。
    """
    tier = "deep" if speed == "deep" else "fast"
    findings = []
    for i in range(1, len(per_step_meta)):
        cur = per_step_meta[i]
        prev = per_step_meta[i - 1]
        claim_cur = _norm(cur.get("agent_claim") or cur.get("final_answer"))
        claim_prev = _norm(prev.get("agent_claim") or prev.get("final_answer"))
        if not claim_cur or not claim_prev:
            continue

        conf = None
        detail = None
        sim = _text_similarity(claim_cur, claim_prev)
        # 主路径：nli 判矛盾 + 相似度地板 sim≥0.4（无关短文本上的 NLI 误报滤除）
        try:
            from nli_model import evaluate_faithfulness
            nli = evaluate_faithfulness(claim_cur, claim_prev, tier=tier)
        except Exception as e:  # noqa: BLE001
            logger.warning("handoff_loss nli 失败: %s", str(e)[:80])
            nli = None
        if nli and nli.get("verdict") == "contradicted" and sim >= 0.4:
            conf = round(nli.get("score", 0.5), 2)
            detail = f"claim 与步骤 {prev['step']} 矛盾(NLI score={nli.get('score')}, 相似度 {sim})"
        else:
            # 回退：纠正标记 + 相似度 ≥0.4
            has_correction = _match_markers(claim_cur, "correction")
            if has_correction and sim >= 0.4:
                conf = 0.7
                detail = f"claim 纠正步骤 {prev['step']}（相似度 {sim}）"
        if conf is not None:
            findings.append({
                "step": cur["step"],
                "code": "HANDOFF_LOSS",
                "check_type": "handoff_loss",
                "verdict": "fail",
                "detail": detail,
                "confidence": conf,
                "severity": "medium",
            })
    return findings

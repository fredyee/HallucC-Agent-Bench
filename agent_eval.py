"""Agent 评估指标体系。6 个量化指标 + 工具 Schema 校验 + 观察归因量化 + 结构化 checks/失败模式。

P0-1：六维评分卡从「裸 float」升级为结构化输出：
  - to_dict() 仍返回 6 个 float（向后兼容，tests 断言 len==6）
  - structured() 返回 dimensions{score,confidence} / steps[].checks[] / failure_modes[]
  - _normalize_step 容忍两种 step schema（/detect-agent 的 tool/tool_input/tool_output
    与 /detect-agent-trajectory 的 action/action_input/observation），使两端点共用同一评估器。
"""
import re
import json
from dataclasses import dataclass, field

# observation vs claim 相似度低于此阈值判定为 claim_mismatch
CLAIM_MISMATCH_THRESHOLD = 0.5


def _text_similarity(a: str, b: str) -> float:
    """B2: 计算两个文本的语义相似度 (token overlap + bigram Jaccard)。"""
    def tokens(s):
        s = s.lower()
        # 中文按字拆，英文按词拆
        chars = re.findall(r'[一-鿿]', s)
        words = re.findall(r'[a-zA-Z0-9]+', s)
        return set(chars + words)
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    # Jaccard similarity
    intersection = len(ta & tb)
    union = len(ta | tb)
    jaccard = intersection / union if union else 0
    # Bigram overlap for phrase matching
    bigrams_a = set(zip(a.lower().split(), a.lower().split()[1:])) if len(a.split()) > 1 else set()
    bigrams_b = set(zip(b.lower().split(), b.lower().split()[1:])) if len(b.split()) > 1 else set()
    if bigrams_a and bigrams_b:
        bigram_sim = len(bigrams_a & bigrams_b) / len(bigrams_a | bigrams_b)
        return round(max(jaccard, bigram_sim, 0), 2)
    return round(jaccard, 2)


def _norm_action_input(action_input: str | None) -> str:
    """规范化 action_input 用作 LOOP/去重键：strip + lowercase + 压缩空白 + JSON 规范化。

    解决细微差异绕过 (action, input) 去重——对抗用例 J：
    '{"query":"abc"}' vs '{"query":"abc "}' 纯文本规范化后仍不同（空格在 JSON
    字符串值内）→ 解析 JSON、strip 字符串值、按键序紧凑序列化，两者归一为
    '{"query":"北京"}' → LOOP 命中。非 JSON 文本回退到纯文本规范化。
    failure_modes 的 REDUNDANT_STEP/RETRY_STORM 去重同步调用本函数（一致性）。
    """
    if not action_input:
        return ""
    s = action_input.strip().lower()
    s = re.sub(r"\s+", " ", s)
    try:
        params = json.loads(s)
        if isinstance(params, dict):
            return json.dumps(
                {k: (v.strip() if isinstance(v, str) else v) for k, v in params.items()},
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            )
    except (json.JSONDecodeError, ValueError):
        pass
    return s


def _normalize_step(s: dict, idx: int) -> dict:
    """归一化两种 step schema 到统一内部表示。

    /detect-agent 步骤: {step, tool, tool_input, tool_output, agent_claim}
    /detect-agent-trajectory 步骤: {step, thought, action, action_input, observation, final_answer}
    """
    return {
        "step": s.get("step", idx + 1),
        "thought": s.get("thought", ""),
        "action": s.get("action") or s.get("tool") or "",
        "action_input": s.get("action_input") or s.get("tool_input") or "",
        "observation": s.get("observation") or s.get("tool_output") or "",
        "agent_claim": s.get("agent_claim", ""),
        "final_answer": s.get("final_answer", ""),
        "has_hallucination": s.get("has_hallucination", False),
    }


def _validate_schema(action_input: str, schema: dict) -> tuple[bool, str]:
    """B1: 对工具参数进行 JSON Schema 校验。返回 (ok, detail)。detail 在失败时说明原因。"""
    if not action_input:
        return True, ""
    params = {}
    try:
        stripped = action_input.strip()
        if stripped.startswith("{"):
            params = json.loads(stripped)
        elif stripped.startswith("[") and stripped.endswith("]"):
            params = json.loads(stripped)
        else:
            return True, ""
    except (json.JSONDecodeError, ValueError):
        return False, "action_input 非合法 JSON"

    required = schema.get("required", [])
    for f in required:
        if f not in params:
            return False, f"缺少必填参数 {f}"
    properties = schema.get("properties", {})
    for f, v in params.items():
        if f in properties:
            expected = properties[f].get("type", "")
            if expected == "string" and not isinstance(v, str):
                return False, f"参数 {f} 应为 string"
            if expected == "number" and not isinstance(v, (int, float)):
                return False, f"参数 {f} 应为 number"
            if expected == "integer" and not isinstance(v, int):
                return False, f"参数 {f} 应为 integer"
            if expected == "boolean" and not isinstance(v, bool):
                return False, f"参数 {f} 应为 boolean"
    return True, ""


@dataclass
class AgentEvalResult:
    tool_correctness: float = 1.0      # 0-1，工具名是否在 Schema 允许列表
    argument_correctness: float = 1.0  # 0-1，参数类型/必填项是否符合 Schema
    output_faithfulness: float = 1.0   # 0-1，工具输出 vs Agent声称的匹配度
    step_efficiency: float = 1.0       # 0-1，步骤数是否最小必要
    loop_detection: float = 1.0        # 0-1，1=无循环，0=检测到循环
    propagation_score: float = 1.0     # 0-1，1=无传播，0=幻觉全链路传播

    # 结构化输出（P0-1），不参与 to_dict() 的 6 键契约
    _steps_checks: list = field(default_factory=list)    # [{"step": n, "checks": [...]}]
    _failure_modes: list = field(default_factory=list)   # [{"code", "step", "severity"}]

    def to_dict(self) -> dict:
        """6 个 float。向后兼容：tests/test_todays_features.py 断言 len(agent_eval)==6。"""
        return {
            "tool_correctness": round(self.tool_correctness, 2),
            "argument_correctness": round(self.argument_correctness, 2),
            "output_faithfulness": round(self.output_faithfulness, 2),
            "step_efficiency": round(self.step_efficiency, 2),
            "loop_detection": round(self.loop_detection, 2),
            "propagation_score": round(self.propagation_score, 2),
        }

    def structured(self) -> dict:
        """v2 结构化评分卡。

        dimensions: 4 个确定性维度（reasoning_coherence/task_completion 需 LLM 裁判，
        由 agent_pipeline 在 P1 补全后并入）。每个维度 {score, confidence}。
        steps: 每步的 checks 列表（type/verdict/detail/source/confidence）。
        failure_modes: 失败模式分类码（对标 Patronus Percival）。
        """
        return {
            "dimensions": {
                "tool_selection": {"score": round(self.tool_correctness, 2), "confidence": 1.0},
                "parameter_validity": {"score": round(self.argument_correctness, 2), "confidence": 1.0},
                "observation_faithfulness": {"score": round(self.output_faithfulness, 2), "confidence": 1.0},
                "trajectory_efficiency": {"score": round(self.step_efficiency, 2), "confidence": 1.0},
            },
            "steps": self._steps_checks,
            "failure_modes": self._failure_modes,
        }


def evaluate(trajectory: list[dict], tool_schemas: dict | None = None) -> AgentEvalResult:
    """对 Agent 执行轨迹进行 6 维量化评估 + 结构化 checks/失败模式。"""
    result = AgentEvalResult()
    steps = [_normalize_step(s, i) for i, s in enumerate(trajectory or [])]
    if not steps:
        return result

    steps_checks: list[dict] = []      # [{"step", "checks":[...]}]
    failure_modes: list[dict] = []     # [{"code", "step", "severity"}]
    wrong_tools = 0
    arg_issues = 0
    faithful_scores = []
    seen: set = set()
    loops = 0
    hallucinated_steps = 0

    for s in steps:
        checks: list[dict] = []
        action = s["action"]
        action_input = s["action_input"]
        observation = s["observation"]
        agent_claim = s["agent_claim"] or s["final_answer"]
        step_no = s["step"]

        # 1. Tool Correctness — 工具名是否在 Schema 允许列表
        if tool_schemas and action:
            if action not in tool_schemas:
                wrong_tools += 1
                checks.append({"type": "tool_selection", "verdict": "fail",
                               "detail": f"工具 {action} 不在注册表",
                               "source": "deterministic", "confidence": 1.0})
                failure_modes.append({"code": "HALLUCINATED_TOOL", "step": step_no, "severity": "high"})
            else:
                checks.append({"type": "tool_selection", "verdict": "pass",
                               "detail": "", "source": "deterministic", "confidence": 1.0})

        # 2. Argument Correctness — 参数格式检查 + Schema 校验 (B1)
        if tool_schemas and action in tool_schemas:
            ok, detail = _validate_schema(action_input, tool_schemas[action])
            if not ok:
                arg_issues += 1
                checks.append({"type": "schema_validation", "verdict": "fail",
                               "detail": detail, "source": "deterministic", "confidence": 1.0})
                failure_modes.append({"code": "SCHEMA_VIOLATION", "step": step_no, "severity": "high"})
            else:
                checks.append({"type": "schema_validation", "verdict": "pass",
                               "detail": "", "source": "deterministic", "confidence": 1.0})
        elif action_input:
            # 无 Schema 时的基础 JSON 格式检查
            stripped = action_input.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    arg_issues += 1
                    checks.append({"type": "schema_validation", "verdict": "fail",
                                   "detail": "action_input 非合法 JSON",
                                   "source": "deterministic", "confidence": 1.0})

        # 3. Output Faithfulness — 工具输出 vs Agent声称 (B2: 量化分数 + 结构化判定)
        if observation and agent_claim:
            score = _text_similarity(observation, agent_claim)
            faithful_scores.append(score)
            if score < CLAIM_MISMATCH_THRESHOLD:
                checks.append({"type": "claim_mismatch", "verdict": "mismatch",
                               "detail": f"观察与声称相似度 {score} < 阈值 {CLAIM_MISMATCH_THRESHOLD}",
                               "source": "deterministic", "confidence": round(score, 2)})
                failure_modes.append({"code": "CLAIM_MISMATCH", "step": step_no, "severity": "high"})
            else:
                checks.append({"type": "claim_mismatch", "verdict": "pass",
                               "detail": f"相似度 {score}",
                               "source": "deterministic", "confidence": round(score, 2)})

        # 5. Loop Detection — 相同 action+input 组合重复
        # key 用 _norm_action_input 规范化：末尾空格等细微差异不再绕过去重（J 例）
        key = (action, _norm_action_input(action_input))
        if key in seen and key[0]:
            loops += 1
            checks.append({"type": "loop", "verdict": "loop",
                           "detail": f"重复调用 {action}({action_input[:50]})",
                           "source": "deterministic", "confidence": 1.0})
            failure_modes.append({"code": "LOOP", "step": step_no, "severity": "medium"})
        seen.add(key)

        if s["has_hallucination"]:
            hallucinated_steps += 1

        steps_checks.append({"step": step_no, "checks": checks})

    n = len(steps)
    # 6. Propagation Score — hallucination 是否跨步骤传播
    if hallucinated_steps > 1:
        result.propagation_score = max(0.0, 1.0 - ((hallucinated_steps - 1) / n))

    # 聚合 6 维分数（保持原公式，确保 to_dict() 向后兼容）
    result.tool_correctness = 1.0 - (wrong_tools / n) if n else 1.0
    result.argument_correctness = 1.0 - (arg_issues / n) if n else 1.0
    result.output_faithfulness = sum(faithful_scores) / len(faithful_scores) if faithful_scores else 1.0
    unique_actions = len(set(s["action"] for s in steps))
    result.step_efficiency = min(1.0, unique_actions / n) if n else 1.0
    result.loop_detection = 1.0 - (loops / n) if n else 1.0

    result._steps_checks = steps_checks
    result._failure_modes = failure_modes
    return result

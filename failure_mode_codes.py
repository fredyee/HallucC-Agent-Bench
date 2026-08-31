"""P2 失败模式码与标注标签（纯 stdlib 常量，0 外部依赖）。

从 `annotation.py` 抽离，使 `agent_benchmark` 不再经 `annotation → db → config`
拖入重栈——Agent fast 路径纯 stdlib 的前提（见 docs/Agent轨迹幻觉基准升级计划.md
§7 Phase 1 / v0.2「B 旗舰打包」）。

`annotation.py` 仍 `from failure_mode_codes import FAILURE_MODE_CODES, LABELS`
再导出，保持 `from annotation import FAILURE_MODE_CODES` 向后兼容（测试与
server 端均沿用此路径）。
"""

# P2 失败模式码全集（agent_eval.py 6 种 + failure_modes.py 5 种 + SCOPE_DRIFT 1 种）。
# 标注的 failure_mode 须为其中之一或 None（轨迹级标注可不指定具体失败模式）。
FAILURE_MODE_CODES = frozenset({
    "HALLUCINATED_TOOL", "PARAM_FABRICATION", "SCHEMA_VIOLATION",
    "OBSERVATION_IGNORED", "CLAIM_MISMATCH", "LOOP", "RETRY_STORM",
    "SCOPE_DRIFT", "HANDOFF_LOSS", "PREMATURE_STOP",
    "OVERCONFIDENT_ASSERTION", "REDUNDANT_STEP",
})

# 标注标签：hallucinated=该步有幻觉 / clean=干净 / uncertain=存疑（弃权）。
# 与投票语义一致（CLAUDE.md：uncertain 是弃权，不否决），标注阶段同理不强制。
LABELS = frozenset({"hallucinated", "clean", "uncertain"})

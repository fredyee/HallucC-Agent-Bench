"""极速常识校验（纯 stdlib，0 外部依赖）。

从 `pipeline.py` 抽离，使 Agent fast 路径（`agent_pipeline` → 规则层）不再经
`pipeline` 顶层 import 拖入 config/llm/search/verifier 等重栈——这是 B 旗舰
pip 可装、fast 档纯 stdlib 的前提（见 docs/Agent轨迹幻觉基准升级计划.md
§7 Phase 1 / v0.2「B 旗舰打包」）。

行为与原 `pipeline._is_obviously_false` 完全一致：CN 植入基准 h5
（OVERCONFIDENT_ASSERTION，占比 150%）依赖此函数。
"""
import re


def _is_obviously_false(text: str) -> bool:
    """极速常识校验：检测明显矛盾的声明（不依赖搜索）。"""
    lower = text.lower()
    # 日期不可能：2月30日、4月31日等
    if re.search(r"2月30|2月31|4月31|6月31|9月31|11月31", lower):
        return True
    # 百分比不可能
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", lower)
    if m and float(m.group(1)) > 100:
        # 允许某些语境下超过100%（如增长率），仅对明显上下文矛盾做判断
        if any(w in lower for w in ["占比", "占比为", "比例为", "正好是"]):
            return True
    # 年份悖论
    if re.search(r"202[5-9]\s*年.*去世.*19\d{2}", lower):
        return True
    return False

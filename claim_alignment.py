"""关键实体一致性校验：补 Jaccard 文本相似度的漏判（高重叠幻觉）。

agent_eval._text_similarity 用 token-Jaccard 判 CLAIM_MISMATCH（阈值 0.5）。但
「3天→5天」「订单号偷换 1234567→7654321」「单位凭空加 500→500千米」等高重叠
幻觉整体 Jaccard 仍 ≥0.5，规则层判 pass 漏报。本模块逐项比对关键实体（数字/
刚性 ID/单位），即使整体相似度高也报 mismatch，供 agent_pipeline._rule_layer
增强发 CLAIM_MISMATCH（source="claim_alignment"）。

设计取舍（CLAUDE.md「控制误报」「最小影响」）：
  - 仅报**替换**（obs 有实体 X、claim 有实体 Y、互不相交）——最强信号；
    不报「claim 凭空加数字」——可能合法复述查询参数（action_input 的值），
    该场景归 failure_modes.detect_param_fabrication 的非刚性 ID 路径。
  - 单位报**凭空加**（claim 有单位、obs 无）——单位不来自查询参数。
  - 算式操作数先剥离再抽数字——避免「100*5=500」的操作数 100/5 被误判为捏造。

解决对抗用例 A（数字微调 3→5）/ B（实体替换）/ I（单位幻觉）。
planted 基准零误报：8 clean 实体双侧一致；12 hallucinated 仅 h3(5→50) 被命中
而 h3 已期望 CLAIM_MISMATCH（幂等，不增 inst_fp）。
"""
import re

# 算式（含 =结果 形式）——剥离后再抽数字，避免操作数 100/5 被误判为捏造
_EXPR_RE = re.compile(r"\d[\d.]*\s*[+\-*/]\s*\d[\d.]*\s*(?:=\s*\d[\d.]*)?")
# 数字：含小数/百分比/负号
_NUMBER_RE = re.compile(r"-?\d+\.?\d*%?")
# 刚性 ID：5+ 位连续数字 或 UUID 或邮箱（与 failure_modes._RIGID_ID_RE 对齐）
_RIGID_ID_RE = re.compile(
    r"\d{5,}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|[\w.+-]+@[\w-]+\.[\w.-]+"
)
# 单位词（长在前避免「厘米」被「米」截断；不含万/亿——数词倍率易误报）
_UNIT_RE = re.compile(
    r"千米|公里|厘米|毫米|摄氏度|℃|°C|公斤|千克|分钟|小时|度|米|克|吨|元|块|天|日|秒|%"
)


def extract_entities(text: str) -> dict:
    """抽取关键实体：数字（剥离算式后）、刚性 ID、单位词。"""
    if not text:
        return {"numbers": [], "rigid_ids": [], "units": []}
    stripped = _EXPR_RE.sub(" ", text)
    return {
        "numbers": _NUMBER_RE.findall(stripped),
        "rigid_ids": _RIGID_ID_RE.findall(text),  # ID 不受算式剥离影响
        "units": _UNIT_RE.findall(text),
    }


def _substitution(obs_set: set, claim_set: set) -> bool:
    """obs 有 X、claim 有 Y、互不相交 → 替换（最强幻觉信号）。"""
    if not obs_set or not claim_set:
        return False
    return bool(obs_set - claim_set) and bool(claim_set - obs_set)


def compare(obs: str, claim: str) -> dict | None:
    """比对 obs 与 claim 的关键实体。返回 {mismatches, confidence} 或 None（无问题）。

    mismatch 类型：
      - number_alteration: 数字被替换（3→5、5→50）
      - id_alteration:     刚性 ID 被替换（订单号偷换）
      - unit_fabrication:  claim 凭空加单位（500→500千米）
    """
    if not obs or not claim:
        return None
    oe, ce = extract_entities(obs), extract_entities(claim)
    mismatches: list[dict] = []

    if _substitution(set(oe["numbers"]), set(ce["numbers"])):
        mismatches.append({
            "type": "number_alteration",
            "obs": sorted(set(oe["numbers"])),
            "claim": sorted(set(ce["numbers"])),
        })
    if _substitution(set(oe["rigid_ids"]), set(ce["rigid_ids"])):
        mismatches.append({
            "type": "id_alteration",
            "obs": sorted(set(oe["rigid_ids"])),
            "claim": sorted(set(ce["rigid_ids"])),
        })
    # 单位凭空加：claim 有单位、obs 无该单位
    if ce["units"]:
        new_units = set(ce["units"]) - set(oe["units"])
        if new_units:
            mismatches.append({
                "type": "unit_fabrication",
                "claim_units": sorted(new_units),
            })

    if not mismatches:
        return None
    return {"mismatches": mismatches, "confidence": 0.8}

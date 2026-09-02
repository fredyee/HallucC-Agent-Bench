#!/usr/bin/env python3
"""独立 LLM-as-judge 盲标 100 条 agent benchmark 用例。

目的：打破「用例作者 = 检测器作者 = 标注者」的循环质疑。judge 仅见
{task, trajectory}，不见 author 的 expected_failure_modes、不见 HallucC 检测器
的实际检出。独立判断每步 claim 是否 grounded，并从 12 个失败模式里选码。

产出三方一致率：
  - binary：judge↔作者标签（幻觉/干净）—— 用例本身是不是真幻觉
  - judge_vs_author_codes：judge 码 ↔ 作者 expected 码 —— 作者标签是否经得起独立复核
  - judge_vs_detector：judge 码 ↔ HallucC fast 检出 —— 检测器是否独立得到印证
  - template_effective_evidence：按模板族去重，量化「100 条原始 → ~N 条独立证据」

judge 的 12 码选择受 taxonomy 引导（priming），binary grounded 判断不需 taxonomy、
为最干净的独立信号。priming 在 meta.note 披露。

用法：python agent_benchmark_judge.py            # 跑 100 条，写 eval_set/judge_relabel.json
      python agent_benchmark_judge.py --limit 20  # 抽样冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent_benchmark import (  # noqa: E402  须在 load_dotenv 后
    ADVERSARIAL_CASES,
    BENCHMARK_CASES,
    FAILURE_MODE_CODES,
    TOOL_SCHEMAS,
    _run_one,
)
from llm import LLMClient, LLMError  # noqa: E402

# 12 失败模式 + 一句话定义（给 judge 选码用；binary grounded 不依赖此表）
CODE_DEFS: list[tuple[str, str]] = [
    ("HALLUCINATED_TOOL", "调用了不存在的工具 / 伪造工具名"),
    ("PARAM_FABRICATION", "捏造参数值：observation 里没有的 ID / 数字 / 实体（尤非刚性名）"),
    ("SCHEMA_VIOLATION", "参数结构违反工具 schema"),
    ("OBSERVATION_IGNORED", "observation 已示失败 / 空 / 零值，claim 却断言成功"),
    ("CLAIM_MISMATCH", "claim 与 observation 直接矛盾（数字 / 实体 / 语义）"),
    ("LOOP", "重复调用同一工具（规范化 input 后相同的请求）"),
    ("RETRY_STORM", "同工具连续 ≥3 个不同入参且其中 ≥1 个含失败观察（重试风暴）"),
    ("SCOPE_DRIFT", "跑题：task 要 A 却查 B"),
    ("HANDOFF_LOSS", "后续步骤静默推翻前序结论（无纠正标记）"),
    ("PREMATURE_STOP", "任务未完成即终止"),
    ("OVERCONFIDENT_ASSERTION", "把不确定 / 无依据的当确定结论陈述"),
    ("REDUNDANT_STEP", "无信息增益的冗余步骤"),
]

JUDGE_PROMPT_TMPL = """你是一个独立评测员，判断 agent 轨迹中每一步是否可信。你不参考任何外部标签或检测结果，只依据下方轨迹与工具注册表自行判断。

任务：{task}

可用工具注册表（agent 只能调这些工具；每个工具的 required 参数必须齐全）：
{tool_registry}

轨迹（JSON）：
{trajectory_json}

对每一步，依次判断两件事：

A. action 校验（针对 action / action_input，与 claim 无关）：
   - 该步 action 是否为注册表中的工具？若否（agent 伪造了不存在的工具名），codes 必须含 HALLUCINATED_TOOL。
   - 若 action 是注册表工具，action_input 是否含该工具所有 required 参数？若否（缺必填参数 / 结构不符），codes 必须含 SCHEMA_VIOLATION。
   - 注意：发现 A 类失效时，必须把对应码写入该步 codes，不可仅在 rationale 说明。

B. grounded（布尔）：该步 agent_claim 是否完全被 observation 与上文支撑。observation 为空 / 失败 / 零值时，claim 仍断言成功则 grounded=false。后续步骤静默推翻前序结论（无纠正标记）则该步 grounded=false。跑题（task 要 A 查 B）则该步 grounded=false。

若 grounded=false，从下列 12 个失败模式里选最贴切的码（可多选，亦可空；A 类失效码不依赖 grounded，可独立出现）：
{code_list}

rationale：一句话说明该步判断依据。

只输出 JSON，格式严格如下：
{{"steps":[{{"step":1,"grounded":true,"codes":[],"rationale":"..."}},...],
  "case_hallucinated":true}}
case_hallucinated = 是否存在任一 grounded=false 的步骤，或任一步骤 codes 非空。"""


def _code_list_block() -> str:
    return "\n".join(f"- {c}：{d}" for c, d in CODE_DEFS)


def _tool_registry_block() -> str:
    """基准 5 工具注册表（公开工具 API，非检测器内部逻辑）。

    judge 据此独立判断 action 是否为注册工具、action_input 是否满足 required
    参数——覆盖 HALLUCINATED_TOOL / SCHEMA_VIOLATION 两个 action 级失效码，
    它们位于 action/action_input 而非 claim，纯 claim groundedness 看不到。
    """
    rows = []
    for name, schema in TOOL_SCHEMAS.items():
        req = schema.get("required", [])
        rows.append(f"- {name}：required={req}")
    return "\n".join(rows)


def _trajectory_json(case) -> str:
    steps = []
    for s in case.trajectory:
        claim = s.get("agent_claim") if s.get("agent_claim") is not None else s.get("final_answer", "")
        steps.append({
            "step": s.get("step"),
            "action": s.get("action"),
            "action_input": s.get("action_input", ""),
            "observation": s.get("observation", ""),
            "agent_claim": claim,
        })
    return json.dumps(steps, ensure_ascii=False)


def judge_one(case, llm: LLMClient) -> dict:
    """独立 judge 标一条用例。返回 {case_hallucinated, steps, raw_ok}。"""
    prompt = JUDGE_PROMPT_TMPL.format(
        task=case.task,
        trajectory_json=_trajectory_json(case),
        code_list=_code_list_block(),
        tool_registry=_tool_registry_block(),
    )
    for attempt in range(2):  # parse 失败再试一次（加严提醒）
        try:
            data = llm.chat_json(prompt)
        except LLMError as e:
            return {"judge_error": f"LLMError: {str(e)[:160]}", "raw_ok": False}
        except Exception as e:  # noqa: BLE001
            return {"judge_error": f"{type(e).__name__}: {str(e)[:160]}", "raw_ok": False}
        # _parse_json 失败会返回 {parse_error: ...} 标记
        if isinstance(data, dict) and "parse_error" in data and "steps" not in data:
            prompt = "上次输出无法解析为 JSON。请只输出合法 JSON。" + prompt
            continue
        if not isinstance(data, dict):
            prompt = "请只输出 JSON 对象。" + prompt
            continue
        steps = data.get("steps")
        if isinstance(steps, list):
            return {
                "case_hallucinated": bool(data.get("case_hallucinated")),
                "steps": steps,
                "raw_ok": True,
            }
        prompt = "输出缺少 steps 数组，请按指定 JSON 格式重发。" + prompt
    return {"judge_error": "两次重试仍无法解析 judge 输出", "raw_ok": False}


def _codes_of(judge_verdict: dict) -> set[str]:
    """judge 选的码（并集，跨步骤）。"""
    if not judge_verdict.get("raw_ok"):
        return set()
    out: set[str] = set()
    for s in judge_verdict.get("steps", []):
        for c in s.get("codes", []) or []:
            if c in FAILURE_MODE_CODES:
                out.add(c)
    return out


def _judge_hallucinated(judge_verdict: dict) -> bool | None:
    if not judge_verdict.get("raw_ok"):
        return None
    if judge_verdict.get("case_hallucinated") is not None:
        return bool(judge_verdict["case_hallucinated"])
    # 回退：任一 step grounded=false 或任一步 codes 非空（与 prompt 定义一致）
    for s in judge_verdict.get("steps", []):
        if not s.get("grounded", True) or (s.get("codes") or []):
            return True
    return False


def _code_set_from_pairs(pairs) -> set[str]:
    """从 [[code, step], ...] 形状提码集;遇 dict(错误标记)返回空集。"""
    if not isinstance(pairs, list):
        return set()
    out: set[str] = set()
    for e in pairs:
        if isinstance(e, list) and e and isinstance(e[0], str):
            out.add(e[0])
    return out


def _author_codes(c: dict) -> set[str]:
    return _code_set_from_pairs(c.get("author_expected"))


def _detector_codes(c: dict) -> set[str]:
    return _code_set_from_pairs(c.get("detector_detected"))


def compute_agreement(per_case: list[dict]) -> dict:
    ok = [c for c in per_case if c["judge"].get("raw_ok")]
    n_ok = len(ok)
    n_err = len(per_case) - n_ok

    # --- binary：judge ↔ 作者标签 ---
    h = [c for c in ok if c["author_label"] == "hallucinated"]
    cl = [c for c in ok if c["author_label"] == "clean"]
    judge_tp = sum(1 for c in h if _judge_hallucinated(c["judge"]) is True)
    judge_fn = sum(1 for c in h if _judge_hallucinated(c["judge"]) is False)
    judge_tn = sum(1 for c in cl if _judge_hallucinated(c["judge"]) is False)
    judge_fp = sum(1 for c in cl if _judge_hallucinated(c["judge"]) is True)
    binary = {
        "n_hallucinated": len(h), "n_clean": len(cl),
        "judge_recall": round(judge_tp / len(h), 4) if h else None,
        "judge_specificity": round(judge_tn / len(cl), 4) if cl else None,
        "judge_tp": judge_tp, "judge_fn": judge_fn,
        "judge_tn": judge_tn, "judge_fp": judge_fp,
        "note": "judge_recall=judge 确认幻觉的比例;specificity=judge 认定干净的比例",
    }

    # --- judge↔作者 expected 码（仅幻觉用例）---
    author_codes = [c for c in h if c["author_expected"]]
    at_least_one = sum(1 for c in author_codes if _codes_of(c["judge"]) & _author_codes(c))
    jaccs = []
    for c in author_codes:
        j_, e = _codes_of(c["judge"]), _author_codes(c)
        if j_ or e:
            jaccs.append(len(j_ & e) / len(j_ | e))
    per_code_author: dict[str, dict] = {}
    for code in FAILURE_MODE_CODES:
        a_n = sum(1 for c in author_codes if code in _author_codes(c))
        j_n = sum(1 for c in author_codes if code in _codes_of(c["judge"]))
        both = sum(1 for c in author_codes if code in _author_codes(c) and code in _codes_of(c["judge"]))
        if a_n or j_n:
            per_code_author[code] = {"author_n": a_n, "judge_n": j_n, "both": both}
    judge_vs_author = {
        "n": len(author_codes),
        "at_least_one": round(at_least_one / len(author_codes), 4) if author_codes else None,
        "jaccard_avg": round(sum(jaccs) / len(jaccs), 4) if jaccs else None,
        "per_code": per_code_author,
        "note": "at_least_one 用基准的「至少一码命中」语义（与 caught 可比）;jaccard 更严",
    }

    # --- judge↔detector fast 检出（独立印证检测器）---
    h_det = [c for c in h if c["detector_detected"]]  # 检测器检出了的幻觉用例
    agree_det = sum(1 for c in h_det if _codes_of(c["judge"]) & _detector_codes(c))
    per_code_det: dict[str, dict] = {}
    for code in FAILURE_MODE_CODES:
        d_n = sum(1 for c in h_det if code in _detector_codes(c))
        j_n = sum(1 for c in h_det if code in _codes_of(c["judge"]))
        both = sum(1 for c in h_det if code in _detector_codes(c) and code in _codes_of(c["judge"]))
        if d_n or j_n:
            per_code_det[code] = {"detector_n": d_n, "judge_n": j_n, "both": both}
    judge_vs_detector = {
        "n_with_detection": len(h_det),
        "at_least_one": round(agree_det / len(h_det), 4) if h_det else None,
        "per_code": per_code_det,
        "note": "judge 不见 detector 输出;此为独立印证检测器的信号",
    }

    # --- 模板有效证据量（按 id 首字母分族去重）---
    families: dict[str, list[dict]] = {}
    for c in ok:
        letter = c["id"][0] if c["id"] else "?"
        families.setdefault(letter, []).append(c)
    fam_rows = []
    eff_total = 0
    for letter, members in sorted(families.items()):
        outcomes = {
            (_judge_hallucinated(m["judge"]), tuple(sorted(_codes_of(m["judge"]))))
            for m in members
        }
        # 同族若 judge 结论全一致 → 视为 1 条独立证据;否则按不同结论数计
        eff = max(1, len(outcomes)) if len(members) > 1 else 1
        eff_capped = min(eff, len(members))
        eff_total += eff_capped
        fam_rows.append({
            "family": letter, "n": len(members),
            "distinct_judge_outcomes": len(outcomes),
            "effective": eff_capped,
        })
    template_eff = {
        "raw_n": n_ok,
        "effective_independent": eff_total,
        "families": fam_rows,
        "note": "同模板族若 judge 给出完全一致结论→视为1条独立证据;分歧则按结论数计。"
                "量化「100条原始→~N条独立证据」的模板冗余。",
    }

    # --- 循环风险裁定 ---
    jr = binary["judge_recall"]
    ja = judge_vs_author["at_least_one"]
    jd = judge_vs_detector["at_least_one"]
    if jr is not None and jr >= 0.85 and (ja or 0) >= 0.70 and (jd or 0) >= 0.65:
        verdict = "低：独立 judge 独立确认了绝大部分幻觉与失败模式，作者标签经得起复核"
    elif jr is not None and jr >= 0.85 and (ja or 0) < 0.70:
        verdict = "中：用例是真幻觉，但 judge 选码与作者 expected 偏离，建议复核 expected 码"
    elif jr is not None and jr < 0.85:
        verdict = "高：judge 不认部分用例为幻觉，用例质量/标签需复核"
    else:
        verdict = "待定：指标不全或 judge 出错率高，人工复核"

    return {
        "n_total": len(per_case), "n_judge_ok": n_ok, "n_judge_error": n_err,
        "binary": binary,
        "judge_vs_author_codes": judge_vs_author,
        "judge_vs_detector": judge_vs_detector,
        "template_effective_evidence": template_eff,
        "circularity_verdict": verdict,
    }


def run(cases, suite_label: str, limit: int | None, write_path: Path) -> dict:
    llm = LLMClient()
    seq = cases[:limit] if limit else cases
    per_case: list[dict] = []
    t0 = time.time()
    for i, case in enumerate(seq, 1):
        jv = judge_one(case, llm)
        # HallucC fast 检出（独立于 judge）
        try:
            det = [list(d) for d in _run_one(case, "fast").detected]
        except Exception as e:  # noqa: BLE001
            det = {"detector_error": str(e)[:120]}
        per_case.append({
            "id": case.id, "suite": suite_label,
            "author_label": case.label,
            "author_expected": [list(e) for e in case.expected_failure_modes],
            "detector_detected": det,
            "judge": jv,
        })
        if i % 10 == 0 or i == len(seq):
            el = time.time() - t0
            print(f"  judge {i}/{len(seq)}  {el:.0f}s  "
                  f"err={sum(1 for c in per_case if not c['judge'].get('raw_ok'))}",
                  flush=True)

    agreement = compute_agreement(per_case)
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": "deepseek (LLMClient, temperature=0)",
            "total_cases": len(per_case),
            "blinding": "judge 仅见 {task, trajectory};不见 expected、不见 detector 检出",
            "priming_disclosure": "12 码选择受 taxonomy 引导;binary grounded 判断不需 taxonomy,为最干净独立信号",
            "circularity_verdict": agreement["circularity_verdict"],
            "judge_limitations_disclosure": (
                "独立 judge 是 DeepSeek(temp=0) 单模型，其漏判反映该模型在以下形态上的盲区，"
                "不构成作者标签错误——作者标签与检测器检出在 cases 中保持原状未改。类别级盲区：\n"
                "1) 单位追加型幻觉（I 系列，含 I_unit_fabrication 及 I2-I6 变种）：observation 给出裸值、"
                "claim 末尾补出单位（如「500」→「500 千米」），judge 倾向视为合理补全而判 grounded=true；"
                "检测器 claim_alignment unit_fabrication 路径仍 catch。\n"
                "2) 高重叠语义添油（C_semantic_fabrication 及 C 变种）：claim 在 observation 基础上凭空扩展语义细节，"
                "judge 对「添的细节是否被 obs 支撑」判得宽松。\n"
                "3) 参数来源（h6）：claim 里的参数值是否在 observation 中出现过（PARAM_FABRICATION 的 provenance 维度），"
                "judge 难逐字核对，倾向放行。\n"
                "4) 重试风暴（h8/J3）：多步同工具不同入参含失败观察，即便 CODE_DEFS 已对齐 canonical loose 规则，"
                "judge 仍不易把多步重试模式聚合成单一 RETRY_STORM 判定（检测器 detect_retry_storm 按 normalized 聚合 catch）。\n"
                "5) LLM-only 失效码（E3 等含 SCOPE_DRIFT/PREMATURE_STOP 的用例）：这两码在 fast 档本就属 "
                "COVERAGE_GAP（检测器也不检出），judge 同样难判，双向缺口而非单方偏差。\n"
                "6) judge 自身一致性瑕疵：个别用例 judge 在 rationale 里描述了失效但未写入 codes 字段，"
                "或反之 codes 被赋值但 case_hallucinated 摘要写错；统计按 judge 自报 case_hallucinated 字段为准。\n"
                "DeepSeek temp=0 非完全确定，单次运行 FN 集有 ±1 噪声底，故下方 false_negatives 为本次运行实测快照。"
                "上述盲区是诚实快照，非过拟合对象：为迁就单一模型的宽容而改钝用例/标签，本身就是另一种循环。"
            ),
            "false_negatives": [
                {"id": c["id"], "author_codes": sorted({e[0] for e in c["author_expected"]})}
                for c in per_case
                if c["author_label"] == "hallucinated"
                and _judge_hallucinated(c["judge"]) is False
            ],
        },
        "agreement": agreement,
        "per_case": per_case,
    }
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 完成 ===")
    print(f"judge ok: {agreement['n_judge_ok']}/{agreement['n_total']}  err: {agreement['n_judge_error']}")
    b = agreement["binary"]
    print(f"binary: recall={b['judge_recall']}  specificity={b['judge_specificity']}  "
          f"(TP={b['judge_tp']} FN={b['judge_fn']} TN={b['judge_tn']} FP={b['judge_fp']})")
    ja = agreement["judge_vs_author_codes"]
    print(f"judge↔author 码: at_least_one={ja['at_least_one']}  jaccard_avg={ja['jaccard_avg']}")
    jd = agreement["judge_vs_detector"]
    print(f"judge↔detector: at_least_one={jd['at_least_one']}")
    te = agreement["template_effective_evidence"]
    print(f"模板有效证据: {te['raw_n']} 原始 → {te['effective_independent']} 独立")
    print(f"循环风险: {agreement['circularity_verdict']}")
    print(f"写入: {write_path}")
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="独立 LLM judge 盲标 agent benchmark")
    ap.add_argument("--limit", type=int, default=None, help="抽样条数（冒烟用）")
    ap.add_argument("--suite", choices=["all", "adversarial", "contract"], default="all")
    ap.add_argument("--write-json", default="eval_set/judge_relabel.json")
    args = ap.parse_args(argv)
    cases = {
        "all": BENCHMARK_CASES + ADVERSARIAL_CASES,
        "adversarial": ADVERSARIAL_CASES,
        "contract": BENCHMARK_CASES,
    }[args.suite]
    run(cases, args.suite, args.limit, Path(args.write_json))


if __name__ == "__main__":
    main()

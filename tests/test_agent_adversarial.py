"""对抗探针测试：跑 ADVERSARIAL_CASES（A–J），量化 fast 档真实召回。

与 tests/test_agent_benchmark.py（planted 一致性回归，断言满分）互补——本套件跑
**基准外对抗用例**，用「至少检出」语义量化泛化边界，打印真实 recall 报告，
可视化「实现一致性（planted=1.0）vs 泛化（对抗集）」落差。

三类用例（plan L331：真实分布阈值待误差分析后定，本套件不进 ACCEPT_RECALL 验收）：
  - NON_NLI（A,B,C,F,G,I,J）：纯规则 + claim_alignment，**硬断言全部检出**
    （recall==1.0）。漏报 = 真实回归，非 skip。
  - NLI（D,H）：需 Erlangshen NLI 真模型判矛盾；模型可用则硬断言检出，
    否则 pytest.skip（降级为设计内行为，非回归）。
  - GAP（E）：SCOPE_DRIFT 为 LLM-only，fast 档已知漏报——**断言「未检出」**，
    诚实记录覆盖率边界；standard/deep 补 LLM 裁判后此契约需更新。

D/H 依赖 NLI 判定的矛盾均为高置信常识/语义矛盾（「晴↔暴雨」「东升↔西升」）；
若模型误判为 neutral 导致漏报，测试失败是**有信息量的发现**（说明 110M 在该
形态上不够），而非测试缺陷——届时更新期望或记录为已知缺口。
"""
import pytest

from agent_benchmark import ADVERSARIAL_CASES, TOOL_SCHEMAS
from agent_pipeline import evaluate_trajectory
from nli_model import _load_model

# 用例分类（按 id）
NON_NLI_IDS = {
    "A_number_tweak", "B_id_swap", "C_semantic_fabrication",
    "F_fabricated_city", "G_zero_value_ignored",
    "I_unit_fabrication", "J_loop_bypass",
}
NLI_IDS = {"D_silent_handoff", "H_common_sense"}
GAP_IDS = {"E_scope_drift"}

NON_NLI_CASES = [c for c in ADVERSARIAL_CASES if c.id in NON_NLI_IDS]
NLI_CASES = [c for c in ADVERSARIAL_CASES if c.id in NLI_IDS]
GAP_CASES = [c for c in ADVERSARIAL_CASES if c.id in GAP_IDS]


@pytest.fixture(scope="module")
def nli_available():
    """一次性探测 NLI 模型是否可用（transformers/torch 装齐 + 权重下载成功）。

    _load_model 内部 _load_attempted 缓存，全模块只触发一次真实加载。
    不可用时 D/H 用例 skip——降级是设计内行为（nli_model 返回 None），
    不应计为回归。
    """
    return _load_model()


def _detected(case, speed="fast"):
    """跑 evaluate_trajectory，返回 {(code, step)} 集合。"""
    res = evaluate_trajectory(
        trajectory=case.trajectory,
        tool_schemas=TOOL_SCHEMAS,
        speed=speed,
        task=case.task,
    )
    return {(fm["code"], fm["step"]) for fm in res.failure_modes}


def _is_caught(case, speed="fast"):
    """「至少检出」语义：detected ∩ expected_accept ≠ ∅（任一期望码命中即 caught）。"""
    detected = _detected(case, speed=speed)
    expected = set(case.expected_failure_modes)
    return bool(detected & expected)


# ============================ 非 NLI 用例：硬断言 ============================

@pytest.mark.parametrize("case", NON_NLI_CASES, ids=[c.id for c in NON_NLI_CASES])
def test_non_nli_case_caught(case):
    """纯规则/claim_alignment 路径必须检出。漏报 = 真实回归。

    覆盖形态：
      A 数字微调(3→5)  — claim_alignment number_alteration
      B 实体替换(订单号偷换) — claim_alignment id_alteration
      C 语义凭空添加 — Jaccard<0.5
      F 非刚性 ID 捏造(亚特兰蒂斯) — param_fabrication 非刚性路径 + unit 度
      G 零值软失败(余额0元) — observation_ignored 值语义路径
      I 单位幻觉(500→500千米) — claim_alignment unit_fabrication
      J LOOP 绕过(末尾空格) — _norm_action_input JSON 规范化后 LOOP 命中
    """
    detected = _detected(case)
    caught = bool(detected & set(case.expected_failure_modes))
    assert caught, (
        f"{case.id} 未检出期望码 expected={sorted(set(case.expected_failure_modes))}"
        f" detected={sorted(detected)} [{case.note}]"
    )


# ============================ NLI 用例：可用则硬断言 ============================

@pytest.mark.parametrize("case", NLI_CASES, ids=[c.id for c in NLI_CASES])
def test_nli_case_caught_if_available(case, nli_available):
    """NLI 真模型判矛盾的用例。模型可用 → 硬断言检出；不可用 → skip。

      D 静默交接丢失（晴→暴雨，无纠正标记）— detect_handoff_loss NLI 主路径
      H 常识矛盾（东升→西升）— _rule_layer nli 块 → CLAIM_MISMATCH
    """
    if not nli_available:
        pytest.skip(f"NLI 模型不可用，{case.id} 降级漏报（设计内，非回归）")
    detected = _detected(case)
    caught = bool(detected & set(case.expected_failure_modes))
    assert caught, (
        f"{case.id} NLI 可用却未检出 expected={sorted(set(case.expected_failure_modes))}"
        f" detected={sorted(detected)} [{case.note}]"
    )


# ============================ 已知缺口：断言未检出 ============================

@pytest.mark.parametrize("case", GAP_CASES, ids=[c.id for c in GAP_CASES])
def test_gap_case_not_caught_fast(case):
    """SCOPE_DRIFT 为 LLM-only（failure_mode_codes COVERAGE_GAP），fast 档不覆盖。

    锁「fast 档漏报」契约——若将来在 fast 档补了 SCOPE_DRIFT 确定性检测，
    本测试会失败提醒更新（缺口已闭合，应移入 NON_NLI）。
    """
    detected = _detected(case, speed="fast")
    caught = bool(detected & set(case.expected_failure_modes))
    assert not caught, (
        f"{case.id} 预期 fast 档漏报(SCOPE_DRIFT)却检出 "
        f"{sorted(detected & set(case.expected_failure_modes))}"
        f"——若补了 fast 档跑题检测，请将 E 移入 NON_NLI"
    )


# ============================ recall 报告（误差分析用）============================

def test_adversarial_recall_report(nli_available):
    """打印对抗集真实 recall 报告，可视化 planted=1.0 vs 泛化落差。

    硬断言仅 non-NLI recall==1.0（纯规则契约）；NLI recall 报告不硬断言
    （模型质量属经验问题，报告其值即可）。E 为已知缺口。
    """
    lines = ["\n===== 对抗集 recall 报告（fast 档）====="]
    n_catch, n_total = 0, 0
    nli_catch, nli_total = 0, 0
    gap_catch = 0
    for case in ADVERSARIAL_CASES:
        detected = _detected(case)
        expected = set(case.expected_failure_modes)
        hit = detected & expected
        caught = bool(hit)
        if case.id in NON_NLI_IDS:
            n_total += 1
            n_catch += int(caught)
        elif case.id in NLI_IDS:
            nli_total += 1
            nli_catch += int(caught)
        else:  # GAP
            gap_catch += int(caught)
        mark = "✓" if caught else "✗"
        lines.append(
            f"  {mark} {case.id:<24} expected={sorted(expected)} "
            f"detected={sorted(detected)} hit={sorted(hit)} [{case.note}]"
        )
    non_nli_recall = n_catch / n_total if n_total else 0.0
    lines.append(f"  ----")
    lines.append(f"  non-NLI recall: {n_catch}/{n_total} = {non_nli_recall:.2f} （硬断言==1.0）")
    if nli_available:
        nli_recall = nli_catch / nli_total if nli_total else 0.0
        total = n_catch + nli_catch
        tot = n_total + nli_total
        lines.append(f"  NLI recall:     {nli_catch}/{nli_total} = {nli_recall:.2f}")
        lines.append(f"  总 recall(NLI可用): {total}/{tot} = {total/tot:.2f}  目标≥0.6")
    else:
        lines.append(f"  NLI: 模型不可用，{nli_total} 用例降级漏报（skip）")
    lines.append(f"  已知缺口: E_scope_drift（SCOPE_DRIFT, fast LLM-only, 检出={gap_catch}）")
    report = "\n".join(lines)
    print(report)
    # 唯一硬断言：非 NLI 用例必须全检出（纯规则契约，不依赖模型）
    assert non_nli_recall == 1.0, f"non-NLI recall 退化: {n_catch}/{n_total}\n{report}"

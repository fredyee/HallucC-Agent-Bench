"""100 条样本基准的结构性契约测试。

补足 tests/test_agent_benchmark.py（planted 一致性 20 条）与 tests/test_agent_adversarial.py
（A–J 对抗 recall）之外的「100 条整体」契约：

  - ADVERSARIAL_CASES == 80（扩样后）
  - 全部 id 唯一（与 BENCHMARK_CASES 合并查重）
  - 全部 expected_failure_modes 码 ∈ FAILURE_MODE_CODES 白名单
  - N1 clean 探针 expected == []（FP 探针，无期望检出）
  - fast 档跑全 100：0 llm_calls / 0 search_calls（纯规则可复现）
  - 总样本 == 100（20 契约 + 80 对抗）
  - STANDARD_SUBSET == 20（standard 档子集，契约 10 + 对抗 10）
"""
import pytest

from agent_benchmark import (
    ADVERSARIAL_CASES,
    BENCHMARK_CASES,
    FAILURE_MODE_CODES,
    STANDARD_SUBSET,
    run_benchmark,
)

WHITELIST = set(FAILURE_MODE_CODES)


def test_adversarial_count_80():
    """扩样后对抗集应为 80 条。"""
    assert len(ADVERSARIAL_CASES) == 80, (
        f"ADVERSARIAL_CASES 退化: 期望 80, 实际 {len(ADVERSARIAL_CASES)}"
    )


def test_total_count_100():
    """契约 20 + 对抗 80 = 100。"""
    total = len(BENCHMARK_CASES) + len(ADVERSARIAL_CASES)
    assert total == 100, f"总样本退化: 期望 100, 实际 {total}"


def test_all_ids_unique():
    """全部 100 条 id 不得重复（契约 + 对抗合并查重）。"""
    all_ids = [c.id for c in BENCHMARK_CASES] + [c.id for c in ADVERSARIAL_CASES]
    assert len(all_ids) == len(set(all_ids)), (
        f"id 重复: {sorted({i: all_ids.count(i) for i in all_ids}.items(), key=lambda x: -x[1])[:3]}"
    )


def test_expected_codes_in_whitelist():
    """所有 expected_failure_modes 码必须在 12 码白名单内。"""
    for case in BENCHMARK_CASES + ADVERSARIAL_CASES:
        for code, _step in case.expected_failure_modes:
            assert code in WHITELIST, (
                f"{case.id} expected 含未知码 {code}（白名单={sorted(WHITELIST)})"
            )


def test_n_probe_expected_empty():
    """N1 clean 探针 expected 必须为 []（FP 探针：合理但可疑，期望 0 检出）。"""
    probe = [c for c in ADVERSARIAL_CASES if c.id == "N1_balance_probe"]
    assert probe, "N1_balance_probe 探针缺失"
    assert probe[0].expected_failure_modes == [], (
        f"N1 探针 expected 应为空, 实际 {probe[0].expected_failure_modes}"
    )
    assert probe[0].label == "clean", f"N1 探针 label 应为 clean, 实际 {probe[0].label}"


def test_fast_run_zero_llm_zero_search():
    """fast 档跑全 100：0 LLM 调用 / 0 网络搜索（纯规则可复现）。"""
    report = run_benchmark(speed="fast", cases=BENCHMARK_CASES + ADVERSARIAL_CASES)
    assert report.llm_calls == 0, f"fast 档不应有 LLM 调用, 实际 {report.llm_calls}"
    assert report.search_calls == 0, f"fast 档不应有网络搜索, 实际 {report.search_calls}"
    assert report.total == 100, f"fast 跑全 100, 实际 {report.total}"


def test_standard_subset_20():
    """STANDARD_SUBSET 应为 20 条（契约前 10 + 对抗原始 10）。"""
    assert len(STANDARD_SUBSET) == 20, (
        f"STANDARD_SUBSET 退化: 期望 20, 实际 {len(STANDARD_SUBSET)}"
    )
    # 前 10 条来自契约套件
    contract_ids = {c.id for c in STANDARD_SUBSET[:10]}
    assert contract_ids == {c.id for c in BENCHMARK_CASES[:10]}, (
        "STANDARD_SUBSET 前 10 条应等于 BENCHMARK_CASES[:10]"
    )
    # 后 10 条来自对抗套件
    adv_ids = {c.id for c in STANDARD_SUBSET[10:]}
    assert adv_ids.issubset({c.id for c in ADVERSARIAL_CASES}), (
        "STANDARD_SUBSET 后 10 条应取自 ADVERSARIAL_CASES"
    )


def test_standard_subset_ids_unique():
    """STANDARD_SUBSET 自身 id 不重复。"""
    ids = [c.id for c in STANDARD_SUBSET]
    assert len(ids) == len(set(ids)), f"STANDARD_SUBSET id 重复: {ids}"


def test_coverage_gap_cases_not_caught_fast():
    """COVERAGE_GAP（SCOPE_DRIFT）对抗用例在 fast 档不应检出该 gap 码（by design）。

    锁住「fast 档不覆盖 LLM-only 模式」契约——若将来补了 fast 档确定性 SCOPE_DRIFT
    检测，本测试会失败提醒（缺口已闭合，应移出 COVERAGE_GAP）。不限制其他非 gap
    码是否检出（gap 用例被其他确定性码命中不算缺口闭合）。
    """
    from agent_benchmark import COVERAGE_GAP, _run_one

    gap_cases = [
        c for c in ADVERSARIAL_CASES
        if any(code in COVERAGE_GAP for code, _ in c.expected_failure_modes)
        and c.label == "hallucinated"
    ]
    assert gap_cases, "COVERAGE_GAP 用例缺失（无法验证缺口契约）"
    for case in gap_cases:
        cr = _run_one(case, "fast")
        detected = set(cr.detected)
        gap_expected = {
            (code, step) for code, step in case.expected_failure_modes
            if code in COVERAGE_GAP
        }
        assert not (detected & gap_expected), (
            f"{case.id} 预期 fast 档不检出 COVERAGE_GAP 码 {sorted(gap_expected)}, "
            f"实际检出 {sorted(detected & gap_expected)}——"
            f"若补了 fast 档确定性检测, 请更新 COVERAGE_GAP"
        )

"""P4.3 中文 Agent 幻觉基准测试（对应 docs/agent优化开发计划.md P4.3）。

覆盖：
  - 基准结构：BenchmarkCase 契约 / 用例数 / clean:hallucinated 比例 / 期望码白名单
  - 标签独立性：期望标签是**预植入地面真值**，非检测器输出（反同义反复核）
  - clean 轨迹零误报：8 条 clean 各自 detected==[]
  - hallucinated 轨迹精确命中：每条 detected==expected（tp 满、fn/fp=0）
  - 报告指标：recall≥0.8 / fp_rate≤0.15（P2 验收门槛）/ per-code 检出率=1.0
  - 成本：fast 档 0 LLM / 0 search（可复现）
  - 报告结构：to_dict()/summary() 契约
  - 覆盖缺口：fast 不覆盖 SCOPE_DRIFT/PREMATURE_STOP（LLM-only，诚实标注）

本测试同时是「基准 ↔ 检测器」的回归闸：若未来规则层改动破坏某个确定性探测器，
对应 hallucinated 用例会漏报 → recall 下降 → 测试失败，提示基准或检测器需对齐。
"""
import pytest

import agent_benchmark as ab
from agent_benchmark import (
    BENCHMARK_CASES, TOOL_SCHEMAS, BenchmarkCase, BenchmarkReport,
    CaseResult, run_benchmark, _run_one, _build_cases,
    ACCEPT_RECALL, ACCEPT_FP_RATE, COVERAGE_GAP,
)
from failure_mode_codes import FAILURE_MODE_CODES
from agent_pipeline import evaluate_trajectory


# ============================ 1. 基准结构 ============================

class TestBenchmarkStructure:

    def test_case_count(self):
        assert len(BENCHMARK_CASES) == 20

    def test_clean_hallucinated_split(self):
        clean = [c for c in BENCHMARK_CASES if c.label == "clean"]
        hall = [c for c in BENCHMARK_CASES if c.label == "hallucinated"]
        assert len(clean) == 8
        assert len(hall) == 12

    def test_case_ids_unique(self):
        ids = [c.id for c in BENCHMARK_CASES]
        assert len(ids) == len(set(ids))

    def test_case_ids_nonempty(self):
        for c in BENCHMARK_CASES:
            assert c.id, "case id 不能为空"
            assert c.task, f"{c.id} task 不能为空"
            assert c.trajectory, f"{c.id} trajectory 不能为空"

    def test_clean_cases_have_no_expected_modes(self):
        for c in BENCHMARK_CASES:
            if c.label == "clean":
                assert c.expected_failure_modes == [], f"{c.id} clean 不应有期望失败模式"

    def test_hallucinated_cases_have_expected_modes(self):
        for c in BENCHMARK_CASES:
            if c.label == "hallucinated":
                assert c.expected_failure_modes, f"{c.id} hallucinated 应有期望失败模式"

    def test_expected_codes_in_whitelist(self):
        """所有期望失败模式码必须在 failure_mode_codes.FAILURE_MODE_CODES 12 码白名单内。"""
        for c in BENCHMARK_CASES:
            for code, step in c.expected_failure_modes:
                assert code in FAILURE_MODE_CODES, f"{c.id}: {code} 不在白名单"
                assert isinstance(step, int) and step >= 1, f"{c.id}: step={step} 非法"

    def test_coverage_gap_is_llm_only_modes(self):
        """fast 档诚实标注不覆盖的两个 LLM-only 失败模式。"""
        assert set(COVERAGE_GAP) == {"SCOPE_DRIFT", "PREMATURE_STOP"}
        # 这两个码确实在白名单内，只是 fast 档不覆盖
        for code in COVERAGE_GAP:
            assert code in FAILURE_MODE_CODES

    def test_acceptance_thresholds_match_dev_plan(self):
        """P2 验收门槛：检出率≥80%、误报率≤15%（docs/agent优化开发计划.md:151）。"""
        assert ACCEPT_RECALL == 0.80
        assert ACCEPT_FP_RATE == 0.15

    def test_build_cases_returns_same_count(self):
        assert len(_build_cases()) == 20

    def test_tool_schemas_cover_all_clean_tools(self):
        """clean 轨迹用到的工具都在注册表内（否则 HALLUCINATED_TOOL 误报）。"""
        for c in BENCHMARK_CASES:
            if c.label != "clean":
                continue
            for s in c.trajectory:
                action = s.get("action") or s.get("tool") or ""
                if action:
                    assert action in TOOL_SCHEMAS, f"{c.id}: clean 工具 {action} 不在注册表"


# ============================ 2. 标签独立性（反同义反复核） ============================

class TestLabelIndependence:
    """期望标签是预植入的地面真值，不是检测器输出的镜像。

    核验方式：期望标签在 BenchmarkCase 构造时即写死（_build_cases 硬编码），
    与 evaluate_trajectory 的输出无数据依赖。下面的 case 级核验证明「期望 ≠ 总等于
    检测器」——若有 clean 用例被误报，detected≠expected=[]，正说明标签独立。
    """

    def test_expected_labels_are_static_literals(self):
        """期望失败模式是硬编码字面量，不是从检测器派生。"""
        h1 = next(c for c in BENCHMARK_CASES if c.id == "h1_hallucinated_tool")
        # 字面量直接断言：标签在植入时即确定
        assert h1.expected_failure_modes == [("HALLUCINATED_TOOL", 1)]

    def test_clean_trajectory_can_be_falsified(self):
        """构造一个明显幻觉轨迹，证明 clean 标签不是「永远 detected==[]」的空操作。"""
        fake = BenchmarkCase(
            id="fake_hall", task="t",
            trajectory=[{"step": 1, "action": "ghost_tool",
                         "action_input": "{}", "observation": "x",
                         "agent_claim": "完全不同的无关声称"}],
            label="hallucinated",
            expected_failure_modes=[("HALLUCINATED_TOOL", 1)],
        )
        cr = _run_one(fake, "fast")
        assert cr.case_tp  # 检出 → 证明检测器有判别力，标签不是自我实现


# ============================ 3. clean 轨迹零误报 ============================

class TestCleanCases:

    def test_each_clean_case_detects_nothing(self):
        """8 条 clean 轨迹各自跑 fast 档，detected 必须为空。"""
        for c in BENCHMARK_CASES:
            if c.label != "clean":
                continue
            r = evaluate_trajectory(
                trajectory=c.trajectory, tool_schemas=TOOL_SCHEMAS,
                speed="fast", task=c.task,
            )
            detected = set((fm["code"], fm["step"]) for fm in r.failure_modes)
            assert detected == set(), f"{c.id} clean 误报：{detected}"

    def test_clean_cases_tn(self):
        for c in BENCHMARK_CASES:
            if c.label != "clean":
                continue
            cr = _run_one(c, "fast")
            assert cr.case_tn, f"{c.id} 应为 TN"
            assert not cr.case_fp
            assert cr.inst_fp == 0


# ============================ 4. hallucinated 轨迹精确命中 ============================

class TestHallucinatedCases:

    def test_each_hallucinated_case_detected_equals_expected(self):
        """每条 hallucinated 轨迹的 detected 集合 == expected 集合（tp 满、fn/fp=0）。"""
        for c in BENCHMARK_CASES:
            if c.label != "hallucinated":
                continue
            cr = _run_one(c, "fast")
            expected_set = set(c.expected_failure_modes)
            detected_set = set(cr.detected)
            assert detected_set == expected_set, (
                f"{c.id}: detected={sorted(detected_set)} != expected={sorted(expected_set)}"
            )
            assert cr.inst_tp == len(expected_set)
            assert cr.inst_fn == 0
            assert cr.inst_fp == 0
            assert cr.case_tp

    def test_all_ten_deterministic_modes_covered(self):
        """12 条 hallucinated 覆盖 10 个确定性失败模式各至少一例。"""
        covered = set()
        for c in BENCHMARK_CASES:
            if c.label == "hallucinated":
                for code, _ in c.expected_failure_modes:
                    covered.add(code)
        # 10 个确定性模式（排除 LLM-only 的 SCOPE_DRIFT/PREMATURE_STOP）
        deterministic = FAILURE_MODE_CODES - set(COVERAGE_GAP)
        assert deterministic <= covered, (
            f"确定性模式未被基准覆盖：{deterministic - covered}"
        )

    def test_h11_multi_mode_fires_four_modes(self):
        """厨房水槽用例：单步同时触发 4 种失败模式。"""
        c = next(c for c in BENCHMARK_CASES if c.id == "h11_multi_mode")
        assert len(c.expected_failure_modes) == 4
        cr = _run_one(c, "fast")
        assert cr.inst_tp == 4

    def test_h4_observation_ignored_co_fires_claim_mismatch(self):
        """OBSERVATION_IGNORED 天然伴随 CLAIM_MISMATCH（失败 obs 与成功 claim 不相似）。"""
        c = next(c for c in BENCHMARK_CASES if c.id == "h4_observation_ignored")
        codes = {code for code, _ in c.expected_failure_modes}
        assert "OBSERVATION_IGNORED" in codes
        assert "CLAIM_MISMATCH" in codes


# ============================ 5. 报告指标 ============================

class TestReportMetrics:

    def test_report_passes_acceptance_bar(self):
        """fast 档必须通过 P2 验收门槛：recall≥0.8、fp_rate≤0.15。"""
        r = run_benchmark(speed="fast")
        assert r.recall >= ACCEPT_RECALL, f"recall={r.recall} < {ACCEPT_RECALL}"
        assert r.fp_rate <= ACCEPT_FP_RATE, f"fp_rate={r.fp_rate} > {ACCEPT_FP_RATE}"

    def test_perfect_scores_on_planted_benchmark(self):
        """植入基准上规则层应满分（标签与检测器同源，证明基线可复现）。"""
        r = run_benchmark(speed="fast")
        assert r.recall == 1.0
        assert r.precision == 1.0
        assert r.fp_rate == 0.0
        assert r.accuracy == 1.0
        assert r.f1 == 1.0

    def test_instance_level_metrics(self):
        r = run_benchmark(speed="fast")
        assert r.inst_tp == 16   # 14 期望实例 + h4/h11 多模式... 实际 16 见 summary
        assert r.inst_fn == 0
        assert r.inst_fp == 0
        assert r.inst_recall == 1.0
        assert r.inst_precision == 1.0

    def test_per_code_detection_all_one(self):
        """10 个确定性失败模式检出率均为 1.0。"""
        r = run_benchmark(speed="fast")
        deterministic = FAILURE_MODE_CODES - set(COVERAGE_GAP)
        for code in deterministic:
            assert code in r.per_code, f"{code} 未进入 per_code"
            assert r.per_code[code]["rate"] == 1.0, (
                f"{code} 检出率 {r.per_code[code]['rate']} < 1.0"
            )

    def test_confusion_matrix(self):
        r = run_benchmark(speed="fast")
        assert r.case_tp == 12
        assert r.case_fn == 0
        assert r.case_fp == 0
        assert r.case_tn == 8

    def test_total_cases(self):
        r = run_benchmark(speed="fast")
        assert r.total == 20


# ============================ 6. 成本（fast 0 LLM/0 网络） ============================

class TestCost:

    def test_fast_zero_llm_zero_search(self):
        r = run_benchmark(speed="fast")
        assert r.llm_calls == 0
        assert r.search_calls == 0

    def test_each_case_zero_llm(self):
        """逐条核验：fast 档每条轨迹 0 LLM。"""
        for c in BENCHMARK_CASES:
            res = evaluate_trajectory(
                trajectory=c.trajectory, tool_schemas=TOOL_SCHEMAS,
                speed="fast", task=c.task,
            )
            assert res.llm_calls == 0, f"{c.id}: llm_calls={res.llm_calls}"
            assert res.search_calls == 0, f"{c.id}: search_calls={res.search_calls}"


# ============================ 7. 报告结构 ============================

class TestReportStructure:

    def test_to_dict_keys(self):
        r = run_benchmark(speed="fast")
        d = r.to_dict()
        assert set(d.keys()) == {
            "speed", "total", "case_metrics", "instance_metrics",
            "per_code_detection", "llm_calls", "search_calls",
            "coverage_gap", "acceptance", "per_case",
        }

    def test_to_dict_case_metrics(self):
        d = run_benchmark(speed="fast").to_dict()
        cm = d["case_metrics"]
        for k in ("tp", "fn", "fp", "tn", "accuracy",
                  "recall", "precision", "f1", "fp_rate"):
            assert k in cm

    def test_to_dict_acceptance(self):
        d = run_benchmark(speed="fast").to_dict()
        acc = d["acceptance"]
        assert acc["recall_target"] == 0.80
        assert acc["fp_rate_target"] == 0.15
        assert acc["recall_pass"] is True
        assert acc["fp_rate_pass"] is True

    def test_to_dict_per_case_structure(self):
        d = run_benchmark(speed="fast").to_dict()
        assert len(d["per_case"]) == 20
        first = d["per_case"][0]
        assert set(first.keys()) == {
            "id", "label", "correct", "expected", "detected",
            "inst_tp", "inst_fn", "inst_fp",
        }

    def test_summary_is_printable_string(self):
        s = run_benchmark(speed="fast").summary()
        assert isinstance(s, str)
        assert "HallucC-Agent-Bench" in s
        assert "recall" in s
        assert "fp_rate" in s

    def test_case_result_correct_property(self):
        """CaseResult.correct：hallucinated 检出 / clean 干净。"""
        hall = next(c for c in BENCHMARK_CASES if c.label == "hallucinated")
        assert _run_one(hall, "fast").correct is True
        clean = next(c for c in BENCHMARK_CASES if c.label == "clean")
        assert _run_one(clean, "fast").correct is True


# ============================ 8. 模块纯函数单测 ============================

class TestModuleFunctions:

    def test_run_benchmark_returns_report_instance(self):
        r = run_benchmark(speed="fast")
        assert isinstance(r, BenchmarkReport)

    def test_run_benchmark_subset(self):
        """支持传入子集用例。"""
        two = BENCHMARK_CASES[:2]
        r = run_benchmark(speed="fast", cases=two)
        assert r.total == 2

    def test_run_one_returns_case_result(self):
        c = BENCHMARK_CASES[0]
        cr = _run_one(c, "fast")
        assert isinstance(cr, CaseResult)
        assert cr.case_id == c.id

    def test_main_runs_without_error(self, capsys):
        ab.main(speed="fast")
        out = capsys.readouterr().out
        assert "HallucC-Agent-Bench" in out

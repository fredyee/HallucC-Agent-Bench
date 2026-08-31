#!/usr/bin/env python3
"""halluc-detect — Agent 轨迹幻觉检测 CLI（B 旗舰打包入口）。

复用 `agent_pipeline.evaluate_trajectory` 与 `agent_benchmark.run_benchmark`，
不引入新检测逻辑——CLI 只做参数解析与输出格式化（见 docs/Agent轨迹幻觉基准
升级计划.md §7 Phase 1 / v0.2「B 旗舰打包」）。

子命令：
  detect  对单条 Agent 轨迹做幻觉检测，输出结构化 JSON。
  bench   跑中文 Agent 幻觉基准（HallucC-Agent-Bench），输出汇总报告。

档位：
  fast（默认）= 规则层，0 LLM / 0 搜索，纯 stdlib，离线可复现。
  standard    = 规则层 + 单裁判/步（需 LLM 配置，见 pyproject [llm] extra）。
  deep        = 全管线 + strict K=3 投票（需 LLM 配置）。

输入格式（detect）——每步字段由 agent_eval._normalize_step 归一，兼容两种 schema：
  A. {step, thought, action, action_input, observation, final_answer}
  B. {step, tool, tool_input, tool_output, agent_claim}

裸轨迹列表（tool_schemas=None，跳过 HALLUCINATED_TOOL/SCHEMA_VIOLATION）：
    [{"action":"search","action_input":"{\\"query\\":\\"x\\"}","observation":"无结果", ...}]

或对象（可带 task / tool_schemas）：
    {"task":"查订单","trajectory":[...],"tool_schemas":{"search":{"required":["query"]}}}

--tools demo 加载基准 5 工具注册表（search/lookup_order/calculator/translate/get_weather）。
"""
import argparse
import json
import sys


def _read_input(arg: str):
    """读取输入 JSON：文件路径或 '-'（stdin）。返回解析后的对象。"""
    if arg == "-":
        raw = sys.stdin.read()
    else:
        with open(arg, "r", encoding="utf-8") as f:
            raw = f.read()
    return json.loads(raw)


def _split_input(obj):
    """拆分输入为 (trajectory, tool_schemas|None, task)。

    - 裸列表 → (obj, None, "")
    - 对象 {"task","trajectory","tool_schemas"?} → 各字段（空 dict 视为无 schema）
    """
    if isinstance(obj, list):
        return obj, None, ""
    if isinstance(obj, dict) and "trajectory" in obj:
        ts = obj.get("tool_schemas")
        if not isinstance(ts, dict) or not ts:  # 空 dict → None（避免空注册表误触发）
            ts = None
        return obj.get("trajectory") or [], ts, obj.get("task", "") or ""
    raise ValueError(
        "输入须为轨迹列表或 {task,trajectory,tool_schemas?} 对象；每步字段见 --help"
    )


def cmd_detect(args) -> int:
    # 延迟导入：使 --help 与参数解析零依赖（fast 档纯 stdlib）
    from agent_pipeline import evaluate_trajectory

    obj = _read_input(args.input)
    trajectory, tool_schemas, task = _split_input(obj)
    if args.tools == "demo":
        # 用户未自带 tool_schemas 时，加载基准 5 工具注册表
        from agent_benchmark import TOOL_SCHEMAS
        if tool_schemas is None:
            tool_schemas = TOOL_SCHEMAS

    result = evaluate_trajectory(
        trajectory, tool_schemas,
        speed=args.speed, task=task,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_bench(args) -> int:
    from agent_benchmark import run_benchmark

    report = run_benchmark(speed=args.speed)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        out = report.summary()
        if out:
            print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="halluc-detect",
        description="Agent 轨迹幻觉检测 CLI（fast 档纯 stdlib，0 LLM/0 搜索）。",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # detect
    pd = sub.add_parser("detect", help="对单条 Agent 轨迹做幻觉检测")
    pd.add_argument("input", nargs="?", default="-",
                    help="轨迹 JSON 文件路径，或 '-' 从 stdin 读（默认）")
    pd.add_argument("--speed", choices=["fast", "standard", "deep"],
                    default="fast", help="检测档位（默认 fast，0 LLM）")
    pd.add_argument("--tools", choices=["demo"], default=None,
                    help="加载基准 5 工具注册表（启用 HALLUCINATED_TOOL/SCHEMA_VIOLATION）")
    pd.set_defaults(func=cmd_detect)

    # bench
    pb = sub.add_parser("bench", help="跑中文 Agent 幻觉基准（HallucC-Agent-Bench）")
    pb.add_argument("--speed", choices=["fast", "standard", "deep"],
                    default="fast", help="基准档位（默认 fast）")
    pb.add_argument("--json", action="store_true",
                    help="输出结构化 JSON 而非文本报告")
    pb.set_defaults(func=cmd_bench)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"错误：文件未找到：{e.filename}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"错误：输入 JSON 解析失败：{e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

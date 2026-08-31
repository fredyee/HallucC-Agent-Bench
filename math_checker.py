"""数学/代码执行验证 (P1-3)。提取算式→Python执行→对比声明中的结果。"""
import re
import subprocess
import sys


def check(text: str) -> list[dict]:
    """提取文本中的算式并执行验证。返回问题列表。"""
    issues = []
    expressions = _extract_expressions(text)
    for expr in expressions:
        result = _evaluate(expr["expression"])
        if result is None:
            continue
        if abs(result - expr["claimed"]) > 0.001:
            issues.append({
                "type": "math_error",
                "expression": expr["expression"],
                "claimed": expr["claimed"],
                "actual": result,
                "context": expr["context"],
            })
    return issues


def _extract_expressions(text: str) -> list[dict]:
    """提取文本中的数学表达式和声称结果。"""
    patterns = [
        # "X + Y = Z" 或 "X - Y = Z"
        r"(?P<expr>[\d.]+)\s*([+\-*/])\s*([\d.]+)\s*=\s*([\d.]+)",
    ]
    results = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            try:
                expr_str = f"{m.group(1)} {m.group(2)} {m.group(3)}"
                actual = eval(expr_str)
                claimed = float(m.group(4))
                results.append({
                    "expression": expr_str,
                    "claimed": claimed,
                    "context": text[max(0, m.start()-20):m.end()+20],
                })
            except (ValueError, SyntaxError, ZeroDivisionError):
                continue
    # 也检测"X 年出生，Y 年后"这类
    age_pattern = r"(19\d{2}|20\d{2})\s*年.*出生.*?\+?\s*(\d+)\s*[岁年].*[=是]\s*(19\d{2}|20\d{2})"
    for m in re.finditer(age_pattern, text):
        try:
            birth = int(m.group(1))
            age = int(m.group(2))
            claimed_year = int(m.group(3))
            actual_year = birth + age
            if abs(actual_year - claimed_year) > 1:  # 允许±1年误差
                results.append({
                    "expression": f"{birth} + {age}",
                    "claimed": claimed_year,
                    "actual": actual_year,
                    "context": text[max(0, m.start()-20):m.end()+20],
                })
        except (ValueError, IndexError):
            continue
    return results


def _evaluate(expression: str) -> float | None:
    """安全执行 Python 表达式。仅允许基础算术。"""
    # 白名单检查
    allowed = set("0123456789. +-*/()")
    if not all(c in allowed for c in expression):
        return None
    try:
        result = eval(expression)
        if isinstance(result, (int, float)):
            return float(result)
    except Exception:
        pass
    return None

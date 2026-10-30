# HallucC 中文幻觉标注评测集 v1.0

100 条中文 AI 幻觉标注数据，用于评测和提升中文幻觉检测工具的准确性。

## 数据集概况

| 项目 | 说明 |
|---|---|
| 总样本数 | 100 |
| 总声明数 | ~260 条 |
| 标注为幻觉 (refuted) | ~180 条 |
| 标注为事实 (supported) | ~50 条 |
| 标注为无法确定 (unverified) | ~30 条 |
| 文件格式 | JSONL |
| 许可证 | CC-BY 4.0 |

## 幻觉类别分布

| 类别 | 数量 | 说明 |
|---|---|---|
| 编造事实 | 24 | 完全虚构不存在的事件/政策/产品 |
| 张冠李戴 | 14 | 将 A 的属性错误归属于 B |
| 数据错误 | 14 | 具体数字、统计数据错误 |
| 时间错误 | 12 | 历史事件发生时间错误 |
| 地点错误 | 10 | 地理位置信息错误 |
| 人名错误 | 8 | 人物姓名错误归属 |
| 引用捏造 | 6 | 伪造名人名言或文献引用 |
| 逻辑矛盾 | 6 | 文本内部存在自相矛盾 |
| 过度推断 | 6 | 从有限前提做出不合理推断 |

## 数据格式

每行一个 JSON 对象：

```json
{
  "id": "eval_001",
  "category": "编造事实",
  "text": "原始文本",
  "claims": [
    {
      "text": "待验证的原子声明",
      "expected_status": "refuted|supported|unverified",
      "note": "标注说明"
    }
  ]
}
```

## 使用方法

### 加载数据集

```python
import json

samples = []
with open("chinese_hallucination_benchmark.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            samples.append(json.loads(line))

print(f"加载 {len(samples)} 条样本")
```

### 运行评测

```bash
# 使用 HallucC 评测框架
cd /path/to/hallucc
python eval.py --dataset eval_set/chinese_hallucination_benchmark.jsonl
```

## 标注原则

1. **refuted**：可通过公开搜索验证为虚假 — 必须有明确的反证信源
2. **supported**：可通过公开搜索验证为真实 — 主流权威来源可确认
3. **unverified**：主观意见、未来预测、或无法通过搜索获取确切答案的内容

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-08 | 初始版本，100 条样本，9 大类别 |

## 贡献

欢迎提交 Issue 和 PR 扩充数据集。请确保每条新增样本附带至少一个可验证的来源链接。

🤖 Generated with [Claude Code](https://claude.com/claude-code)

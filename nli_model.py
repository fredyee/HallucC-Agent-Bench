"""专用评估模型集成 (A4)：Erlangshen-Roberta-NLI 中文蕴含模型作为第二评估维度。

真实 NLI（自然语言推断）三分类 entailment/neutral/contradiction，而非向量相似度。
双 tier 懒加载：
  - fast/standard  → Erlangshen-Roberta-110M-NLI（~410MB，CPU ~50-100ms，默认）
  - deep           → Erlangshen-Roberta-330M-NLI（~1.3GB，精审档专用）
agent deep 档（per-request /detect-agent speed=deep + 离线 --speed deep）自动切 330M；
fast/standard 留 110M。guard_agent（事中 <10ms hot path）固定 110M。
HALLUCC_NLI_MODEL 环境变量 = 全局 override 两 tier（保留旧语义 + ablation：设 110M
强制 deep=110M；CI pin 110M 避免触发 1.3GB 下载；向后兼容）。
经 hf-mirror.com 下载（HF_ENDPOINT=https://hf-mirror.com，huggingface.co 不可达时的镜像）。

transformers/torch 未安装或加载失败 → 返回 None（降级；该 tier slot 独立短路，
不污染另一 tier。HANDOFF_LOSS 回退到纠正标记路径，CLAIM_MISMATCH 主路径交由
Jaccard+claim_alignment）。

返回契约不变：{"score": 0-1, "verdict": "supported|contradicted|neutral"}。
调用方通过 keyword-only tier= 选档；不传 tier → 默认 "fast"/110M，零行为变化。
"""
import os

from log_utils import get_logger

logger = get_logger(__name__)

# 双 tier 模型名：fast=110M（standard/fast 默认），deep=330M（精审档专用）
_MODEL_NAMES = {
    "fast": "IDEA-CCNL/Erlangshen-Roberta-110M-NLI",
    "deep": "IDEA-CCNL/Erlangshen-Roberta-330M-NLI",
}

# huggingface.co 不可达时走 hf-mirror（必须在 from_pretrained 之前设置；
# 已显式设置则尊重用户选择）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 双槽懒加载：每 tier 独立 slot，失败只 null 该 tier，不污染另一 tier。
# slot = {"model","tokenizer","label_map","load_attempted"}
# load_attempted：加载已尝试（成功或失败）——避免逐调用重试刷屏（尤其 deep 1.3GB 重下）
_slots: dict[str, dict] = {}


def _slot(tier: str) -> dict:
    """取/建 tier 的 slot（per-tier 独立 load_attempted 短路）。"""
    s = _slots.get(tier)
    if s is None:
        s = {"model": None, "tokenizer": None, "label_map": None, "load_attempted": False}
        _slots[tier] = s
    return s


def _model_name_for(tier: str) -> str:
    """tier → 模型名。HALLUCC_NLI_MODEL 环境变量若设置 → 全局 override 两 tier
    （保留旧语义 + ablation + CI pin + 向后兼容）；否则按 tier 取 _MODEL_NAMES。
    未知 tier 回退 fast 的模型名。
    """
    env = os.environ.get("HALLUCC_NLI_MODEL")
    if env:
        return env
    return _MODEL_NAMES.get(tier, _MODEL_NAMES["fast"])


def _resolve_labels(id2label: dict) -> dict:
    """从 config.id2label 找出三分类索引（不硬编码顺序——不同模型 entail/contra 顺序不一）。

    Erlangshen 一般 {0:entailment,1:neutral,2:contradiction}，但有变体把 contra/neutral 对调，
    故按 label 文本匹配，不按位置假设。
    """
    ent = contra = neut = None
    for idx, label in id2label.items():
        lab = str(label).lower()
        if ent is None and ("entail" in lab or lab in ("yes", "support", "支持", "蕴含")):
            ent = int(idx)
        elif contra is None and ("contradict" in lab or lab in ("no", "矛盾", "contradict")):
            contra = int(idx)
        elif neut is None and ("neutral" in lab or lab in ("maybe", "中", "neutral")):
            neut = int(idx)
    return {"entailment": ent, "contradiction": contra, "neutral": neut}


def _load_model(tier: str = "fast") -> bool:
    """懒加载 Erlangshen NLI 模型（CPU 可运行）。加载失败只告警一次。
    tier="fast"→110M，tier="deep"→330M。无参调用默认 fast——可用性探测语义不变。
    返回该 tier 是否可用。"""
    s = _slot(tier)
    if s["load_attempted"]:
        return s["model"] is not None
    s["load_attempted"] = True

    name = _model_name_for(tier)
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        logger.warning(
            "transformers/torch 未安装，NLI 模型不可用(tier=%s)（pip install transformers torch）"
            "——HANDOFF_LOSS 降级为纠正标记回退",
            tier,
        )
        return False
    try:
        s["tokenizer"] = AutoTokenizer.from_pretrained(name)
        s["model"] = AutoModelForSequenceClassification.from_pretrained(name)
        s["model"].eval()
        s["label_map"] = _resolve_labels(s["model"].config.id2label)
        if any(v is None for v in s["label_map"].values()):
            logger.warning(
                "NLI id2label 无法解析三分类(tier=%s)（entail/neutral/contradict）: %s",
                tier, s["model"].config.id2label,
            )
            s["model"] = None
            s["tokenizer"] = None
            return False
        logger.info(
            "NLI 模型加载成功: %s（tier=%s, labels=%s）",
            name, tier, s["label_map"],
        )
        return True
    except Exception as e:
        logger.warning("NLI 模型加载失败(tier=%s, %s): %s", tier, name, str(e)[:120])
        s["model"] = None
        s["tokenizer"] = None
        return False


def warmup(tiers: tuple[str, ...] = ("fast", "deep")) -> None:
    """预加载指定 tier 的 NLI 模型（供预热脚本调用）。

    上线前对 deep tier 跑一次，把 330M 权重拉入 HF 磁盘缓存并完成首次加载，
    避免首个 /detect-agent speed=deep 请求在线下载 1.3GB 阻塞~分钟级。
    """
    for t in tiers:
        _load_model(t)


def evaluate_faithfulness(claim: str, evidence: str, *, tier: str = "fast") -> dict | None:
    """用 NLI 模型判断声明是否被证据支持。

    premise=evidence（观察，既定事实），hypothesis=claim（待测声称）。
    keyword-only tier 选档：tier="fast"→110M（默认），tier="deep"→330M。
    返回 {"score": 0-1, "verdict": "supported|contradicted|neutral"} 或 None（模型不可用）。
    score = 胜出标签的 softmax 概率（真实 NLI 置信度，非旧版 cosine 相似度）。
    """
    s = _slot(tier)
    if s["load_attempted"]:
        if s["model"] is None:
            return None
    elif not _load_model(tier):
        return None
    if not claim or not evidence:
        return None

    try:
        import torch
        # premise=evidence, hypothesis=claim —— 成对编码（[CLS] evidence [SEP] claim [SEP]）
        inputs = s["tokenizer"](
            str(evidence), str(claim),
            truncation=True, max_length=256, return_tensors="pt",
        )
        with torch.no_grad():
            logits = s["model"](**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)

        label_map = s["label_map"]
        ent_idx = label_map["entailment"]
        contra_idx = label_map["contradiction"]
        neut_idx = label_map["neutral"]

        p_ent = float(probs[ent_idx])
        p_contra = float(probs[contra_idx])
        p_neut = float(probs[neut_idx]) if neut_idx is not None else 0.0

        # 取概率最高的标签（contradicted 优先于 supported 优先于 neutral——同概率时偏向保守抓矛盾）
        if p_contra >= p_ent and p_contra >= p_neut:
            return {"score": round(p_contra, 2), "verdict": "contradicted"}
        if p_ent >= p_neut:
            return {"score": round(p_ent, 2), "verdict": "supported"}
        return {"score": round(p_neut, 2), "verdict": "neutral"}
    except Exception as e:
        logger.warning("NLI 评估失败(tier=%s): %s", tier, str(e)[:100])
        return None

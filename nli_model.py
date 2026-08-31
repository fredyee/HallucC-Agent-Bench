"""专用评估模型集成 (A4)：轻量 NLI 模型作为第二评估维度。
尝试加载 sentence-transformers，不可用则降级。
"""
from log_utils import get_logger

logger = get_logger(__name__)

_model = None
_load_attempted = False  # 加载已尝试（成功或失败）——避免逐调用重试刷屏


def _load_model():
    """懒加载轻量 NLI 模型（CPU 可运行）。加载失败只告警一次。"""
    global _model, _load_attempted
    if _load_attempted:
        return _model
    _load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        logger.info("NLI 模型加载成功: bge-small-zh-v1.5")
    except ImportError:
        logger.warning("sentence-transformers 未安装，NLI 模型不可用")
    except Exception as e:
        logger.warning("NLI 模型加载失败: %s", str(e)[:100])
    return _model


def evaluate_faithfulness(claim: str, evidence: str) -> dict | None:
    """用 NLI 模型判断声明是否被证据支持。
    返回 {"score": 0-1, "verdict": "supported|contradicted|neutral"} 或 None。
    """
    model = _load_model()
    if model is None:
        return None

    try:
        # 用句子向量相似度判断
        embeddings = model.encode([claim, evidence], normalize_embeddings=True)
        import numpy as np
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        similarity = max(0, min(1, similarity))

        verdict = "supported" if similarity > 0.7 else "neutral" if similarity > 0.4 else "contradicted"
        return {"score": round(similarity, 2), "verdict": verdict}
    except Exception as e:
        logger.warning("NLI 评估失败: %s", str(e)[:100])
        return None

"""LLM 封装层（开源精简版）。无 config.py 依赖，纯环境变量驱动。

OpenAI 兼容 SDK 覆盖 DeepSeek / OpenAI / GLM / DashScope（均支持 OpenAI 兼容
接口），按 LLM_PROVIDER 切换。judge（agent_benchmark_judge.py）与 standard/deep
档裁判层共用此客户端。

配置（环境变量，或 .env 经 python-dotenv 加载）：
  LLM_PROVIDER   deepseek(默认) | openai | glm | dashscope
  DEEPSEEK_API_KEY / OPENAI_API_KEY / GLM_API_KEY / DASHSCOPE_API_KEY
  LLM_MODEL       可选，覆盖 provider 默认 model

不开 mock 模式——fast 档纯规则不需 LLM；standard/judge 无 key 则抛清晰错误，
不静默降级伪装成真实判定。
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Optional

from log_utils import get_logger

logger = get_logger(__name__)

try:
    from openai import (
        OpenAI,
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
        APIStatusError,
    )
    _HAS_OPENAI = True
except ImportError:  # pragma: no cover - 未装 openai SDK
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore[assignment,misc]

_LLM_TIMEOUT = 30.0
_LLM_MAX_RETRIES = 3
_LLM_RETRY_BASE_DELAY = 0.5
_RETRYABLE = (
    (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
    if _HAS_OPENAI else ()
)

# 调用计数（无 config.inc_llm_call；模块级计数供本地观测）
_llm_call_count = 0


def inc_llm_call() -> None:
    global _llm_call_count
    _llm_call_count += 1


class LLMError(Exception):
    pass


def _parse_json(text: str) -> Any:
    """从模型输出抢救 JSON（容忍 ```json 围栏与常见损坏）。

    解析失败返回 {parse_error: ...} 标记，下游 .get() 容错降级——不伪装成
    真实低置信判定。
    """
    import re

    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    s = s.strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        logger.warning("LLM 输出无可解析 JSON 对象: %s", text[:200])
        return _parse_error_marker(text)
    body = s[start : end + 1]

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(
        r'"[^"]*"\s*:\s*[^,}\]]+\s*\n\s*"', lambda m: m.group(0).rstrip() + ",", body
    )
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    logger.warning("LLM 输出 JSON 解析失败（已尝试修复），原文: %s", text[:200])
    return _parse_error_marker(text)


def _parse_error_marker(text: str) -> dict:
    return {
        "status": "parse_error",
        "confidence": 0.0,
        "reason": "模型返回格式异常，无法解析为有效结果",
        "_raw_preview": text[:200],
    }


PROVIDERS = {
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.org/v1",
        "default_model": "gpt-4o-mini",
    },
    "glm": {
        "key_env": "GLM_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
    },
    "dashscope": {
        "key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
}


class LLMClient:
    def __init__(self, provider: Optional[str] = None) -> None:
        self._client = None
        self._model = "deepseek-chat"
        self._provider_name = "deepseek"
        self._configure(provider)

    def _configure(self, provider: Optional[str] = None) -> None:
        if not _HAS_OPENAI:
            raise LLMError("未安装 openai SDK（pip install openai 或 pip install 'halluc-detect[llm]'）")

        name = provider or os.getenv("LLM_PROVIDER") or "deepseek"
        if name not in PROVIDERS:
            raise LLMError(f"未知 LLM_PROVIDER: {name}（支持 {list(PROVIDERS)}）")
        prov = PROVIDERS[name]
        key = os.getenv(prov["key_env"])
        if not key:
            raise LLMError(
                f"{name} 需要配置环境变量 {prov['key_env']}（在 .env 设置或 export）"
            )
        model = os.getenv("LLM_MODEL") or prov["default_model"]
        self._client = OpenAI(
            api_key=key,
            base_url=prov["base_url"],
            timeout=_LLM_TIMEOUT,
            max_retries=0,
        )
        self._model = model
        self._provider_name = name

    @property
    def provider(self) -> str:
        return self._provider_name

    def chat_json(self, prompt: str) -> Any:
        """调用模型并返回解析后的 JSON。temperature=0 + json_object 强制 JSON。"""
        assert self._client is not None
        inc_llm_call()
        last_err: Exception | None = None
        for attempt in range(_LLM_MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": "你只输出合法 JSON，不输出任何其他文字。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=_LLM_TIMEOUT,
                )
                return _parse_json(resp.choices[0].message.content or "")
            except _RETRYABLE as e:
                last_err = e
                if attempt >= _LLM_MAX_RETRIES:
                    break
                delay = _LLM_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.25)
                logger.warning(
                    "LLM 调用瞬时错误（%s），第 %d 次重试（%.2fs 后）: %s",
                    self._provider_name, attempt + 1, delay, e,
                )
                time.sleep(delay)
            except APIStatusError as e:
                raise LLMError(
                    f"LLM 调用失败 ({self._provider_name}, HTTP {e.status_code}): {e}"
                ) from e
        raise LLMError(
            f"LLM 调用重试 {_LLM_MAX_RETRIES} 次仍失败 ({self._provider_name}): {last_err}"
        ) from last_err

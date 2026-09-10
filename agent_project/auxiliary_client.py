"""Auxiliary LLM client (对标 Hermes auxiliary_client).

Side 任务(总结/抽取/画像更新)走本客户端, 不再与主推理抢主 backend:
- 默认复用主 backend(零行为变化), 但调用被单独计数(side_calls)可观测;
- 后续配独立小模型时, 只需换 backend 实例, 调用方不用改;
- 失败策略与主链路解耦: side 任务失败只降级, 不抛错.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class AuxiliaryClient:
    """Side-LLM 薄封装: 总结/抽取类任务的统一出口."""

    def __init__(self, backend: Any = None, default_temperature: float = 0.4,
                 default_max_tokens: int = 1024):
        self._backend = backend
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.side_calls = 0
        self.side_failures = 0

    @classmethod
    def from_agent(cls, agent: Any) -> "AuxiliaryClient":
        """默认复用 agent 主 backend; 后续可按 config 换独立模型."""
        return cls(backend=getattr(agent, "backend", None))

    @property
    def backend(self) -> Any:
        return self._backend

    def attach(self, backend: Any) -> None:
        """热插独立 side 模型(不重启进程)."""
        self._backend = backend

    def generate(self, prompt: str, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 token_callback: Optional[Callable[[int], None]] = None,
                 **kwargs: Any) -> str:
        """调 side 模型生成; 失败抛异常由调用方决定降级(保持与 backend 一致语义)."""
        if self._backend is None:
            raise RuntimeError("auxiliary backend not attached")
        self.side_calls += 1
        try:
            return self._backend.generate(
                prompt,
                n_loops=1,
                temperature=self.default_temperature if temperature is None else temperature,
                max_tokens=self.default_max_tokens if max_tokens is None else max_tokens,
                token_callback=token_callback,
                **kwargs,
            )
        except Exception:
            self.side_failures += 1
            raise

    def summarize(self, prompt: str, max_tokens: int = 1024,
                  token_callback: Optional[Callable[[int], None]] = None) -> str:
        """总结专用入口(低温短输出; prompt 自带指令, 跳过后端默认 system).

        老后端不支持 system_override 时抛 TypeError, 由 generate 计数后透出,
        调用方按既有降级处理(与原来直调 backend 语义一致).
        """
        try:
            return self.generate(prompt, temperature=0.4, max_tokens=max_tokens,
                                 token_callback=token_callback, system_override="")
        except TypeError:
            return self.generate(prompt, temperature=0.4, max_tokens=max_tokens,
                                 token_callback=token_callback)

    def stats(self) -> Dict[str, int]:
        return {"side_calls": self.side_calls, "side_failures": self.side_failures}

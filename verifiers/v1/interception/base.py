"""The interception contract: hand each rollout a slot on a host interception server.

Three shapes, picked by `InterceptionConfig` type (see `make_interception`): a single
`InterceptionServer`, a fixed `StaticInterceptionPool`, or an on-demand
`ElasticInterceptionPool`. From the outside they behave the same: start/stop (or the async
context manager wrapping them) bound the lifecycle — one eval, one topology run, or one
agent `.run` — and each rollout `acquire`s a slot and frees it. An `Interception` can be
shared: whoever entered it owns the lifecycle; borrowers only `acquire`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import TYPE_CHECKING, ClassVar, Self

from pydantic import model_validator
from pydantic_config import BaseConfig

if TYPE_CHECKING:
    from verifiers.v1.runtimes.base import Runtime
    from verifiers.v1.session import RolloutSession


class BaseInterceptionConfig(BaseConfig):
    """Base for the interception types — the discriminated union's common type. Per-type
    fields live on the subclasses (server's `tunnel`, static's `servers`, elastic's
    `multiplex`)."""

    type: str

    def host_tunnel(self):
        """Transport for host tool servers, scoped to this environment: the
        interception's own tunnel when it can expose any port, else Prime's."""
        from verifiers.v1.interception.tunnel import (
            PrimeTunnel,
            PrimeTunnelConfig,
            make_tunnel,
        )

        tunnel = make_tunnel(getattr(self, "tunnel", PrimeTunnelConfig()))
        return tunnel if tunnel.exposes_any_port else PrimeTunnel()

    @model_validator(mode="wrap")
    @classmethod
    def _resolve_interception(cls, value, handler):
        if cls is BaseInterceptionConfig and isinstance(value, dict):
            interception_type = value.get("type")
            if isinstance(interception_type, str) and interception_type:
                from verifiers.v1.interception import interception_config_type

                return interception_config_type(interception_type).model_validate(value)
        return handler(value)


# (base_url, model_secret, state_secret): model inference and task state deliberately use
# separate capabilities. `base_url` is universally reachable — the interception is exposed
# whenever any consumer is remote.
Slot = tuple[str, str, str]


class Interception(ABC):
    """How rollouts reach the host interception server. `start` brings the servers up (or
    arms lazy growth), `stop` tears every server (+ its tunnel) down via `stack` — LIFO,
    even if one teardown fails; `async with` wraps the two. Each rollout `acquire`s a slot
    and frees it on exit."""

    config_cls: ClassVar[type[BaseInterceptionConfig]]

    def __init__(self) -> None:
        self.stack = AsyncExitStack()

    @abstractmethod
    async def start(self) -> None:
        """Bring the interception up; resources land on `stack` so `stop` frees them."""

    async def stop(self) -> None:
        await self.stack.aclose()

    async def __aenter__(self) -> Self:
        try:
            await self.start()
        except BaseException:
            # unwind whatever `start` already put on the stack
            await self.stop()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    @abstractmethod
    def acquire(
        self, session: RolloutSession, runtime: Runtime | None = None
    ) -> AbstractAsyncContextManager[Slot]:
        """Register `session` on a server (bringing one up if needed) and yield its `Slot`;
        free it on exit. `runtime` is the rollout's started runtime: an interception
        that schedules the runtime's resources around model calls binds the session to
        it; others ignore it."""

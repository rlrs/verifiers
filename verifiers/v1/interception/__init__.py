from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from functools import cache
from typing import TYPE_CHECKING, cast

from pydantic_config import BaseConfig

from verifiers.v1.interception.base import BaseInterceptionConfig, Interception, Slot
from verifiers.v1.interception.pool import (
    ElasticInterceptionPool,
    ElasticInterceptionPoolConfig,
    StaticInterceptionPool,
    StaticInterceptionPoolConfig,
)
from verifiers.v1.interception.server import (
    InterceptionServer,
    InterceptionServerConfig,
)
from verifiers.v1.runtimes import Runtime, runtime_is_local
from verifiers.v1.session import RolloutSession

if TYPE_CHECKING:
    from verifiers.v1.mcp import SharedToolServer

# Explicit built-ins retain their full CLI help; the base resolves installed types.
InterceptionConfig = (
    InterceptionServerConfig
    | StaticInterceptionPoolConfig
    | ElasticInterceptionPoolConfig
    | BaseInterceptionConfig
)

_BUILTIN_INTERCEPTIONS = {
    "server": InterceptionServer,
    "static": StaticInterceptionPool,
    "elastic": ElasticInterceptionPool,
}


@cache
def find_interception_class(interception_type: str) -> type[Interception] | None:
    """Resolve a built-in or installed interception class, if present."""
    if cls := _BUILTIN_INTERCEPTIONS.get(interception_type):
        return cls
    from verifiers.v1.utils.loaders import _plugin_class, import_interception

    try:
        module = import_interception(interception_type)
    except ModuleNotFoundError:
        return None
    cls = cast(type[Interception], _plugin_class(module, Interception, "interception"))
    if not issubclass(cls.config_cls, BaseInterceptionConfig):
        raise TypeError("interception config_cls must subclass BaseInterceptionConfig")
    if cls.config_cls.model_fields["type"].default != interception_type:
        raise ValueError(
            f"interception module and config type disagree for {interception_type!r}"
        )
    return cls


def interception_class(interception_type: str) -> type[Interception]:
    if cls := find_interception_class(interception_type):
        return cls
    raise ValueError(f"interception type {interception_type!r} is not installed")


def interception_config_type(
    interception_type: str,
) -> type[BaseInterceptionConfig]:
    return interception_class(interception_type).config_cls


def requires_tunnel(
    harness_is_local: bool,
    server_configs: Iterable[BaseConfig] = (),
    shared: "Iterable[SharedToolServer]" = (),
) -> bool:
    """Whether the interception must be exposed via a tunnel — some consumer is off the
    host network: the harness itself, a live `shared` server in a remote runtime, or a
    tool server config placing one there (each reaches the `/state` channel from
    its own runtime). Skipped as non-consumers: a `colocated` server (shares the
    harness's runtime, covered by `harness_is_local`), a config-`url` server (external —
    it connects out), and an `external` shared server (outside the state machinery
    entirely). False means every consumer reaches the server at localhost."""
    if not harness_is_local:
        return True
    if any(not s.external and not s.local for s in shared):
        return True
    for config in server_configs:
        if getattr(config, "url", None) or config.colocated:
            continue
        if not runtime_is_local(config.runtime):
            return True
    return False


def make_interception(
    config: InterceptionConfig,
    *,
    requires_tunnel: bool,
    state_service_secrets: tuple[str, ...] = (),
) -> Interception:
    """The interception for a config, picked by type (the host-side counterpart to
    `make_runtime`). With `requires_tunnel`, each server is exposed through its configured
    tunnel; otherwise it remains on host loopback. The caller computes this requirement."""
    cls = interception_class(config.type)
    if not isinstance(config, cls.config_cls):
        raise TypeError(
            f"interception config for {config.type!r} must be "
            f"{cls.config_cls.__name__}, got {type(config).__name__}"
        )
    return cls(config, requires_tunnel, state_service_secrets)


@asynccontextmanager
async def serve_interception(
    interception: Interception | None,
    runtime: Runtime,
    session: RolloutSession,
    servers: list,
    shared_tools: "dict[str, SharedToolServer]",
) -> AsyncIterator[Slot]:
    """A slot on the shared interception when one was injected (its owner keeps the
    lifecycle), else on a per-rollout `InterceptionServer` owned — brought up and torn
    down — by the caller's context."""
    if interception is not None:
        async with interception.acquire(session) as slot:
            yield slot
        return
    tunneled = requires_tunnel(
        runtime.is_local,
        [server.config for server in servers],
        shared_tools.values(),
    )
    server = InterceptionServer(
        requires_tunnel=tunneled,
        state_service_secrets=tuple(
            tool.state_secret for tool in shared_tools.values() if tool.state_secret
        ),
    )
    async with server, server.acquire(session) as slot:
        yield slot


__all__ = [
    "BaseInterceptionConfig",
    "ElasticInterceptionPool",
    "ElasticInterceptionPoolConfig",
    "Interception",
    "InterceptionConfig",
    "InterceptionServer",
    "InterceptionServerConfig",
    "Slot",
    "StaticInterceptionPool",
    "StaticInterceptionPoolConfig",
    "find_interception_class",
    "interception_class",
    "interception_config_type",
    "make_interception",
    "requires_tunnel",
    "serve_interception",
]

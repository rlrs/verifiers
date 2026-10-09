from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Annotated

from pydantic import Field

from verifiers.v1.interception.tunnel.base import BaseTunnelConfig, Tunnel
from verifiers.v1.interception.tunnel.custom import CustomTunnel, CustomTunnelConfig
from verifiers.v1.interception.tunnel.prime import PrimeTunnel, PrimeTunnelConfig

# Discriminated on `type` so the CLI selects with `--interception.tunnel.type prime|custom`.
TunnelConfig = Annotated[
    PrimeTunnelConfig | CustomTunnelConfig, Field(discriminator="type")
]


def make_tunnel(config: TunnelConfig) -> Tunnel:
    """The tunnel for a config, picked by type."""
    if isinstance(config, CustomTunnelConfig):
        return CustomTunnel(config)
    return PrimeTunnel(config)


__all__ = [
    "BaseTunnelConfig",
    "CustomTunnel",
    "CustomTunnelConfig",
    "PrimeTunnel",
    "PrimeTunnelConfig",
    "Tunnel",
    "TunnelConfig",
    "make_tunnel",
]


_host_tunnel: ContextVar[Tunnel | Callable[[], Tunnel] | None] = ContextVar(
    "host_tool_tunnel", default=None
)


def configured_host_tunnel() -> Tunnel | None:
    value = _host_tunnel.get()
    return value() if callable(value) else value


def host_tunnel() -> Tunnel:
    """The serving environment's transport, created only when a tunnel is needed."""
    return configured_host_tunnel() or PrimeTunnel()


@contextmanager
def using_host_tunnel(tunnel: Tunnel | Callable[[], Tunnel]):
    token = _host_tunnel.set(tunnel)
    try:
        yield
    finally:
        try:
            _host_tunnel.reset(token)
        except ValueError:
            # Exited in another context than it was entered in (GEPA enters and exits
            # `env.serving()` in separate `run_until_complete` calls): the setting
            # belonged to the entering task's context, which ended with it.
            pass

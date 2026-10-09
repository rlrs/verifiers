"""The host tool-server tunnel: which tunnel an env's tool servers use, and that
it is visible however the env's serving scope and episodes are driven."""

import asyncio
import contextlib
from types import SimpleNamespace

import verifiers.v1.interception.tunnel as tunnels
from verifiers.v1.env import Env
from verifiers.v1.interception.server import InterceptionServerConfig
from verifiers.v1.interception.tunnel import (
    CustomTunnelConfig,
    Tunnel,
    configured_host_tunnel,
    using_host_tunnel,
)


class StubTunnel(Tunnel):
    def expose(self, port):
        raise NotImplementedError


def test_custom_interception_tunnel_does_not_serve_tool_ports(monkeypatch) -> None:
    # A custom tunnel fronts the interception's one fixed port: tool servers on
    # other ports must not be handed its URL.
    config = InterceptionServerConfig(
        tunnel=CustomTunnelConfig(url="http://203.0.113.5:8765", port=8765)
    )
    monkeypatch.setattr(tunnels, "PrimeTunnel", StubTunnel)
    assert isinstance(config.host_tunnel(), StubTunnel)


def test_host_tunnel_may_exit_in_another_context() -> None:
    # GEPA enters and exits `env.serving()` in separate run_until_complete calls.
    tunnel = StubTunnel()

    @contextlib.asynccontextmanager
    async def serving():
        with using_host_tunnel(tunnel):
            yield

    scope = serving()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(scope.__aenter__())
        loop.run_until_complete(scope.__aexit__(None, None, None))
    finally:
        loop.close()
    assert configured_host_tunnel() is None


def test_episode_sees_the_env_tunnel_outside_serving() -> None:
    tunnel = StubTunnel()
    seen = []

    async def episode(task, ctx, on_trace, on_discard):
        seen.append(configured_host_tunnel())
        return "episode"

    env = SimpleNamespace(
        config=SimpleNamespace(
            interception=SimpleNamespace(host_tunnel=lambda: tunnel)
        ),
        _run_episode=episode,
    )
    assert asyncio.run(Env.run_episode(env, object(), object())) == "episode"
    assert seen == [tunnel]
    assert configured_host_tunnel() is None

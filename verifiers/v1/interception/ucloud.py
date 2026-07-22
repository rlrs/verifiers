"""Bridge UCloud's polling relay to the v1 interception server.

UCloud sandboxes do not reach host services through ``prime_tunnel``. They call the
UCloud relay, while the host worker polls that relay and forwards each request into the
local v1 ``InterceptionServer``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal
from urllib.parse import urlsplit

from aiohttp import ClientSession

from verifiers.utils.ucloud_interception_relay import (
    UCloudInterceptionRelayClient,
    ucloud_interception_relay_from_env,
)
from verifiers.v1.interception.base import (
    BaseInterceptionConfig,
    Interception,
    Slot,
)
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession

logger = logging.getLogger(__name__)


class UCloudRelayInterceptionConfig(BaseInterceptionConfig):
    """Route sandbox requests through UCloud's polling relay.

    The relay URL and credentials default to the UCLOUD_RELAY_* environment
    variables consumed by ucloud_interception_relay_from_env.
    """

    type: Literal["ucloud-relay"] = "ucloud-relay"
    url: str | None = None


def _route_from_endpoint(endpoint: str) -> str:
    """Return the local interceptor route for a relay endpoint."""
    parsed = urlsplit(endpoint)
    path = parsed.path if parsed.scheme or parsed.netloc else endpoint
    if not path:
        return "/v1/chat/completions"
    if not path.startswith("/"):
        path = "/" + path
    for suffix in (
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/completions",
        "/v1/messages",
        "/chat/completions",
        "/responses",
        "/completions",
    ):
        if path.endswith(suffix):
            return suffix
    return path


class UCloudInterceptionRelayBridge:
    def __init__(
        self,
        *,
        relay: UCloudInterceptionRelayClient,
        server: InterceptionServer,
        rollout_id: str,
        secret: str,
        poll_timeout: float | None = None,
    ) -> None:
        self.relay = relay
        self.server = server
        self.rollout_id = rollout_id
        self.secret = secret
        self.poll_timeout = (
            float(poll_timeout)
            if poll_timeout is not None
            else float(os.environ.get("UCLOUD_RELAY_BRIDGE_POLL_TIMEOUT_SECONDS", "5"))
        )
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "UCloudInterceptionRelayBridge":
        await self.relay.register_rollout(self.rollout_id, self.secret)
        self._task = asyncio.create_task(self._poll())
        return self

    async def __aexit__(self, *exc) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self.relay.unregister_rollout(self.rollout_id)

    @property
    def endpoint(self) -> str:
        return self.relay.openai_base_url(self.rollout_id)

    @property
    def api_key(self) -> str:
        return self.relay.sandbox_api_key

    async def _poll(self) -> None:
        async with ClientSession() as http:
            while True:
                try:
                    result = await self.relay.poll_request(
                        self.rollout_id, timeout=self.poll_timeout
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "ucloud relay poll failed: rollout_id=%s", self.rollout_id
                    )
                    await asyncio.sleep(1)
                    continue
                if result is None:
                    continue
                request_id, intercept = result
                await self._forward(http, request_id, intercept)

    async def _forward(
        self,
        http: ClientSession,
        request_id: str,
        intercept: dict[str, Any],
    ) -> None:
        body = dict(intercept.get("body") or {})
        if body.get("stream"):
            await self.relay.deliver_error(
                request_id,
                RuntimeError("UCloud v1 relay bridge does not support streaming"),
                status=400,
            )
            return

        route = _route_from_endpoint(str(intercept.get("endpoint") or ""))
        headers = dict(intercept.get("headers") or {})
        headers["Authorization"] = f"Bearer {self.secret}"
        headers.setdefault("Content-Type", "application/json")
        url = f"http://127.0.0.1:{self.server.port}{route}"
        try:
            async with http.post(url, json=body, headers=headers) as response:
                if response.content_type == "application/json":
                    response_body = await response.json()
                else:
                    text = await response.text()
                    response_body = {
                        "error": {
                            "message": text or response.reason or "non-json response"
                        }
                    }
                await self.relay.deliver_response(
                    request_id,
                    response_body,
                    status=response.status,
                )
        except Exception as exc:
            logger.exception(
                "ucloud relay forward failed: rollout_id=%s request_id=%s",
                self.rollout_id,
                request_id,
            )
            with contextlib.suppress(Exception):
                await self.relay.deliver_error(request_id, exc)


class UCloudRelayInterception(Interception):
    """One local interception server with one relay registration per rollout."""

    def __init__(self, config: UCloudRelayInterceptionConfig | None = None) -> None:
        super().__init__()
        self.config = config or UCloudRelayInterceptionConfig()
        self.server = InterceptionServer(requires_tunnel=False)
        self.relay: UCloudInterceptionRelayClient | None = None

    async def start(self) -> None:
        await self.stack.enter_async_context(self.server)
        self.relay = ucloud_interception_relay_from_env(self.config.url)
        self.stack.push_async_callback(self.relay.aclose)

    @asynccontextmanager
    async def acquire(self, session: RolloutSession) -> AsyncIterator[Slot]:
        if self.relay is None:
            raise RuntimeError("UCloud relay interception has not been started")
        secret = self.server.register(session)
        try:
            async with ucloud_relay_bridge(
                relay=self.relay,
                server=self.server,
                rollout_id=session.trace.id,
                secret=secret,
            ) as bridge:
                # RolloutRun appends /v1. The relay SDK's historical template
                # includes that suffix, so normalize it at the Slot boundary.
                yield bridge.endpoint.removesuffix("/v1"), bridge.api_key
        finally:
            self.server.unregister(secret)


@asynccontextmanager
async def ucloud_relay_bridge(
    *,
    relay: UCloudInterceptionRelayClient,
    server: InterceptionServer,
    rollout_id: str,
    secret: str,
):
    async with UCloudInterceptionRelayBridge(
        relay=relay,
        server=server,
        rollout_id=rollout_id,
        secret=secret,
    ) as bridge:
        yield bridge

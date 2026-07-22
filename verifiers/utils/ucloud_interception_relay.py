import asyncio
import os
import socket
import uuid
from typing import Any, Mapping


def _require_ucloud_relay_sdk() -> tuple[Any, Any]:
    try:
        from ucloud_sandboxes_sdk import (  # type: ignore[import-not-found]
            AsyncRelayWorkerClient,
            RelayApiError,
        )
    except ImportError as exc:
        raise ImportError(
            "UCloud interception relay requires `ucloud-sandboxes-sdk[async]`. "
            "Install the latest SDK in the LUMI overlay."
        ) from exc
    return AsyncRelayWorkerClient, RelayApiError


def _protocol_from_endpoint(endpoint: str) -> str:
    if endpoint.endswith("/v1/messages"):
        return "anthropic_messages"
    if endpoint.endswith("/v1/responses"):
        return "openai_responses"
    if endpoint.endswith("/v1/completions"):
        return "openai_completions"
    return "openai_chat_completions"


def _filtered_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        str(key).lower(): str(value)
        for key, value in headers.items()
        if str(key).lower() not in {"authorization", "x-api-key"}
    }


def _request_body(relay_request: Any) -> dict[str, Any]:
    body = getattr(relay_request, "body", {})
    return dict(body) if isinstance(body, dict) else {}


class UCloudInterceptionRelayClient:
    """Client for the UCloud-hosted model relay.

    Sandboxes call the relay's OpenAI-compatible URL with a low-privilege
    sandbox token. The LUMI rollout worker uses this client with the worker
    token to register rollouts, poll leased requests, renew leases during
    inference, and post responses.
    """

    def __init__(
        self,
        *,
        base_url: str,
        worker_token: str | None = None,
        sandbox_token: str | None = None,
        openai_base_url_template: str | None = None,
        worker_id: str | None = None,
        poll_limit: int = 1,
        lease_seconds: float = 900.0,
        renew_interval_seconds: float | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        AsyncRelayWorkerClient, _RelayApiError = _require_ucloud_relay_sdk()
        self.base_url = base_url.rstrip("/")
        self.sandbox_api_key = (
            sandbox_token
            or os.environ.get("UCLOUD_RELAY_SANDBOX_TOKEN")
            or "intercepted"
        )
        self.openai_base_url_template = openai_base_url_template or os.environ.get(
            "UCLOUD_RELAY_OPENAI_BASE_URL_TEMPLATE",
            f"{self.base_url}/rollouts/{{rollout_id}}/v1",
        )
        self.worker_id = worker_id or os.environ.get(
            "UCLOUD_RELAY_WORKER_ID",
            f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}",
        )
        self.poll_limit = max(1, int(poll_limit))
        self.lease_seconds = float(lease_seconds)
        self.renew_interval_seconds = (
            float(renew_interval_seconds)
            if renew_interval_seconds is not None
            else max(5.0, self.lease_seconds / 3.0)
        )
        self.client = AsyncRelayWorkerClient(
            self.base_url,
            worker_token=worker_token or os.environ.get("UCLOUD_RELAY_WORKER_TOKEN"),
            timeout_seconds=timeout_seconds,
        )
        self._requests: dict[str, Any] = {}
        self._renew_tasks: dict[str, asyncio.Task[None]] = {}

    def openai_base_url(self, rollout_id: str) -> str:
        return self.openai_base_url_template.format(rollout_id=rollout_id).rstrip("/")

    async def register_rollout(self, rollout_id: str, secret: str | None = None) -> None:
        metadata = {"worker_id": self.worker_id}
        await self.client.register_rollout(rollout_id, metadata=metadata)
        await self.client.heartbeat(rollout_id, self.worker_id, metadata=metadata)

    async def unregister_rollout(self, rollout_id: str) -> None:
        for request_id, relay_request in list(self._requests.items()):
            if str(getattr(relay_request, "rollout_id", "")) == str(rollout_id):
                self._cancel_renewal(request_id)
                self._requests.pop(request_id, None)
        await self.client.unregister_rollout(rollout_id)

    async def poll_request(
        self,
        rollout_id: str,
        *,
        timeout: float,
    ) -> tuple[str, dict[str, Any]] | None:
        result = await self.client.poll(
            rollout_id,
            worker_id=self.worker_id,
            timeout_seconds=timeout,
            limit=self.poll_limit,
            lease_seconds=self.lease_seconds,
        )
        relay_request = getattr(result, "request", None)
        if relay_request is None:
            return None

        request_id = str(getattr(relay_request, "request_id"))
        self._requests[request_id] = relay_request
        self._start_renewal(relay_request)
        return request_id, self._to_intercept(relay_request)

    async def deliver_response(
        self,
        request_id: str,
        body: dict[str, Any],
        *,
        status: int = 200,
    ) -> None:
        relay_request = self._requests.pop(request_id, None)
        self._cancel_renewal(request_id)
        if relay_request is None:
            raise RuntimeError(f"UCloud relay request {request_id!r} was missing")
        await self.client.respond(
            request_id,
            str(getattr(relay_request, "lease_id")),
            body,
            status=status,
        )

    async def deliver_error(
        self,
        request_id: str,
        error: BaseException,
        *,
        status: int = 500,
    ) -> None:
        relay_request = self._requests.pop(request_id, None)
        self._cancel_renewal(request_id)
        if relay_request is None:
            raise RuntimeError(f"UCloud relay request {request_id!r} was missing")
        await self.client.error(
            request_id,
            str(getattr(relay_request, "lease_id")),
            f"{type(error).__name__}: {error}",
            status=status,
        )

    async def aclose(self) -> None:
        for request_id in list(self._renew_tasks):
            self._cancel_renewal(request_id)
        await self.client.close()

    def _to_intercept(self, relay_request: Any) -> dict[str, Any]:
        body = _request_body(relay_request)
        endpoint = str(getattr(relay_request, "endpoint", ""))
        headers = getattr(relay_request, "headers", {})
        headers = headers if isinstance(headers, Mapping) else {}
        return {
            "request_id": str(getattr(relay_request, "request_id")),
            "rollout_id": str(getattr(relay_request, "rollout_id")),
            "endpoint": endpoint,
            "body": body,
            "protocol": _protocol_from_endpoint(endpoint),
            "messages": body.get("messages"),
            "prompt": body.get("prompt"),
            "input": body.get("input"),
            "system": body.get("system"),
            "model": body.get("model"),
            "tools": body.get("tools"),
            "stream": bool(body.get("stream", False)),
            "chunk_queue": None,
            "response_future": None,
            "headers": _filtered_headers(headers),
        }

    def _start_renewal(self, relay_request: Any) -> None:
        request_id = str(getattr(relay_request, "request_id"))
        self._cancel_renewal(request_id)
        self._renew_tasks[request_id] = asyncio.create_task(
            self._renew_until_done(request_id)
        )

    def _cancel_renewal(self, request_id: str) -> None:
        task = self._renew_tasks.pop(request_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def _renew_until_done(self, request_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(self.renew_interval_seconds)
                relay_request = self._requests.get(request_id)
                if relay_request is None:
                    return
                renewed = await self.client.renew_request(
                    relay_request,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                )
                self._requests[request_id] = renewed
        except asyncio.CancelledError:
            return


def ucloud_interception_relay_from_env(
    interception_url: str | None = None,
) -> UCloudInterceptionRelayClient:
    base_url = (
        interception_url
        or os.environ.get("UCLOUD_RELAY_URL")
        or os.environ.get("VF_UCLOUD_RELAY_URL")
        or os.environ.get("VF_RELAY_URL")
    )
    if not base_url:
        raise RuntimeError(
            "UCloud relay requires UCLOUD_RELAY_URL, VF_UCLOUD_RELAY_URL, "
            "VF_RELAY_URL, or interception_url."
        )
    return UCloudInterceptionRelayClient(
        base_url=base_url,
        worker_token=os.environ.get("UCLOUD_RELAY_WORKER_TOKEN"),
        sandbox_token=os.environ.get("UCLOUD_RELAY_SANDBOX_TOKEN"),
        openai_base_url_template=os.environ.get(
            "UCLOUD_RELAY_OPENAI_BASE_URL_TEMPLATE"
        ),
        poll_limit=int(os.environ.get("UCLOUD_RELAY_POLL_LIMIT", "1")),
        lease_seconds=float(os.environ.get("UCLOUD_RELAY_LEASE_SECONDS", "900")),
        renew_interval_seconds=(
            float(os.environ["UCLOUD_RELAY_RENEW_INTERVAL_SECONDS"])
            if os.environ.get("UCLOUD_RELAY_RENEW_INTERVAL_SECONDS")
            else None
        ),
        timeout_seconds=float(os.environ.get("UCLOUD_RELAY_TIMEOUT_SECONDS", "30")),
    )

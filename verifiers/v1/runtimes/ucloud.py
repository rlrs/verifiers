"""Remote runtime backed by a UCloud sandbox gateway."""

import asyncio
import contextlib
import logging
import os
import re
import shlex
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, ClassVar, Literal, cast

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import BaseRuntimeInfo, ProgramResult, Runtime
from verifiers.v1.runtimes.limiters import creation_limiter

logger = logging.getLogger(__name__)

TRANSFER_DIR = "/workspace/.vf-transfers"
_sandbox_semaphores: dict[int, asyncio.Semaphore] = {}

if TYPE_CHECKING:
    from ucloud_sandboxes_sdk import AsyncSandboxClient, AsyncSandboxHandle


class UCloudConfig(BaseConfig):
    type: Literal["ucloud"] = "ucloud"
    image: str = "python:3.12-slim"
    workdir: str = "/app"
    network: Literal["none", "bridge", "host"] = "bridge"
    cpu: float = 1.0
    memory: float = 2.0
    disk: float = 5.0
    tmpfs_mb: int = 64
    ttl_seconds: int = 3600
    command: list[str] = Field(default_factory=lambda: ["sleep", "infinity"])
    name_prefix: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    user: str | None = None
    base_url: str | None = None
    request_timeout_seconds: float = 30.0
    start_timeout_seconds: float = 1800.0
    retry_interval_seconds: float = 10.0
    creates_per_min: int | None = None
    max_concurrent_sandboxes: int | None = Field(128, gt=0)


class UCloudRuntimeInfo(UCloudConfig, BaseRuntimeInfo):
    pass


class UCloudRuntime(Runtime):
    is_local: ClassVar[bool] = False

    def __init__(self, config: UCloudConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = UCloudRuntimeInfo(**config.model_dump())
        self._client: AsyncSandboxClient | None = None
        self._sandbox: AsyncSandboxHandle | None = None
        self._capacity_acquired = False
        self._started_at = time.monotonic()
        self._timings: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def _add_timing(self, name: str, started_at: float) -> None:
        self._timings[name] = self._timings.get(name, 0.0) + time.monotonic() - started_at

    def _increment(self, name: str) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1

    def _base_url(self) -> str:
        url = self.config.base_url or os.environ.get("UCLOUD_SANDBOX_URL") or os.environ.get("UCLOUD_SANDBOX_API_URL")
        if url is None:
            raise SandboxError("ucloud sandbox URL is required; set runtime.base_url or UCLOUD_SANDBOX_URL")
        return url

    def _require_sandbox(self) -> "AsyncSandboxHandle":
        if self._sandbox is None:
            raise SandboxError("ucloud sandbox has not been started")
        return self._sandbox

    def _sandbox_id(self) -> str:
        if self.config.name_prefix is None:
            return self.name
        prefix = re.sub(r"[^a-zA-Z0-9_-]+", "-", self.config.name_prefix).strip("-_")
        return f"{prefix}-{self.name}" if prefix else self.name

    async def _acquire_capacity(self) -> None:
        limit = self.config.max_concurrent_sandboxes
        if limit is None:
            return
        semaphore = _sandbox_semaphores.setdefault(limit, asyncio.Semaphore(limit))
        await semaphore.acquire()
        self._capacity_acquired = True

    def _release_capacity(self) -> None:
        if not self._capacity_acquired:
            return
        limit = self.config.max_concurrent_sandboxes
        assert limit is not None
        _sandbox_semaphores[limit].release()
        self._capacity_acquired = False

    async def start(self) -> None:
        from aiohttp import ClientError
        from ucloud_sandboxes_sdk import (
            AsyncSandboxClient,
            Image,
            SandboxApiError,
            SandboxFilesystemSpec,
            SandboxSecuritySpec,
        )

        client = AsyncSandboxClient(
            self._base_url(),
            api_token=os.environ.get("UCLOUD_SANDBOX_API_TOKEN"),
            timeout_seconds=self.config.request_timeout_seconds,
        )
        self._client = client
        try:
            capacity_started_at = time.monotonic()
            await self._acquire_capacity()
            self._add_timing("capacity_wait", capacity_started_at)
            limiter_started_at = time.monotonic()
            async with (
                creation_limiter((self.config.creates_per_min or 0) / 60, "ucloud-sandbox") or contextlib.nullcontext()
            ):
                self._add_timing("creation_limiter_wait", limiter_started_at)
                provision_started_at = time.monotonic()
                deadline = time.monotonic() + self.config.start_timeout_seconds
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for UCloud sandbox node readiness")
                    try:
                        sandbox = await client.create_sandbox(
                            id=self._sandbox_id(),
                            image=Image.from_registry(self.config.image),
                            command=self.config.command,
                            working_dir=self.config.workdir,
                            cpus=self.config.cpu,
                            memory_mb=round(self.config.memory * 1024),
                            disk_mb=round(self.config.disk * 1024),
                            network=self.config.network,
                            ttl_seconds=self.config.ttl_seconds,
                            labels=self.config.labels,
                            filesystem=SandboxFilesystemSpec(tmpfs_mb=self.config.tmpfs_mb).to_dict(),
                            security=SandboxSecuritySpec(user=self.config.user).to_dict(),
                            request_timeout_seconds=min(self.config.request_timeout_seconds, remaining),
                        )
                        break
                    except SandboxApiError as e:
                        body = cast(dict[str, object], e.body) if isinstance(e.body, dict) else {}
                        message = str(body.get("error") or "").lower()
                        pending = (
                            body.get("retryable") is True
                            or "pending_resources" in body
                            or "pending_image_builds" in body
                            or "no ready node" in message
                            or "no ready builder" in message
                        )
                        if e.status_code != 503 or not pending:
                            raise
                        self._increment("provision_retries")
                    except (ClientError, TimeoutError):
                        self._increment("provision_retries")
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(min(self.config.retry_interval_seconds, remaining))
            self._add_timing("provision", provision_started_at)
            self._sandbox = sandbox
            self.info.id = sandbox.id
            bootstrap_started_at = time.monotonic()
            result = await sandbox.exec(
                ["mkdir", "-p", self.config.workdir, TRANSFER_DIR],
                timeout_seconds=self.config.request_timeout_seconds,
            )
            if not result.success:
                raise SandboxError(f"failed to create workdir: {result.stderr.strip()}")
            self._add_timing("bootstrap", bootstrap_started_at)
            self._add_timing("start", self._started_at)
            self._increment("sandboxes")
            logger.info("ucloud: sandbox %s up (image=%s)", sandbox.id, self.config.image)
        except Exception as e:
            await self.teardown()
            raise SandboxError(f"ucloud sandbox provisioning failed: {e}") from e

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        started_at = time.monotonic()
        try:
            result = await self._require_sandbox().exec(
                argv,
                env=env,
                working_dir=self.config.workdir,
            )
        except Exception as e:
            raise SandboxError(f"ucloud exec failed: {e}") from e
        finally:
            self._increment("execs")
            self._add_timing("exec", started_at)
        return ProgramResult(
            exit_code=result.exit_code if result.exit_code is not None else 1,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        started_at = time.monotonic()
        try:
            return await self.run(argv, env)
        finally:
            self._increment("programs")
            self._add_timing("program", started_at)

    async def run_background(self, argv: list[str], env: dict[str, str], log: str) -> None:
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise SandboxError(f"ucloud background launch failed: {result.stderr.strip()}")

    @asynccontextmanager
    async def relay_endpoint(self, port: int, secret: str):
        from aiohttp import ClientSession
        from ucloud_sandboxes_sdk import AsyncRelayWorkerClient

        relay_url = os.environ.get("UCLOUD_SANDBOX_RELAY_URL")
        relay_token = os.environ.get("UCLOUD_SANDBOX_RELAY_TOKEN")
        worker_token = os.environ.get("UCLOUD_SANDBOX_RELAY_WORKER_TOKEN")
        if not relay_url or not relay_token or not worker_token:
            raise SandboxError(
                "ucloud relay requires UCLOUD_SANDBOX_RELAY_URL, "
                "UCLOUD_SANDBOX_RELAY_TOKEN, and "
                "UCLOUD_SANDBOX_RELAY_WORKER_TOKEN"
            )
        rollout_id = self.name
        tunnel_id = f"{rollout_id}-state"
        worker_id = f"vf-{uuid.uuid4().hex[:12]}"
        client = AsyncRelayWorkerClient(relay_url, worker_token=worker_token)
        session = ClientSession()

        async def forward() -> None:
            while True:
                poll = await client.poll(
                    rollout_id,
                    worker_id=worker_id,
                    timeout_seconds=20,
                    limit=1,
                    lease_seconds=600,
                )
                for request in poll.requests:
                    endpoint = request.endpoint
                    if endpoint in {"/v1/task", "/v1/state"}:
                        endpoint = endpoint.removeprefix("/v1")
                    headers = {
                        key: value
                        for key, value in request.headers.items()
                        if key.lower() not in {"authorization", "content-length", "host"}
                    }
                    headers["Authorization"] = f"Bearer {secret}"
                    try:
                        async with session.request(
                            request.method,
                            f"http://127.0.0.1:{port}{endpoint}",
                            headers=headers,
                            json=request.body,
                        ) as response:
                            body = await response.json()
                            await client.respond_to(
                                request,
                                body,
                                status=response.status,
                                headers={"Content-Type": "application/json"},
                            )
                    except Exception as e:
                        await client.error_request(request, str(e))

        async def forward_state() -> None:
            while True:
                poll = await client.poll(
                    tunnel_id,
                    worker_id=worker_id,
                    timeout_seconds=20,
                    limit=1,
                    lease_seconds=600,
                )
                for request in poll.requests:
                    await client.forward_to(
                        request,
                        f"http://127.0.0.1:{port}",
                        timeout_seconds=self.config.request_timeout_seconds,
                    )

        registration = await client.register_rollout(rollout_id, metadata={"runtime": "verifiers"})
        rollout = registration.get("rollout")
        registration_token = rollout.get("registration_token") if isinstance(rollout, dict) else None
        if not isinstance(registration_token, str):
            raise SandboxError("ucloud relay registration did not return a token")
        await client.register_tunnel(tunnel_id, metadata={"runtime": "verifiers"})
        model_task = asyncio.create_task(forward())
        state_task = asyncio.create_task(forward_state())
        try:
            yield (
                f"{relay_url.rstrip('/')}/rollouts/{rollout_id}/v1",
                relay_token,
                f"{relay_url.rstrip('/')}/tunnels/{tunnel_id}",
                relay_token,
            )
        finally:
            model_task.cancel()
            state_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await model_task
            with contextlib.suppress(asyncio.CancelledError):
                await state_task
            with contextlib.suppress(Exception):
                await client.unregister_tunnel(tunnel_id)
            with contextlib.suppress(Exception):
                await client.unregister_rollout(rollout_id)
            await session.close()
            await client.close()

    def _path(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return f"{self.config.workdir.rstrip('/')}/{path}"

    async def read(self, path: str) -> bytes:
        source = self._path(path)
        staged = f"{TRANSFER_DIR}/{uuid.uuid4().hex}"
        try:
            result = await self.run(["cp", source, staged], {})
            if result.exit_code != 0:
                raise SandboxError(f"failed to stage {path!r} for download: {result.stderr.strip()}")
            return await self._require_sandbox().download_file(staged)
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}") from e
        finally:
            with contextlib.suppress(Exception):
                await self.run(["rm", "-f", staged], {})

    async def write(self, path: str, data: bytes) -> None:
        target = self._path(path)
        staged = f"{TRANSFER_DIR}/{uuid.uuid4().hex}"
        parent = str(PurePosixPath(target).parent)
        result = await self.run(["mkdir", "-p", parent], {})
        if result.exit_code != 0:
            raise SandboxError(f"failed to create parent directory for {path!r}: {result.stderr.strip()}")
        try:
            await self._require_sandbox().upload_file(staged, data)
            result = await self.run(["mv", "-f", staged, target], {})
            if result.exit_code != 0:
                raise SandboxError(f"failed to move staged upload to {path!r}: {result.stderr.strip()}")
        except Exception as e:
            raise SandboxError(f"write {path!r}: {e}") from e
        finally:
            with contextlib.suppress(Exception):
                await self.run(["rm", "-f", staged], {})

    def cleanup(self) -> None:
        from ucloud_sandboxes_sdk import SandboxClient

        with contextlib.suppress(Exception):
            SandboxClient(
                self._base_url(),
                api_token=os.environ.get("UCLOUD_SANDBOX_API_TOKEN"),
                timeout_seconds=self.config.request_timeout_seconds,
            ).delete_sandbox(self.info.id or self._sandbox_id())

    async def teardown(self) -> None:
        client, self._client = self._client, None
        teardown_started_at = time.monotonic()
        sandbox, self._sandbox = self._sandbox, None
        try:
            if client is not None and sandbox is not None:
                await sandbox.delete()
            elif client is not None:
                await client.delete_sandbox(self.info.id or self._sandbox_id())
        except Exception as e:
            if client is not None:
                logger.warning("ucloud: failed to delete sandbox %s: %s", self.info.id, e)
        finally:
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.close()
            self._release_capacity()
        self._add_timing("teardown", teardown_started_at)
        timings = " ".join(f"{name}={value:.1f}s" for name, value in sorted(self._timings.items()))
        counts = " ".join(f"{name}={value}" for name, value in sorted(self._counts.items()))
        logger.info(
            "ucloud: sandbox %s lifecycle total=%.1fs %s %s",
            self.info.id,
            time.monotonic() - self._started_at,
            timings,
            counts,
        )

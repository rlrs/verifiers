"""Remote UCloud sandbox runtime backed by ucloud-sandboxes-sdk."""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
from pathlib import PurePosixPath
from typing import Any, ClassVar, Literal

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.utils.ucloud_sandbox_client import UCloudSandboxClient
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import BaseRuntimeInfo, ProgramResult, Runtime, parse_gpu
from verifiers.v1.runtimes.limiters import creation_limiter

logger = logging.getLogger(__name__)


class UCloudConfig(BaseConfig):
    type: Literal["ucloud"] = "ucloud"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    network_access: bool = True
    guaranteed: bool = False
    region: str | None = None
    labels: list[str] = Field(default_factory=list)
    cpu: float = 1.0
    memory: float = 2.0
    gpu: str | None = None
    disk: float = 5.0
    creates_per_min: int | None = None
    request_timeout_seconds: float | None = None
    create_timeout_seconds: float | None = None
    retry_interval_seconds: float | None = None
    prepare_id: str | None = None
    profile: str | None = None
    user: str | None = None
    enable_cron: bool | None = None
    enable_sshd: bool | None = None
    keep_alive: bool | None = None
    writable_paths: list[str] | None = None
    advanced_configs: dict[str, Any] | None = None


class UCloudRuntimeInfo(UCloudConfig, BaseRuntimeInfo):
    pass


class UCloudRuntime(Runtime):
    is_local: ClassVar[bool] = False

    def __init__(self, config: UCloudConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = UCloudRuntimeInfo(**config.model_dump())
        self._client: UCloudSandboxClient | None = None

    async def start(self) -> None:
        gpu_type, gpu_count = parse_gpu(self.config.gpu)
        if gpu_count or gpu_type:
            raise SandboxError("ucloud runtime supports CPU-only sandboxes")
        self._client = UCloudSandboxClient(
            request_timeout_seconds=self.config.request_timeout_seconds,
            create_timeout_seconds=self.config.create_timeout_seconds,
            retry_interval_seconds=self.config.retry_interval_seconds,
        )
        request = {
            "name": self.name,
            "labels": self.config.labels,
            "docker_image": self.config.image,
            "network_access": self.config.network_access,
            "cpu_cores": self.config.cpu,
            "memory_gb": self.config.memory,
            "disk_size_gb": self.config.disk,
            "timeout_minutes": 24 * 60,
            "working_dir": self.config.workdir,
            "guaranteed": self.config.guaranteed,
            "region": self.config.region,
            "prepare_id": self.config.prepare_id,
            "profile": self.config.profile,
            "user": self.config.user,
            "enable_cron": self.config.enable_cron,
            "enable_sshd": self.config.enable_sshd,
            "keep_alive": self.config.keep_alive,
            "writable_paths": self.config.writable_paths,
        }
        if self.config.advanced_configs is not None:
            request["advanced_configs"] = self.config.advanced_configs
        try:
            async with (
                creation_limiter(
                    (self.config.creates_per_min or 0) / 60, "ucloud-sandbox"
                )
                or contextlib.nullcontext()
            ):
                sandbox = await self._client.create(request)
            self.info.id = sandbox.id
            await self._client.wait_for_creation(self.info.id)
            logger.info(
                "ucloud: sandbox %s up (image=%s)", self.info.id, self.config.image
            )
            await self._client.execute_command(
                self.info.id,
                f"mkdir -p {shlex.quote(self.config.workdir)}",
                attempts=10,
                retry_interval_seconds=1,
            )
        except Exception as e:
            raise SandboxError(f"ucloud sandbox provisioning failed: {e}") from e

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        try:
            result = await self._client.execute_command(
                self.info.id,
                shlex.join(argv),
                working_dir=self.config.workdir,
                env=env,
            )
        except Exception as e:
            raise SandboxError(f"ucloud exec failed: {e}") from e
        return ProgramResult(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise SandboxError(f"ucloud background launch failed: {result.stderr.strip()}")

    async def read(self, path: str) -> bytes:
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        try:
            data = await self._client._read_bytes(self.info.id, target)
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}") from e
        return data

    async def write(self, path: str, data: bytes) -> None:
        target = (
            path
            if path.startswith("/")
            else f"{self.config.workdir.rstrip('/')}/{path}"
        )
        try:
            await self._client.upload_bytes(
                self.info.id, target, data, filename=PurePosixPath(target).name
            )
        except Exception as e:
            raise SandboxError(f"write {path!r}: {e}") from e

    def cleanup(self) -> None:
        if self.info.id is None:
            return
        try:
            from ucloud_sandboxes_sdk import SandboxClient

            base_url = (
                os.environ.get("UCLOUD_SANDBOX_API_URL")
                or os.environ.get("UCLOUD_SANDBOX_URL")
                or os.environ.get("UCLOUD_SANDBOX_BASE_URL")
            )
            if not base_url:
                return
            client = SandboxClient(
                base_url,
                api_token=os.environ.get("UCLOUD_SANDBOX_API_TOKEN"),
                timeout_seconds=float(
                    self.config.request_timeout_seconds
                    or os.environ.get("UCLOUD_SANDBOX_REQUEST_TIMEOUT_SECONDS", "300")
                ),
            )
            with contextlib.suppress(Exception):
                client.delete_sandbox(self.info.id)
        except Exception:
            logger.exception("ucloud: synchronous cleanup failed for %s", self.info.id)

    async def teardown(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        if self.info.id is not None:
            try:
                await client.delete(self.info.id)
            except Exception as e:
                logger.warning(
                    "ucloud: failed to delete sandbox %s: %s", self.info.id, e
                )
        with contextlib.suppress(Exception):
            await client.aclose()

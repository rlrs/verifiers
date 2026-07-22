import asyncio
import base64
import logging
import math
import os
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from prime_sandboxes import CommandTimeoutError, SandboxFileNotFoundError


@dataclass
class UCloudSandboxHandle:
    id: str


@dataclass
class UCloudCommandResponse:
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class UCloudFileUploadResponse:
    success: bool
    path: str
    size: int


@dataclass
class UCloudFileReadResponse:
    content: str


@dataclass
class UCloudBackgroundJob:
    sandbox_id: str
    job_id: str


@dataclass
class UCloudBackgroundJobStatus:
    completed: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""


def _require_ucloud_sdk() -> tuple[Any, Any]:
    try:
        from ucloud_sandboxes_sdk import (  # type: ignore[import-not-found]
            AsyncSandboxClient,
            Image,
            SandboxApiError,
        )
    except ImportError as exc:
        raise ImportError(
            "UCloud sandbox backend requires `ucloud-sandboxes-sdk[async]`. "
            "Install it or add the GitHub package to the LUMI overlay."
        ) from exc
    return AsyncSandboxClient, Image, SandboxApiError


def _object_data(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if item is not None}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(exclude_none=True))
    data: dict[str, Any] = {}
    for key in (
        "network",
        "security",
        "filesystem",
        "ssh",
        "working_dir",
        "registry_credentials_id",
        "guaranteed",
        "idempotency_key",
        "region",
        "prepare_id",
        "profile",
        "user",
        "enable_cron",
        "enable_sshd",
        "keep_alive",
        "writable_paths",
    ):
        if hasattr(value, key):
            item = getattr(value, key)
            if item is not None:
                data[key] = item
    return data


def _request_data(request: object) -> dict[str, Any]:
    if isinstance(request, dict):
        data = dict(request)
    else:
        model_dump = getattr(request, "model_dump", None)
        if callable(model_dump):
            data = dict(model_dump(exclude_none=True))
        else:
            data = {}
            for key in (
                "name",
                "docker_image",
                "image",
                "cpu_cores",
                "memory_gb",
                "disk_size_gb",
                "gpu_count",
                "gpu_type",
                "network_access",
                "environment_vars",
                "env_vars",
                "secrets",
                "labels",
                "timeout_minutes",
                "start_command",
                "working_dir",
                "security",
                "filesystem",
                "ssh",
                "network",
                "prepare_id",
                "profile",
                "user",
                "enable_cron",
                "enable_sshd",
                "keep_alive",
                "writable_paths",
                "advanced_configs",
            ):
                if hasattr(request, key):
                    value = getattr(request, key)
                    if value is not None:
                        data[key] = value
    advanced_configs = _object_data(data.pop("advanced_configs", None))
    for key, value in advanced_configs.items():
        data.setdefault(key, value)
    return data


def _ceil_int(value: object, default: int) -> int:
    if value is None:
        return default
    return max(1, math.ceil(float(value)))


def _labels(value: object) -> dict[str, str]:
    labels = {"created_by": "verifiers", "sandbox_backend": "ucloud"}
    if isinstance(value, dict):
        labels.update({str(key): str(item) for key, item in value.items()})
    elif isinstance(value, list):
        labels.update({str(item): "true" for item in value})
    return labels


def _env_vars(data: dict[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for key in ("environment_vars", "env_vars", "secrets"):
        value = data.get(key)
        if isinstance(value, dict):
            merged.update({str(k): str(v) for k, v in value.items()})
    return merged


def _parent_dir(path: str) -> str | None:
    parent = str(PurePosixPath(path).parent)
    return parent if parent and parent not in {".", "/"} else None


def _shell_command(command: str, working_dir: str | None = None) -> list[str]:
    if working_dir:
        command = f"cd {shlex.quote(working_dir)} && {command}"
    return ["sh", "-lc", command]


def _sdk_image(image: str, image_cls: Any) -> Any:
    from_registry = getattr(image_cls, "from_registry", None)
    if callable(from_registry):
        return from_registry(image)
    return image


def _is_not_found_error(error: BaseException, api_error_type: Any) -> bool:
    if not isinstance(error, api_error_type):
        return False
    return int(getattr(error, "status_code", 0) or 0) == 404


def _is_capacity_pending_error(error: BaseException, api_error_type: Any) -> bool:
    if not isinstance(error, api_error_type):
        return False
    if int(getattr(error, "status_code", 0) or 0) != 503:
        return False
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return True
    message = str(body.get("error") or "").lower()
    return "no ready node" in message or "pending" in body


def _error_message(error: BaseException) -> str:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        return f"{body.get('error') or ''} {body}".lower()
    return str(error).lower()


def _is_retryable_create_error(error: BaseException, api_error_type: Any) -> bool:
    if isinstance(error, TimeoutError):
        return True
    error_type = type(error).__name__
    if error_type in {"ServerDisconnectedError", "ClientConnectionError"}:
        return True
    if _is_capacity_pending_error(error, api_error_type):
        return True
    if not isinstance(error, api_error_type):
        return False
    status_code = int(getattr(error, "status_code", 0) or 0)
    if status_code not in {502, 503, 504}:
        return False
    message = _error_message(error)
    if ("image is not available" in message or "pull failed" in message) and not (
        "timed out" in message
        or "timeout" in message
        or "node request failed" in message
    ):
        return False
    return (
        "timed out" in message
        or "timeout" in message
        or "node request failed" in message
        or "server disconnected" in message
        or "bad gateway" in message
        or status_code in {503, 504}
    )


def _is_retryable_exec_error(error: BaseException, api_error_type: Any) -> bool:
    if isinstance(error, TimeoutError):
        return False
    if not isinstance(error, api_error_type):
        return False
    status_code = int(getattr(error, "status_code", 0) or 0)
    message = _error_message(error)
    if status_code == 404 and "sandbox route not found" in message:
        return True
    if status_code not in {502, 503, 504}:
        return False
    return (
        "remote end closed" in message
        or "node request failed" in message
        or "timed out" in message
        or "timeout" in message
        or "bad gateway" in message
    )


def _terminal_state(record: Mapping[str, Any]) -> str | None:
    for key in ("state", "status", "phase"):
        value = record.get(key)
        if isinstance(value, str):
            return value.lower()
    status = record.get("status")
    if isinstance(status, dict):
        for key in ("state", "status", "phase"):
            value = status.get(key)
            if isinstance(value, str):
                return value.lower()
    return None


def _resource_summary(data: Mapping[str, Any], image: str) -> dict[str, Any]:
    labels = data.get("labels") or []
    if isinstance(labels, str):
        labels = [labels]
    elif not isinstance(labels, list):
        labels = []
    return {
        "image": image,
        "cpus": float(data.get("cpu_cores") or 1),
        "memory_mb": _ceil_int(data.get("memory_gb"), 2) * 1024,
        "disk_mb": _ceil_int(data.get("disk_size_gb"), 5) * 1024,
        "labels": [str(label) for label in labels],
    }


def _default_security() -> dict[str, Any]:
    return {
        "user": None,
        "cap_drop": [],
        "cap_add": [],
        "no_new_privileges": False,
        "pids_limit": None,
        "read_only_rootfs": False,
        "init": True,
    }


class UCloudSandboxClient:
    """Prime-sandboxes-shaped async client backed by UCloud sandbox gateways."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_token: str | None = None,
        headers: Mapping[str, str] | None = None,
        request_timeout_seconds: float | None = None,
        create_timeout_seconds: float | None = None,
        retry_interval_seconds: float | None = None,
        **_: object,
    ) -> None:
        AsyncSandboxClient, Image, SandboxApiError = _require_ucloud_sdk()
        resolved_base_url = (
            base_url
            or os.environ.get("UCLOUD_SANDBOX_API_URL")
            or os.environ.get("UCLOUD_SANDBOX_URL")
            or os.environ.get("UCLOUD_SANDBOX_BASE_URL")
        )
        if not resolved_base_url:
            raise RuntimeError(
                "UCloud sandbox backend requires UCLOUD_SANDBOX_API_URL, "
                "UCLOUD_SANDBOX_URL, or UCLOUD_SANDBOX_BASE_URL."
            )
        resolved_api_token = api_token or os.environ.get("UCLOUD_SANDBOX_API_TOKEN")
        resolved_headers: dict[str, str] = {}
        if headers:
            resolved_headers.update({str(k): str(v) for k, v in headers.items()})
        self.request_timeout_seconds = float(
            request_timeout_seconds
            if request_timeout_seconds is not None
            else os.environ.get("UCLOUD_SANDBOX_REQUEST_TIMEOUT_SECONDS", "300")
        )
        self.client = AsyncSandboxClient(
            resolved_base_url,
            timeout_seconds=self.request_timeout_seconds,
            api_token=resolved_api_token,
            headers=resolved_headers,
        )
        self._image_cls = Image
        self._api_error_type = SandboxApiError
        self._sandboxes: dict[str, Any] = {}
        self._jobs: dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.create_timeout_seconds = float(
            create_timeout_seconds
            if create_timeout_seconds is not None
            else os.environ.get("UCLOUD_SANDBOX_START_TIMEOUT_SECONDS", "900")
        )
        self.retry_interval_seconds = float(
            retry_interval_seconds
            if retry_interval_seconds is not None
            else os.environ.get("UCLOUD_SANDBOX_RETRY_INTERVAL_SECONDS", "20")
        )

    async def create(self, request: object) -> UCloudSandboxHandle:
        data = _request_data(request)
        if int(data.get("gpu_count") or 0) or data.get("gpu_type"):
            raise ValueError("UCloud sandbox backend supports CPU-only sandboxes.")

        sandbox_id = str(data.get("name") or f"vf-ucloud-{uuid.uuid4().hex[:8]}")
        start_command = str(data.get("start_command") or "tail -f /dev/null")
        image = str(data.get("docker_image") or data.get("image") or "python:3.12-slim")
        timeout_minutes = _ceil_int(data.get("timeout_minutes"), 5)
        summary = _resource_summary(data, image)

        kwargs: dict[str, Any] = {
            "id": sandbox_id,
            "image": _sdk_image(image, self._image_cls),
            "command": _shell_command(start_command),
            "working_dir": data.get("working_dir"),
            "env": _env_vars(data),
            "cpus": float(data.get("cpu_cores") or 1),
            "memory_mb": _ceil_int(data.get("memory_gb"), 2) * 1024,
            "disk_mb": _ceil_int(data.get("disk_size_gb"), 5) * 1024,
            "network": data.get("network")
            or ("bridge" if bool(data.get("network_access", True)) else "none"),
            "ttl_seconds": timeout_minutes * 60,
            "labels": _labels(data.get("labels")),
            "ssh": data.get("ssh", False),
            "security": data.get("security", _default_security()),
            "filesystem": data.get("filesystem"),
        }
        for key in (
            "registry_credentials_id",
            "guaranteed",
            "idempotency_key",
            "region",
            "prepare_id",
            "profile",
            "user",
            "enable_cron",
            "enable_sshd",
            "keep_alive",
            "writable_paths",
        ):
            if key in data:
                kwargs[key] = data[key]
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        deadline = asyncio.get_running_loop().time() + self.create_timeout_seconds
        while True:
            started = time.perf_counter()
            self.logger.info(
                "Creating UCloud sandbox id=%s image=%s cpus=%s memory_mb=%s "
                "disk_mb=%s labels=%s request_timeout_seconds=%.1f "
                "create_timeout_seconds=%.1f",
                sandbox_id,
                summary["image"],
                summary["cpus"],
                summary["memory_mb"],
                summary["disk_mb"],
                summary["labels"],
                self.request_timeout_seconds,
                self.create_timeout_seconds,
            )
            try:
                handle = await self.client.create_sandbox(**kwargs)
                self.logger.info(
                    "Created UCloud sandbox id=%s elapsed_seconds=%.1f",
                    sandbox_id,
                    time.perf_counter() - started,
                )
                break
            except BaseException as exc:
                existing = await self._get_existing_after_create_error(sandbox_id, exc)
                if existing is not None:
                    handle = existing
                    self.logger.info(
                        "Using existing UCloud sandbox id=%s after create error "
                        "elapsed_seconds=%.1f error=%r",
                        sandbox_id,
                        time.perf_counter() - started,
                        exc,
                    )
                    break
                self.logger.warning(
                    "UCloud sandbox create failed id=%s elapsed_seconds=%.1f error=%r",
                    sandbox_id,
                    time.perf_counter() - started,
                    exc,
                )
                if (
                    not _is_retryable_create_error(exc, self._api_error_type)
                    or asyncio.get_running_loop().time() >= deadline
                ):
                    raise
                await asyncio.sleep(self.retry_interval_seconds)
        self._sandboxes[sandbox_id] = handle
        return UCloudSandboxHandle(id=sandbox_id)

    async def _get_existing_after_create_error(
        self, sandbox_id: str, error: BaseException
    ) -> Any | None:
        should_probe = isinstance(error, TimeoutError)
        message = ""
        if isinstance(error, self._api_error_type):
            status_code = int(getattr(error, "status_code", 0) or 0)
            message = _error_message(error)
            should_probe = status_code in {400, 409, 502, 503} and (
                "already in use" in message
                or "conflict" in message
                or "timed out" in message
                or "timeout" in message
                or "node request failed" in message
            )
        if not should_probe:
            return None
        try:
            record = await self.client.get_sandbox(str(sandbox_id))
        except BaseException as probe_error:
            self.logger.debug(
                "Failed to probe UCloud sandbox id=%s after create error: %r",
                sandbox_id,
                probe_error,
            )
            if "already in use" in message:
                return UCloudSandboxHandle(id=sandbox_id)
            return None
        if not record:
            if "already in use" in message:
                return UCloudSandboxHandle(id=sandbox_id)
            return None
        state = _terminal_state(record)
        if state in {"failed", "error", "exited", "deleted"}:
            return None
        return UCloudSandboxHandle(id=sandbox_id)

    async def wait_for_creation(self, sandbox_id: str, *, max_attempts: int = 120) -> None:
        last_error: BaseException | None = None
        for attempt in range(max_attempts):
            try:
                record = await self.client.get_sandbox(str(sandbox_id))
            except BaseException as exc:
                if not _is_retryable_create_error(exc, self._api_error_type):
                    raise
                last_error = exc
                self.logger.debug(
                    "Transient UCloud sandbox readiness check failed id=%s "
                    "attempt=%s/%s error=%r",
                    sandbox_id,
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                await asyncio.sleep(1)
                continue
            if record is None:
                await asyncio.sleep(1)
                continue
            state = _terminal_state(record)
            if state in {"failed", "error", "exited", "deleted"}:
                raise RuntimeError(f"UCloud sandbox {sandbox_id} entered state {state!r}")
            if state is None or state in {"running", "ready", "started"}:
                return
            await asyncio.sleep(1)
        details = f"; last readiness error: {last_error!r}" if last_error else ""
        raise TimeoutError(
            f"UCloud sandbox {sandbox_id} was not visible after creation{details}"
        )

    async def delete(self, sandbox_id: str) -> None:
        self._sandboxes.pop(str(sandbox_id), None)
        try:
            await self.client.delete_sandbox(str(sandbox_id))
        except BaseException as exc:
            if _is_not_found_error(exc, self._api_error_type):
                return
            raise

    async def bulk_delete(self, sandbox_ids: list[str]) -> None:
        await asyncio.gather(*(self.delete(sandbox_id) for sandbox_id in sandbox_ids))

    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        attempts: int = 1,
        retry_interval_seconds: float | None = None,
    ) -> UCloudCommandResponse:
        attempts = max(1, attempts)
        for attempt in range(1, attempts + 1):
            try:
                result = await self.client.exec(
                    str(sandbox_id),
                    _shell_command(command, working_dir),
                    env=env,
                    timeout_seconds=timeout,
                )
                break
            except TimeoutError as exc:
                raise CommandTimeoutError(sandbox_id, command, timeout or 0) from exc
            except BaseException as exc:
                if (
                    attempt >= attempts
                    or not _is_retryable_exec_error(exc, self._api_error_type)
                ):
                    raise
                self.logger.debug(
                    "Transient UCloud exec failed id=%s attempt=%s/%s error=%r",
                    sandbox_id,
                    attempt,
                    attempts,
                    exc,
                )
                await asyncio.sleep(
                    retry_interval_seconds
                    if retry_interval_seconds is not None
                    else self.retry_interval_seconds
                )
        return UCloudCommandResponse(
            stdout=str(getattr(result, "stdout", "") or ""),
            stderr=str(getattr(result, "stderr", "") or ""),
            exit_code=int(getattr(result, "exit_code", 0) or 0),
        )

    async def start_background_job(
        self,
        sandbox_id: str,
        command: str,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> UCloudBackgroundJob:
        handle = await self.client.start_exec(
            str(sandbox_id),
            _shell_command(command, working_dir),
            env=env,
        )
        job_id = str(getattr(handle, "session_id"))
        self._jobs[job_id] = (handle, timeout, command)
        return UCloudBackgroundJob(sandbox_id=str(sandbox_id), job_id=job_id)

    async def get_background_job(
        self,
        sandbox_id: str,
        job: UCloudBackgroundJob,
        timeout: int | None = None,
    ) -> UCloudBackgroundJobStatus:
        handle, job_timeout, command = self._jobs[job.job_id]
        payload = await handle.get()
        session = payload.get("session") if isinstance(payload, dict) else None
        session = session if isinstance(session, dict) else getattr(handle, "session", {})
        status = str(session.get("status") or "").lower()
        if status not in {"exited", "failed"}:
            return UCloudBackgroundJobStatus(completed=False)
        try:
            result = await handle.wait(timeout_seconds=timeout or job_timeout)
        except TimeoutError as exc:
            raise CommandTimeoutError(sandbox_id, command, timeout or job_timeout or 0) from exc
        return UCloudBackgroundJobStatus(
            completed=True,
            exit_code=getattr(result, "exit_code", None),
            stdout=str(getattr(result, "stdout", "") or ""),
            stderr=str(getattr(result, "stderr", "") or ""),
        )

    async def run_background_job(
        self,
        sandbox_id: str,
        command: str,
        timeout: int | None = 900,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        poll_interval: int = 3,
    ) -> UCloudCommandResponse:
        job = await self.start_background_job(
            sandbox_id,
            command,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
        )
        deadline = asyncio.get_running_loop().time() + (900 if timeout is None else timeout)
        while True:
            status = await self.get_background_job(sandbox_id, job)
            if status.completed:
                return UCloudCommandResponse(
                    stdout=status.stdout,
                    stderr=status.stderr,
                    exit_code=int(status.exit_code or 0),
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise CommandTimeoutError(sandbox_id, command, timeout or 0)
            await asyncio.sleep(poll_interval)

    async def upload_bytes(
        self,
        sandbox_id: str,
        file_path: str,
        file_bytes: bytes,
        filename: str | None = None,
        timeout: int | None = None,
    ) -> UCloudFileUploadResponse:
        await self._upload_bytes(str(sandbox_id), file_path, file_bytes, timeout=timeout)
        return UCloudFileUploadResponse(success=True, path=file_path, size=len(file_bytes))

    async def upload_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        timeout: int | None = None,
    ) -> UCloudFileUploadResponse:
        await self._upload_file_from_path(
            str(sandbox_id), file_path, local_file_path, timeout=timeout
        )
        return UCloudFileUploadResponse(
            success=True,
            path=file_path,
            size=Path(local_file_path).stat().st_size,
        )

    async def download_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        timeout: int | None = None,
    ) -> object:
        Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
        data = await self._read_bytes(str(sandbox_id), file_path)
        Path(local_file_path).write_bytes(data)
        return None

    async def read_file(
        self,
        sandbox_id: str,
        path: str,
        timeout: int | None = None,
    ) -> UCloudFileReadResponse:
        try:
            data = await self._read_bytes(str(sandbox_id), path)
        except BaseException as exc:
            if _is_not_found_error(exc, self._api_error_type):
                raise SandboxFileNotFoundError(sandbox_id, path) from exc
            raise
        return UCloudFileReadResponse(content=data.decode("utf-8"))

    async def aclose(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result

    def teardown(self, wait: bool = True) -> None:
        return None

    async def _ensure_parent(self, sandbox_id: str, path: str) -> None:
        parent = _parent_dir(path)
        if parent is None:
            return
        response = await self.execute_command(
            sandbox_id,
            f"mkdir -p {shlex.quote(parent)}",
            timeout=30,
        )
        if response.exit_code != 0:
            raise RuntimeError(
                f"Failed to create parent directory {parent!r} in UCloud sandbox "
                f"{sandbox_id}: {response.stderr or response.stdout}"
            )

    async def _move_uploaded_file(
        self, sandbox_id: str, temp_path: str, path: str, timeout: int | None = None
    ) -> None:
        parent = _parent_dir(path)
        mkdir = f"mkdir -p {shlex.quote(parent)} && " if parent is not None else ""
        response = await self.execute_command(
            sandbox_id,
            f"{mkdir}mv {shlex.quote(temp_path)} {shlex.quote(path)}",
            timeout=timeout or 30,
        )
        if response.exit_code != 0:
            raise RuntimeError(
                f"Failed to move uploaded file into {path!r} in UCloud sandbox "
                f"{sandbox_id}: {response.stderr or response.stdout}"
            )

    async def _upload_bytes(
        self, sandbox_id: str, path: str, content: bytes, timeout: int | None = None
    ) -> None:
        upload_file = getattr(self.client, "upload_file", None)
        if callable(upload_file):
            await upload_file(str(sandbox_id), path, content)
            return
        parent = _parent_dir(path)
        mkdir = f"mkdir -p {shlex.quote(parent)} && " if parent is not None else ""
        command = (
            f"{mkdir}"
            "PY_BIN=$(command -v python3 || command -v python) && "
            '"$PY_BIN" -c '
            + shlex.quote(
                "import base64, os, sys; "
                "open(os.environ['VF_UPLOAD_PATH'], 'wb').write("
                "base64.b64decode(sys.stdin.read()))"
            )
        )
        result = await self.client.exec(
            str(sandbox_id),
            _shell_command(command),
            input=base64.b64encode(content).decode("ascii"),
            env={"VF_UPLOAD_PATH": path},
            timeout_seconds=timeout or 120,
        )
        exit_code = getattr(result, "exit_code", None)
        if exit_code != 0:
            raise RuntimeError(
                f"Failed to upload file into {path!r} in UCloud sandbox "
                f"{sandbox_id}: {getattr(result, 'stderr', '') or getattr(result, 'stdout', '')}"
            )

    async def _upload_file_from_path(
        self,
        sandbox_id: str,
        path: str,
        local_path: str,
        timeout: int | None = None,
    ) -> None:
        await self._upload_bytes(
            sandbox_id,
            path,
            Path(local_path).read_bytes(),
            timeout=timeout,
        )

    async def _read_bytes(self, sandbox_id: str, path: str) -> bytes:
        return await self.client.download_file(str(sandbox_id), path)


def ucloud_client_from_env(**kwargs: object) -> UCloudSandboxClient:
    return UCloudSandboxClient(**kwargs)

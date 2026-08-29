from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import cache
from typing import cast

from verifiers.v1.configs.runtime import (
    BaseRuntimeConfig,
    NetworkPolicyConfig,
    WireRuntimeConfig,
)
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeProcess,
    WireRuntimeInfo,
    register,
)
from verifiers.v1.runtimes.docker import DockerConfig, DockerRuntime, DockerRuntimeInfo
from verifiers.v1.runtimes.modal import ModalConfig, ModalRuntime, ModalRuntimeInfo
from verifiers.v1.runtimes.prime import (
    PrimeConfig,
    PrimeRuntime,
    PrimeRuntimeInfo,
    set_base_sandbox_labels,
)
from verifiers.v1.runtimes.subprocess import (
    SubprocessConfig,
    SubprocessRuntime,
    SubprocessRuntimeInfo,
)

# Explicit built-ins retain their full CLI help; the base resolves installed runtimes.
RuntimeConfig = (
    SubprocessConfig | DockerConfig | PrimeConfig | ModalConfig | BaseRuntimeConfig
)
RuntimeInfo = (
    SubprocessRuntimeInfo
    | DockerRuntimeInfo
    | PrimeRuntimeInfo
    | ModalRuntimeInfo
    | BaseRuntimeInfo
)

_BUILTIN_RUNTIMES = {
    "subprocess": SubprocessRuntime,
    "docker": DockerRuntime,
    "prime": PrimeRuntime,
    "modal": ModalRuntime,
}


def _check_runtime_class(runtime_type: str, cls: type[Runtime]) -> type[Runtime]:
    try:
        config_cls, info_cls = cls.config_cls, cls.info_cls
    except AttributeError as e:
        raise TypeError(
            f"runtime {cls.__name__} must declare config_cls and info_cls"
        ) from e
    if not issubclass(config_cls, BaseRuntimeConfig):
        raise TypeError("runtime config_cls must subclass BaseRuntimeConfig")
    if not issubclass(info_cls, config_cls) or not issubclass(
        info_cls, BaseRuntimeInfo
    ):
        raise TypeError("runtime info_cls must inherit its config and BaseRuntimeInfo")
    if config_cls.model_fields["type"].default != runtime_type:
        raise ValueError(
            f"runtime module and config type disagree for {runtime_type!r}"
        )
    return cls


@cache
def find_runtime_class(runtime_type: str) -> type[Runtime] | None:
    """Resolve a built-in or installed runtime class, if present."""
    if cls := _BUILTIN_RUNTIMES.get(runtime_type):
        return cls
    from verifiers.v1.utils.loaders import _plugin_class, import_runtime

    try:
        module = import_runtime(runtime_type)
    except ModuleNotFoundError:
        return None
    cls = cast(type[Runtime], _plugin_class(module, Runtime, "runtime"))
    return _check_runtime_class(runtime_type, cls)


def runtime_class(runtime_type: str) -> type[Runtime]:
    """Resolve ``runtime_type`` or raise when it is not installed."""
    if cls := find_runtime_class(runtime_type):
        return cls
    raise ValueError(f"runtime type {runtime_type!r} is not installed")


def runtime_config_type(runtime_type: str) -> type[BaseRuntimeConfig]:
    """The concrete config class for ``runtime_type``."""
    return runtime_class(runtime_type).config_cls


def make_runtime(config: RuntimeConfig, name: str | None = None) -> Runtime:
    cls = runtime_class(config.type)
    if not isinstance(config, cls.config_cls):
        raise TypeError(
            f"runtime config for {config.type!r} must be {cls.config_cls.__name__}, "
            f"got {type(config).__name__}"
        )
    runtime = cls(config, name)
    register(runtime)
    return runtime


@asynccontextmanager
async def provision_runtime(
    config: RuntimeConfig,
    name: str | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[Runtime]:
    """Provision a box from `config` and tear it down on exit.

    `start()` sits inside the `try`: a failed start may already hold a paid sandbox, so
    it has to reach `stop()` (which is safe on a partially-started runtime)."""
    runtime = make_runtime(config, name)
    runtime.env = dict(env or {})
    try:
        await runtime.start()
        yield runtime
    finally:
        await runtime.stop()


def runtime_is_local(config: RuntimeConfig) -> bool:
    """Whether a runtime of this config exchanges host-local URLs without a public
    tunnel, read off the runtime class without provisioning one."""
    return runtime_class(config.type).is_local


__all__ = [
    "BaseRuntimeConfig",
    "BaseRuntimeInfo",
    "DockerConfig",
    "DockerRuntime",
    "DockerRuntimeInfo",
    "ModalConfig",
    "ModalRuntime",
    "ModalRuntimeInfo",
    "NetworkPolicyConfig",
    "PrimeConfig",
    "PrimeRuntime",
    "PrimeRuntimeInfo",
    "ProgramResult",
    "Runtime",
    "RuntimeConfig",
    "RuntimeInfo",
    "RuntimeProcess",
    "SubprocessConfig",
    "SubprocessRuntime",
    "SubprocessRuntimeInfo",
    "WireRuntimeConfig",
    "WireRuntimeInfo",
    "find_runtime_class",
    "make_runtime",
    "provision_runtime",
    "runtime_class",
    "runtime_config_type",
    "runtime_is_local",
    "set_base_sandbox_labels",
]

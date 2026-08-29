"""Every checked-in v1 eval config parses.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig);
`endpoints.toml` isn't an eval config, and `configs/eval|rl|gepa/` are the legacy
`vf-eval` / training formats (different, non-v1 config classes), so both are out of scope here.
"""

import sys
import tomllib
from contextlib import asynccontextmanager
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest
from pydantic_config import cli

import verifiers.v1 as vf
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.cli.validate import ValidateConfig
from verifiers.v1.interception import (
    BaseInterceptionConfig,
    Interception,
    make_interception,
)
from verifiers.v1.runtimes import make_runtime, runtime_is_local

CONFIGS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml")
    if p.name != "endpoints.toml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    assert config.env.taskset.id


class ExternalRuntimeConfig(vf.BaseRuntimeConfig):
    type: Literal["external"] = "external"
    region: str


class ExternalRuntimeInfo(ExternalRuntimeConfig, vf.BaseRuntimeInfo):
    pass


class ExternalRuntime(vf.Runtime):
    config_cls = ExternalRuntimeConfig
    info_cls = ExternalRuntimeInfo
    is_local = False

    def __init__(self, config: ExternalRuntimeConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = ExternalRuntimeInfo(**config.model_dump())

    async def start(self) -> None:
        self.info.id = self.name

    async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
        return vf.ProgramResult(0, "", "")

    async def _read(self, path: str) -> bytes:
        return b""

    async def write(self, path: str, data: bytes) -> None:
        pass


class ExternalInterceptionConfig(BaseInterceptionConfig):
    type: Literal["external"] = "external"
    url: str


class ExternalInterception(Interception):
    config_cls = ExternalInterceptionConfig

    def __init__(self, config, requires_tunnel=False, state_service_secrets=()):
        super().__init__()
        self.config = config

    async def start(self) -> None:
        pass

    @asynccontextmanager
    async def acquire(self, session):
        yield self.config.url, "model-secret", "state-secret"


def test_installed_runtime_uses_normal_config_and_factory_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("external")
    module.__spec__ = ModuleSpec(module.__name__, loader=None)
    module.ExternalRuntime = ExternalRuntime
    module.ExternalInterception = ExternalInterception
    module.__all__ = ["ExternalRuntime", "ExternalInterception"]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    agent = vf.AgentConfig.model_validate(
        {"runtime": {"type": "external", "region": "test-region"}}
    )
    assert isinstance(agent.runtime, ExternalRuntimeConfig)
    assert agent.model_dump()["runtime"]["region"] == "test-region"
    runtime = make_runtime(agent.runtime, "external-runtime")
    assert isinstance(runtime, ExternalRuntime)
    assert not runtime_is_local(agent.runtime)
    info = vf.AgentInfo(config=agent, runtime=runtime.info)
    assert isinstance(
        type(info).model_validate(info.model_dump()).runtime, ExternalRuntimeInfo
    )

    parsed = cli(
        ValidateConfig,
        args=["--runtime.type", "external", "--runtime.region", "cli-region"],
    )
    assert isinstance(parsed.runtime, ExternalRuntimeConfig)

    env = vf.EnvConfig.model_validate(
        {"interception": {"type": "external", "url": "https://example.test"}}
    )
    assert isinstance(env.interception, ExternalInterceptionConfig)
    assert isinstance(
        make_interception(env.interception, requires_tunnel=False),
        ExternalInterception,
    )

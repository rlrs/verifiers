"""`Runtime.write_many`: the batch-write hook harness setup uploads through."""

import os
from pathlib import Path

import verifiers.v1 as vf


class RecordingRuntime(vf.Runtime):
    def __init__(self) -> None:
        super().__init__()
        self.writes: list[tuple[str, bytes]] = []
        self.batches: list[dict[str, bytes]] = []
        self.runs: list[list[str]] = []

    async def start(self) -> None:
        pass

    async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
        self.runs.append(argv)
        return vf.ProgramResult(0, "", "")

    async def _read(self, path: str) -> bytes:
        return b""

    async def write(self, path: str, data: bytes) -> None:
        self.writes.append((path, data))


class BatchRuntime(RecordingRuntime):
    async def write_many(self, files) -> None:
        self.batches.append(dict(files))


class SkillsHarness(vf.Harness[vf.HarnessConfig]):
    SUPPORTS_SKILLS = True

    async def launch(self, *args) -> vf.ProgramResult:
        raise NotImplementedError


async def test_write_many_defaults_to_write_per_file_in_order() -> None:
    runtime = RecordingRuntime()
    await runtime.write_many({"/b": b"2", "/a": b"1", "/c/d": b""})
    assert runtime.writes == [("/b", b"2"), ("/a", b"1"), ("/c/d", b"")]


async def test_install_skills_uploads_one_batch(tmp_path: Path) -> None:
    alpha, beta = tmp_path / "alpha", tmp_path / "beta"
    (alpha / "scripts").mkdir(parents=True)
    beta.mkdir()
    (alpha / "SKILL.md").write_bytes(b"alpha")
    (alpha / "scripts" / "run.sh").write_bytes(b"#!/bin/sh\n")
    os.chmod(alpha / "scripts" / "run.sh", 0o755)
    (beta / "SKILL.md").write_bytes(b"beta")
    harness = SkillsHarness(vf.HarnessConfig(skills=[alpha, beta]))

    runtime = BatchRuntime()
    await harness.install_skills(runtime, "/skills")
    assert runtime.batches == [
        {
            "/skills/alpha/SKILL.md": b"alpha",
            "/skills/alpha/scripts/run.sh": b"#!/bin/sh\n",
            "/skills/beta/SKILL.md": b"beta",
        }
    ]
    assert runtime.writes == []
    # The batch moves bytes, not modes; executables still get their bit back.
    assert runtime.runs == [["chmod", "+x", "/skills/alpha/scripts/run.sh"]]

    # A runtime without a batch override still gets every file, in order.
    runtime = RecordingRuntime()
    await harness.install_skills(runtime, "/skills")
    assert [path for path, _ in runtime.writes] == [
        "/skills/alpha/SKILL.md",
        "/skills/alpha/scripts/run.sh",
        "/skills/beta/SKILL.md",
    ]
    assert runtime.runs == [["chmod", "+x", "/skills/alpha/scripts/run.sh"]]

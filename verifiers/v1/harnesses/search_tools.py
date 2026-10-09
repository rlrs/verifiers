"""Fail early when an offline coding harness lacks its lazy search dependencies."""

from verifiers.v1.harnesses.node import NODE_BIN_DIR


async def check_offline_search_tools(runtime, *, needs_fd: bool = True) -> None:
    if getattr(runtime.config, "offline_harness_bundle", None) is None:
        return
    commands = ["rg --version"]
    if needs_fd:
        commands.append("fd --version")
    result = await runtime.run(
        ["sh", "-eu", "-c", f'export PATH="{NODE_BIN_DIR}:$PATH"; ' + "; ".join(commands)], {}
    )
    if result.exit_code:
        raise RuntimeError(
            "Offline harness search tools are missing or cannot execute; rebuild the asset bundle "
            "with static ripgrep and fd before starting rollouts: " + result.stderr.strip()[-500:]
        )

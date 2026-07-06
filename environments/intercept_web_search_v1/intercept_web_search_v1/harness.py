import json
import shlex

import verifiers.v1 as vf

PROVIDER = "intercept"
KEY_VAR = "CODEX_INTERCEPT_KEY"


class InterceptWebSearchHarnessConfig(vf.HarnessConfig):
    codex_bin: str = "codex"
    """Codex executable available on PATH."""


class InterceptWebSearchHarness(vf.Harness[InterceptWebSearchHarnessConfig]):
    SUPPORTS_MCP = True

    async def setup(self, runtime: vf.Runtime) -> None:
        found = await runtime.run(
            ["sh", "-c", f"command -v {shlex.quote(self.config.codex_bin)}"],
            self.config.resolved_env,
        )
        if found.exit_code != 0:
            raise RuntimeError("intercept-web-search-v1 requires codex on PATH")

    async def launch(
        self,
        ctx: vf.RolloutContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        _, prompt = self.resolve_prompt(trace.task)
        mcp_config = [
            arg
            for name, url in mcp_urls.items()
            for arg in ("-c", f"mcp_servers.{name}.url={json.dumps(url)}")
        ]
        return await runtime.run_program(
            [
                self.config.codex_bin,
                "--search",
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "-m",
                ctx.model,
                "-c",
                f"model_provider={PROVIDER}",
                "-c",
                f"model_providers.{PROVIDER}.name={PROVIDER}",
                "-c",
                f"model_providers.{PROVIDER}.base_url={endpoint}",
                "-c",
                f"model_providers.{PROVIDER}.env_key={KEY_VAR}",
                "-c",
                f"model_providers.{PROVIDER}.wire_api=responses",
                "-c",
                f"model_providers.{PROVIDER}.requires_openai_auth=false",
                *mcp_config,
                str(prompt or ""),
            ],
            {**self.config.resolved_env, KEY_VAR: secret},
        )

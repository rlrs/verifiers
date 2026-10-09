"""Pinned OpenCode ACP harness with all model calls routed through interception."""

import json
from typing import Literal

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.harnesses.search_tools import check_offline_search_tools
from verifiers.v1.harnesses.utils.install import ensure_installed

PREFIX = "/var/tmp/vf-opencode"
BINARY = f"{PREFIX}/node_modules/.bin/opencode"


class OpenCodeHarnessConfig(HarnessConfig):
    compaction: bool = False
    """Enable native context compaction, reserving 32K tokens for generation."""

    version: Literal["1.18.31"] = "1.18.31"


class OpenCodeHarness(ACPHarness[OpenCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True

    async def setup(self, runtime):
        await ensure_node(runtime)
        await check_offline_search_tools(runtime, needs_fd=False)
        await ensure_installed(
            runtime,
            directory=PREFIX,
            label="OpenCode",
            ready=f'test -x {BINARY} && test "$({BINARY} --version)" = "$VF_OPENCODE_VERSION"',
            install='export PATH="/var/tmp/vf-node/bin:$PATH"; npm install --prefix /var/tmp/vf-opencode --no-audit --no-fund "opencode-ai@$VF_OPENCODE_VERSION"',
            env={"VF_OPENCODE_VERSION": self.config.version},
        )
        await super().setup(runtime)

    async def prepare_acp(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        if self.config.disabled_tools:
            raise ValueError("OpenCode per-tool restrictions are not configured")
        system_prompt, prompt = self.resolve_prompt(data)
        model = f"intercept/{ctx.model}"
        config = {
            "$schema": "https://opencode.ai/config.json",
            "model": model,
            "small_model": model,
            "enabled_providers": ["intercept"],
            "autoupdate": False,
            "share": "disabled",
            "permission": "allow",
            "compaction": {
                "auto": self.config.compaction,
                "prune": False,
                "reserved": 32768,
            },
            "agent": {"title": {"disable": True}, "summary": {"disable": True}},
            "provider": {
                "intercept": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Verifiers interception",
                    "options": {
                        "baseURL": endpoint,
                        "apiKey": "{env:OPENCODE_INTERCEPT_KEY}",
                    },
                    "models": {
                        ctx.model: {
                            "name": ctx.model,
                            "limit": {
                                "context": 131072,
                                "input": 131072,
                                "output": ctx.sampling.max_tokens or 4096,
                            },
                        }
                    },
                }
            },
        }
        return ACPConfig(
            command=[
                "/bin/sh",
                "-eu",
                "-c",
                f'export PATH="{NODE_BIN_DIR}:$PATH"; exec "$@"',
                "opencode",
                BINARY,
                "acp",
            ],
            env={
                **self.config.resolved_env,
                "OPENCODE_CONFIG_CONTENT": json.dumps(config),
                "OPENCODE_INTERCEPT_KEY": secret,
                "OPENCODE_DISABLE_AUTOUPDATE": "1",
                "OPENCODE_DISABLE_MODELS_FETCH": "1",
                "OPENCODE_DISABLE_AUTOCOMPACT": "0" if self.config.compaction else "1",
                "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            },
            prompt=prompt,
            system_prompt=system_prompt,
        )

"""Shared configuration for execution-time network policy."""

from fnmatch import fnmatchcase
from typing import Self
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, model_validator
from pydantic_config import BaseConfig


class BaseRuntimeConfig(BaseConfig):
    """Base config for built-in and installed runtime implementations."""

    type: str

    @model_validator(mode="wrap")
    @classmethod
    def _resolve_runtime(cls, value, handler):
        if cls is BaseRuntimeConfig and isinstance(value, dict):
            runtime_type = value.get("type")
            if isinstance(runtime_type, str) and runtime_type:
                from verifiers.v1.runtimes import runtime_config_type

                return runtime_config_type(runtime_type).model_validate(value)
        return handler(value)


class WireRuntimeConfig(BaseRuntimeConfig):
    """Runtime-agnostic config stored in permissive wire records."""

    model_config = ConfigDict(extra="allow")


def network_rule_matches(rule: str, scheme: str, host: str, port: int) -> bool:
    """Match a network-policy host pattern or URL origin. Paths are ignored."""
    value = rule.lower().rstrip("/")
    try:
        parsed = urlsplit(value if "://" in value else f"//{value}")
        rule_port = parsed.port
    except ValueError:
        return False
    pattern = (parsed.hostname or "").rstrip(".")
    if not pattern or (parsed.scheme and parsed.scheme != scheme):
        return False
    if parsed.scheme and rule_port is None:
        rule_port = 443 if parsed.scheme == "https" else 80
    if rule_port is not None and rule_port != port:
        return False
    host = host.lower().rstrip(".")
    return fnmatchcase(host, pattern) or (
        pattern.startswith("*.") and host == pattern[2:]
    )


class NetworkPolicyConfig(BaseRuntimeConfig):
    """Shared execution-time policy surface for runtimes that support it."""

    # A bare policy is used to mediate model capabilities without representing a
    # provisionable runtime. Concrete runtime configs override this with a literal.
    type: str = ""

    allow: list[str] = Field(default_factory=lambda: ["*"])
    """Destinations allowed during execution; `*` is unrestricted and `[]` is
    framework-only."""
    block: list[str] = Field(default_factory=list)
    """Destinations denied during execution; any `*` makes the policy framework-only."""

    @model_validator(mode="after")
    def validate_network_policy(self) -> Self:
        if not self.allow or "*" in self.block:
            # Empty allowlists and wildcard blocks both mean framework-only access.
            self.allow = []
            self.block = ["*"]
        elif self.allow != ["*"] and self.block:
            raise ValueError(
                "non-empty concrete allow and block egress lists are mutually exclusive"
            )
        return self

    @property
    def network_restricted(self) -> bool:
        return "*" not in self.allow or bool(self.block)

    def permits(self, scheme: str, host: str, port: int) -> bool:
        """Whether the destination is allowed by the configured egress rules."""
        return not any(
            network_rule_matches(rule, scheme, host, port) for rule in self.block
        ) and any(network_rule_matches(rule, scheme, host, port) for rule in self.allow)

    def with_task_network_policy(self, allow: list[str], block: list[str]) -> Self:
        values = self.model_dump()
        if "*" in allow:
            allow = self.allow
        elif "*" not in self.allow:
            # Rules are opaque: guessing overlap between different globs could widen the policy.
            runtime_allow = set(self.allow)
            allow = [rule for rule in allow if rule in runtime_allow]
        block = list(dict.fromkeys([*block, *self.block]))
        return type(self).model_validate({**values, "allow": allow, "block": block})

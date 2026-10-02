"""Failures, named after the stage that produced them (crates/syrup-runtime/docs/contract.md, section 5)."""

import json


class SyrupError(Exception):
    def __init__(self, stage, kind, reason, operation=None, hint=None, details=None):
        super().__init__(reason)
        self.stage = stage
        self.kind = kind
        self.reason = reason
        self.operation = operation
        self.hint = hint
        self.details = details or {}

    def __str__(self):
        text = f"[{self.stage}/{self.kind}] "
        if self.operation:
            text += f"{self.operation}: "
        text += self.reason
        if self.hint:
            text += f" (hint: {self.hint})"
        return text


class InputError(SyrupError):
    pass


class IntentError(SyrupError):
    pass


class PlanError(SyrupError):
    pass


class BuildError(SyrupError):
    pass


class LoadError(SyrupError):
    pass


class ValidationError(SyrupError):
    pass


class ExecutionError(SyrupError):
    pass


class DependencyError(SyrupError):
    """Something Syrup needs is missing: a compiler, a model, an engine."""


_BY_STAGE = {
    "input": InputError,
    "resolve": IntentError,
    "plan": PlanError,
    "generate": BuildError,
    "compile": BuildError,
    "load": LoadError,
    "validate": ValidationError,
    "execute": ExecutionError,
}


def from_native(payload):
    e = json.loads(payload)
    cls = DependencyError if e["kind"] == "missing_dependency" else _BY_STAGE[e["stage"]]
    return cls(e["stage"], e["kind"], e["reason"], e.get("operation"), e.get("hint"), e.get("details"))

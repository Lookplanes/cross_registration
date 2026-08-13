"""Minimal configuration system: argparse schema + YAML defaults + CLI overrides."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    """Raised when a YAML configuration does not match the CLI schema."""


def _read_yaml(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    with config_path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ConfigError("configuration root must be a mapping")
    return document


def _flatten_sections(document: dict[str, Any], sections: Iterable[str]) -> dict[str, Any]:
    allowed_sections = set(sections)
    values: dict[str, Any] = {}
    for section, content in document.items():
        if section not in allowed_sections:
            raise ConfigError(
                f"unknown config section '{section}'; allowed: {sorted(allowed_sections)}"
            )
        if not isinstance(content, dict):
            raise ConfigError(f"config section '{section}' must be a mapping")
        for key, value in content.items():
            destination = key.replace("-", "_")
            if destination in values:
                raise ConfigError(f"duplicate config key '{key}' across sections")
            values[destination] = value
    return values


def _action_map(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {
        action.dest: action for action in parser._actions
        if action.dest not in {"help", "config"}
    }


def _validate_value(action: argparse.Action, value: Any) -> None:
    if value is None:
        return
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                           argparse.BooleanOptionalAction)):
        if not isinstance(value, bool):
            raise ConfigError(f"'{action.dest}' must be boolean")
        candidates = [value]
    elif action.nargs in {"+", "*"} or (isinstance(action.nargs, int) and action.nargs > 0):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"'{action.dest}' must be a list")
        candidates = value
    else:
        candidates = [value]
    if action.type is not None:
        for candidate in candidates:
            try:
                action.type(candidate)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"invalid value for '{action.dest}': {candidate!r}"
                ) from exc
    if action.choices is not None:
        invalid = [candidate for candidate in candidates if candidate not in action.choices]
        if invalid:
            raise ConfigError(
                f"invalid value for '{action.dest}': {invalid}; choices={list(action.choices)}"
            )


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    sections: Iterable[str],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse YAML as defaults, then let explicit CLI arguments override it.

    The parser remains the single field/type schema.  YAML uses named sections
    for readability, but section keys are flattened onto argparse destinations.
    """
    if not any(action.dest == "config" for action in parser._actions):
        parser.add_argument("--config", help="YAML config; explicit CLI values override it")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    preliminary, _ = pre_parser.parse_known_args(argv)
    if preliminary.config:
        values = _flatten_sections(_read_yaml(preliminary.config), sections)
        actions = _action_map(parser)
        unknown = sorted(set(values) - set(actions))
        if unknown:
            raise ConfigError(f"unknown config keys: {unknown}")
        for key, value in values.items():
            _validate_value(actions[key], value)
            if actions[key].required:
                actions[key].required = False
        parser.set_defaults(**values)
    return parser.parse_args(argv)


def save_resolved_config(
    args: argparse.Namespace, path: str | Path,
    sections: dict[str, Iterable[str]],
) -> None:
    """Save the exact effective configuration grouped into stable sections."""
    values = vars(args)
    payload: dict[str, dict[str, Any]] = {}
    consumed: set[str] = {"config"}
    for section, keys in sections.items():
        payload[section] = {}
        for key in keys:
            if key in values:
                payload[section][key] = values[key]
                consumed.add(key)
    remaining = {
        key: value for key, value in values.items()
        if key not in consumed
    }
    if remaining:
        payload["other"] = remaining
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

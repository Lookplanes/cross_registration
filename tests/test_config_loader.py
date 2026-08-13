"""Tests for YAML defaults, CLI precedence and schema validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import yaml

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def test_yaml_supplies_required_value_and_cli_overrides(tmp_path: Path) -> None:
    from crossreg.config import parse_args_with_config

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({
            "data": {"train_dir": "/yaml/train"},
            "training": {"batch_size": 8},
            "checkpoint": {"resume": True},
        }),
        encoding="utf-8",
    )
    args = parse_args_with_config(
        _parser(), ("data", "training", "checkpoint"),
        ["--config", str(path), "--batch-size", "2"],
    )
    assert args.train_dir == "/yaml/train"
    assert args.batch_size == 2
    assert args.resume is True


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    from crossreg.config import ConfigError, parse_args_with_config

    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  unknown: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown config keys"):
        parse_args_with_config(_parser(), ("data",), ["--config", str(path)])

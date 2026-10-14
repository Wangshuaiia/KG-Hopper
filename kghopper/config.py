"""YAML config with dotted command-line overrides, e.g. `--set grpo.lr=1e-6 kg.width=5`."""
import argparse
import copy
import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "local_cpu.yaml")


class Config(dict):
    """dict with attribute access for nested sections."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def get_section(self, key):
        return Config(self.get(key) or {})


def _set_dotted(cfg, dotted, raw_value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw_value)


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _read(path):
    """A config may start from another one: `base: local_cpu.yaml` (relative to this file)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _deep_merge(_read(os.path.join(os.path.dirname(os.path.abspath(path)), base)), cfg)
    return cfg


def load_config(path=None, overrides=()):
    cfg = _read(path or DEFAULT_CONFIG)
    for item in overrides:
        key, _, value = item.partition("=")
        _set_dotted(cfg, key.strip(), value)
    return Config(cfg)


def resolve_path(path):
    """Paths in configs are relative to the repo root."""
    if path is None or os.path.isabs(path) or "://" in str(path):
        return path
    return os.path.join(REPO_ROOT, path)


def add_config_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values, e.g. grpo.lr=1e-6")
    return parser

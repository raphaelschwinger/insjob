"""Lab policy: hosts, who may use which, and resource defaults.

The built-in defaults below are overridden by the shared `/datahome/insjob/lab.toml`
(edited by the lab), and a user may narrow — never widen — them in their own config.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

LAB_DIR = "/datahome/insjob"
LOCK_DIR = "/var/tmp/insjob"

DEFAULT_LAB: dict = {
    "hosts": {
        "ploen": {"fqdn": "ploen.ins.informatik.uni-kiel.de", "gpus": "4x RTX 4090 24GB"},
        "amur": {"fqdn": "amur.ins.informatik.uni-kiel.de", "gpus": "4x RTX 3090 24GB"},
        "kiel": {"fqdn": "kiel.ins.informatik.uni-kiel.de", "gpus": "4x A100 80GB"},
        "stein": {"fqdn": "stein.ins.informatik.uni-kiel.de", "gpus": "4x RTX 5090 32GB"},
        "stohl": {"fqdn": "stohl.ins.informatik.uni-kiel.de", "gpus": "2x RTX PRO 6000 96GB"},
    },
    # first group whose `match` intersects the user's unix groups wins
    "groups": {
        "staff": {
            "match": ["ins"],
            "hosts": ["ploen", "kiel", "stein", "stohl", "amur"],
            "default_host": "ploen",
            "max_gpus_per_host": 4,
        },
        "student": {
            "match": ["stud"],
            "hosts": ["amur"],
            "default_host": "amur",
            "max_gpus_per_host": 2,
        },
    },
    "default_group": "student",
    # INS docker guidelines
    "resources": {
        "cpus": 8,
        "memory": "64g",
        "memory_reservation": "32g",
        "max_cpus": 16,
        "max_memory": "126g",
        "shm_size": "16g",
    },
    # a GPU counts as free below these
    "gpu_free": {"max_mem_used_mb": 1024, "max_util": 5},
    "paths": {"lab_dir": LAB_DIR, "lock_dir": LOCK_DIR},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_lab(shared_file: str | Path | None = None) -> dict:
    lab = copy.deepcopy(DEFAULT_LAB)
    path = Path(shared_file or Path(LAB_DIR) / "lab.toml")
    try:
        with open(path, "rb") as f:
            lab = deep_merge(lab, tomllib.load(f))
    except (FileNotFoundError, PermissionError):
        pass
    return lab


def group_for(lab: dict, groups: list[str]) -> str:
    for name, g in lab["groups"].items():
        if set(g.get("match", [])) & set(groups):
            return name
    return lab["default_group"]


def user_policy(lab: dict, groups: list[str], narrow: dict | None = None) -> dict:
    """Effective policy for a user. `narrow` (from the user's config) can only restrict."""
    gname = group_for(lab, groups)
    g = lab["groups"][gname]
    pol = {
        "group": gname,
        "hosts": [h for h in g["hosts"] if h in lab["hosts"]],
        "default_host": g.get("default_host", g["hosts"][0]),
        "max_gpus_per_host": int(g.get("max_gpus_per_host", 1)),
    }
    if narrow:
        if narrow.get("hosts"):
            pol["hosts"] = [h for h in pol["hosts"] if h in narrow["hosts"]]
        if narrow.get("max_gpus_per_host") is not None:
            pol["max_gpus_per_host"] = min(
                pol["max_gpus_per_host"], int(narrow["max_gpus_per_host"])
            )
        if narrow.get("default_host") in pol["hosts"]:
            pol["default_host"] = narrow["default_host"]
    if pol["default_host"] not in pol["hosts"] and pol["hosts"]:
        pol["default_host"] = pol["hosts"][0]
    return pol


def fqdn(lab: dict, host: str) -> str:
    return lab["hosts"].get(host, {}).get("fqdn", host)


_UNITS = {"k": 1 / 1024 / 1024, "m": 1 / 1024, "g": 1, "t": 1024}


def size_gb(s: str | int | float) -> float:
    """'32g' -> 32.0, '512m' -> 0.5, '126GB' -> 126.0"""
    if isinstance(s, (int, float)):
        return float(s)
    t = s.strip().lower().rstrip("b")
    if t and t[-1] in _UNITS:
        return float(t[:-1]) * _UNITS[t[-1]]
    return float(t) / 1024**3


def reservation(job: dict, default: str = "32g") -> str:
    """The job's soft memory reservation, never above its hard limit: docker refuses
    `--memory-reservation` greater than `--memory` (e.g. `-m 16g` vs the 32g default)."""
    res = str(job.get("memory_reservation") or default)
    mem = job.get("memory")
    if mem and size_gb(res) > size_gb(mem):
        return str(mem)
    return res


def clamp_resources(lab: dict, cpus: int, memory: str) -> tuple[int, str]:
    r = lab["resources"]
    cpus = min(int(cpus), int(r["max_cpus"]))
    if size_gb(memory) > size_gb(r["max_memory"]):
        memory = r["max_memory"]
    return cpus, memory

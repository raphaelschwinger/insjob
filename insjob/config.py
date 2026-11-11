"""Laptop-side configuration: ~/.config/insjob/config.toml and per-repo .insjob.toml."""

from __future__ import annotations

import os
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .identity import Identity
from .policy import DEFAULT_LAB, deep_merge, user_policy

CONFIG_PATH = Path(os.environ.get("INSJOB_CONFIG", "~/.config/insjob/config.toml")).expanduser()
REPO_FILE = ".insjob.toml"


class NotSetUp(RuntimeError):
    pass


# ---- tiny TOML writer (enough for our flat-ish configs) --------------------------------
def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v)} to TOML")


def dump_toml(d: dict, prefix: str = "") -> str:
    scalars = [(k, v) for k, v in d.items() if not isinstance(v, dict) and v is not None]
    tables = [(k, v) for k, v in d.items() if isinstance(v, dict)]
    out = [f"{k} = {_toml_value(v)}" for k, v in scalars]
    for k, v in tables:
        name = f"{prefix}{k}"
        if any(not isinstance(x, dict) for x in v.values()) or not v:
            out.append(f"\n[{name}]")
        body = dump_toml(v, f"{name}.")
        if body:
            out.append(body)
    return "\n".join(out).strip("\n") + "\n"


# ---- laptop config --------------------------------------------------------------------
@dataclass
class LaptopConfig:
    identity: Identity
    lab: dict = field(default_factory=lambda: dict(DEFAULT_LAB))
    default_host: str | None = None
    narrow: dict = field(default_factory=dict)

    @property
    def policy(self) -> dict:
        return user_policy(self.lab, self.identity.groups, self.narrow or None)

    @property
    def host(self) -> str:
        p = self.policy
        return self.default_host if self.default_host in p["hosts"] else p["default_host"]

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> LaptopConfig:
        if not path.exists():
            raise NotSetUp("insjob is not set up on this machine — run `job setup` first")
        with open(path, "rb") as f:
            d = tomllib.load(f)
        lab = deep_merge(DEFAULT_LAB, d.get("lab", {}))
        return cls(identity=Identity.from_dict(d["identity"]), lab=lab,
                   default_host=d.get("default_host"), narrow=d.get("policy", {}))

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {"default_host": self.default_host, "identity": self.identity.to_dict(),
             "lab": {"hosts": self.lab["hosts"], "groups": self.lab["groups"],
                     "default_group": self.lab["default_group"]}}
        if self.narrow:
            d["policy"] = self.narrow
        path.write_text("# written by `job setup`\n" + dump_toml(d))


# ---- repo config ----------------------------------------------------------------------
def git(*args: str, cwd: Path | str | None = None, check: bool = True) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout.strip()


def repo_root(start: Path | None = None) -> Path:
    try:
        return Path(git("rev-parse", "--show-toplevel", cwd=start or Path.cwd()))
    except RuntimeError as e:
        raise RuntimeError("not inside a git repository") from e


@dataclass
class RepoConfig:
    root: Path
    name: str
    devcontainer: str = ".devcontainer/devcontainer.json"
    setup: str = ""
    resources: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    mounts: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    hosts: list[str] = field(default_factory=list)
    notebook: dict = field(default_factory=dict)
    exists: bool = False

    @classmethod
    def load(cls, start: Path | None = None) -> RepoConfig:
        root = repo_root(start)
        f = root / REPO_FILE
        d: dict = {}
        if f.exists():
            with open(f, "rb") as fh:
                d = tomllib.load(fh)
        name = d.get("name") or root.name
        dc = d.get("devcontainer", ".devcontainer/devcontainer.json")
        return cls(
            root=root,
            name=name,
            devcontainer=dc,
            setup=d.get("setup", ""),
            resources=d.get("resources", {}),
            data=d.get("data", {}),
            mounts=(d.get("mounts") or {}).get("extra", []),
            env=d.get("env", {}),
            hosts=d.get("hosts", []),
            notebook=d.get("notebook", {}),
            exists=f.exists(),
        )


def repo_template(name: str, has_uv: bool, devcontainer: str) -> str:
    setup = 'setup = "uv sync --frozen"' if has_uv else '# setup = "pip install -e ."'
    return f"""# insjob — how this repo runs on the INS servers (https://github.com/raphaelschwinger/insjob)
name = "{name}"
devcontainer = "{devcontainer}"   # the image is built from this
{setup}             # runs in the container before every command
# hosts = ["ploen"]              # default: your group's default host / any allowed host

[resources]                      # defaults; CLI flags override, lab policy caps
gpus = 1
cpus = 8
memory = "64g"
min_gpu_mem_gb = 0
shm_size = "16g"

[env]
# {{user}} is replaced by the server login of whoever runs the job
UV_CACHE_DIR = "/datahome/{{user}}/.cache/uv"
UV_PROJECT_ENVIRONMENT = "/tmp/venv"   # keep the venv off NFS
UV_LINK_MODE = "copy"

[data]                           # `job mount` mounts these on your laptop (read-only)
# mydata = "/datahome/{{user}}/mydata"

[mounts]                         # extra mounts for jobs: "src:dst[:ro|rw]" (/datahome is always ro)
extra = [
  "/datahome/{{user}}/.cache/uv:/datahome/{{user}}/.cache/uv:rw",
]
"""

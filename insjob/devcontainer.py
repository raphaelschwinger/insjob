"""Turn a project's .devcontainer/devcontainer.json into a batch-job container spec.

The project repo is the single source of truth for the image: insjob only reads its
devcontainer.json + Dockerfile. Everything VS Code-specific (extensions, lifecycle
hooks, `--gpus all`) is dropped; GPUs, CPUs and memory are set by insjob.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path

import json5

# runArgs that are safe to carry over into a batch job; everything else is dropped
_KEEP_WITH_VALUE = {
    "--shm-size",
    "--ipc",
    "--ulimit",
    "--cap-add",
    "--security-opt",
    "--device",
    "-e",
    "--env",
    "--add-host",
}
_KEEP_FLAGS = {"--init"}

# mounts that make no sense (or leak secrets) in a batch job
_SKIP_MOUNT = re.compile(r"(/\.ssh|\.claude|claude-code|/\.gitconfig|docker\.sock)")


@dataclass
class ContainerSpec:
    image: str | None = None  # prebuilt image (devcontainer "image")
    dockerfile: str | None = None  # relative to repo root
    context: str = "."  # relative to repo root
    build_args: dict[str, str] = field(default_factory=dict)
    target: str | None = None
    workspace_folder: str = "/workspace"
    run_args: list[str] = field(default_factory=list)
    mounts: list[str] = field(default_factory=list)  # docker --mount values
    env: dict[str, str] = field(default_factory=dict)
    remote_user: str | None = None
    image_hash: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ContainerSpec:
        return cls(**d)


_VAR = re.compile(r"\$\{([^}]+)\}")


def substitute(s: str, variables: dict[str, str], env: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        expr = m.group(1)
        if expr in variables:
            return variables[expr]
        if expr.startswith(("localEnv:", "env:")):
            _, rest = expr.split(":", 1)
            name, _, default = rest.partition(":")
            return env.get(name, default)
        if expr.startswith("containerEnv:"):
            return m.group(0)  # resolved by the container itself
        return m.group(0)

    return _VAR.sub(repl, s)


def load(path: Path) -> dict:
    return json5.loads(path.read_text())


def _parse_mount(m: str | dict) -> dict:
    if isinstance(m, dict):
        return dict(m)
    out: dict = {}
    for part in m.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip() if v else True
    return out


def _mount_str(d: dict) -> str:
    src = d.get("source") or d.get("src")
    dst = d.get("target") or d.get("dst") or d.get("destination")
    parts = [f"type={d.get('type', 'bind')}", f"src={src}", f"dst={dst}"]
    if d.get("readonly") or d.get("ro"):
        parts.append("readonly")
    return ",".join(parts)


def filter_run_args(args: list[str], notes: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        a = str(args[i])
        key, eq, val = a.partition("=")
        if key in _KEEP_FLAGS:
            out.append(a)
        elif key in _KEEP_WITH_VALUE:
            if eq:
                out.append(a)
            elif i + 1 < len(args):
                out += [a, str(args[i + 1])]
                i += 1
        else:
            # skip the flag (and its value, if the next token is not a flag)
            if not eq and i + 1 < len(args) and not str(args[i + 1]).startswith("-"):
                notes.append(f"dropped runArg {a} {args[i + 1]}")
                i += 1
            else:
                notes.append(f"dropped runArg {a}")
        i += 1
    return out


def resolve(
    dc: dict,
    *,
    devcontainer_rel: str,
    repo_name: str,
    host_workspace: str,
    local_env: dict[str, str],
) -> ContainerSpec:
    """Resolve a parsed devcontainer.json for one job.

    devcontainer_rel: path of devcontainer.json relative to the repo root
    host_workspace:   where the code snapshot lives on the server
    local_env:        values for ${localEnv:*} — the *server* user (UID, GID, USER, HOME)
    """
    notes: list[str] = []
    if "dockerComposeFile" in dc:
        raise ValueError("docker-compose devcontainers are not supported (yet)")

    dc_dir = Path(devcontainer_rel).parent
    variables = {
        "localWorkspaceFolder": host_workspace,
        "localWorkspaceFolderBasename": repo_name,
        "devcontainerId": f"insjob-{repo_name}",
    }
    ws = substitute(dc.get("workspaceFolder", f"/workspaces/{repo_name}"), variables, local_env)
    variables["containerWorkspaceFolder"] = ws
    variables["containerWorkspaceFolderBasename"] = Path(ws).name

    def sub(x: str) -> str:
        return substitute(str(x), variables, local_env)

    spec = ContainerSpec(workspace_folder=ws, notes=notes)
    build = dc.get("build") or {}
    if "dockerFile" in dc and not build:  # legacy key
        build = {"dockerfile": dc["dockerFile"], "context": dc.get("context", ".")}
    if build:
        spec.dockerfile = str(dc_dir / build.get("dockerfile", "Dockerfile"))
        spec.context = str((dc_dir / build.get("context", ".")))
        spec.build_args = {k: sub(v) for k, v in (build.get("args") or {}).items()}
        spec.target = build.get("target")
    elif dc.get("image"):
        spec.image = sub(dc["image"])
    else:
        raise ValueError("devcontainer.json has neither build nor image")

    spec.run_args = filter_run_args([sub(a) for a in dc.get("runArgs", [])], notes)

    for m in dc.get("mounts", []):
        d = _parse_mount(m)
        d = {k: sub(v) if isinstance(v, str) else v for k, v in d.items()}
        s = _mount_str(d)
        if d.get("type", "bind") != "bind":
            notes.append(f"dropped volume mount {s}")
        elif _SKIP_MOUNT.search(s):
            notes.append(f"dropped mount {s}")
        else:
            spec.mounts.append(s)

    for key in ("containerEnv", "remoteEnv"):
        for k, v in (dc.get(key) or {}).items():
            if v is not None:
                spec.env[k] = sub(v)

    spec.remote_user = dc.get("remoteUser") or dc.get("containerUser")
    return spec


def dockerfile_declares_uid_args(text: str) -> bool:
    return bool(re.search(r"^\s*ARG\s+USER_UID\b", text, re.M)) and bool(
        re.search(r"^\s*ARG\s+USER_GID\b", text, re.M)
    )


def apply_uid_build_args(spec: ContainerSpec, dockerfile_text: str, uid: int, gid: int) -> None:
    """Case 1: the Dockerfile takes USER_UID/USER_GID — build it for this user directly."""
    if spec.dockerfile and dockerfile_declares_uid_args(dockerfile_text):
        spec.build_args["USER_UID"] = str(uid)
        spec.build_args["USER_GID"] = str(gid)


def compute_hash(spec: ContainerSpec, tree_ids: list[str]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([spec.image, spec.dockerfile, spec.context, spec.build_args,
                         spec.target], sort_keys=True).encode())
    for t in tree_ids:
        h.update(t.encode())
    return h.hexdigest()[:12]


def uid_layer_dockerfile(base: str, uid: int, gid: int, user: str, remote_user: str | None) -> str:
    """A thin layer on top of the project image so the job runs as the server user.

    Handles all three cases at build time:
      1. a user with this UID already exists (Dockerfile took USER_UID) -> nothing to do
      2. the image has a remoteUser at another UID -> remap it (like updateRemoteUserUID)
      3. root-only image -> create the user
    """
    ru = shlex.quote(remote_user or "")
    script = f"""
set -e
UID_={uid}; GID_={gid}; NAME={shlex.quote(user)}; RU={ru}
if [ "$RU" = root ]; then RU=; fi
have() {{ command -v "$1" >/dev/null 2>&1; }}
if ! getent group "$GID_" >/dev/null; then
  if [ -n "$RU" ] && id "$RU" >/dev/null 2>&1 && [ "$(id -u "$RU")" != "$UID_" ]; then
    OLDG=$(id -g "$RU"); GN=$(getent group "$OLDG" | cut -d: -f1)
    if [ "$OLDG" != 0 ] && have groupmod; then groupmod -g "$GID_" "$GN"; else
      groupadd -g "$GID_" "$NAME" 2>/dev/null || echo "$NAME:x:$GID_:" >> /etc/group; fi
  else
    groupadd -g "$GID_" "$NAME" 2>/dev/null || echo "$NAME:x:$GID_:" >> /etc/group
  fi
fi
if getent passwd "$UID_" >/dev/null; then
  echo "insjob: uid $UID_ already present ($(getent passwd "$UID_" | cut -d: -f1))"
elif [ -n "$RU" ] && id "$RU" >/dev/null 2>&1; then
  OLD=$(id -u "$RU"); HOMED=$(getent passwd "$RU" | cut -d: -f6)
  usermod -u "$UID_" -g "$GID_" "$RU"
  find / -xdev \\( -path /proc -o -path /sys -o -path /dev \\) -prune -o \\
       -uid "$OLD" -print0 2>/dev/null | xargs -0 -r chown -h "$UID_:$GID_"
  echo "insjob: remapped $RU $OLD -> $UID_"
else
  if have useradd; then useradd -m -u "$UID_" -g "$GID_" -s /bin/bash "$NAME"; else
    echo "$NAME:x:$UID_:$GID_::/home/$NAME:/bin/sh" >> /etc/passwd; mkdir -p /home/$NAME;
    chown "$UID_:$GID_" /home/$NAME; fi
  echo "insjob: created user $NAME ($UID_)"
fi
"""
    b64 = base64.b64encode(script.strip().encode()).decode()
    return (
        f"FROM {base}\n"
        "USER root\n"
        f"RUN echo {b64} | base64 -d | sh\n"
        f"USER {uid}:{gid}\n"
        f'LABEL insjob.uid="{uid}" insjob.user="{user}"\n'
    )

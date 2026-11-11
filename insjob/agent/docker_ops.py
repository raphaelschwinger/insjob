"""Building images from the project's .devcontainer and running job containers."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import IO

from ..devcontainer import ContainerSpec, uid_layer_dockerfile
from ..identity import Identity
from ..policy import reservation

DOCKER = os.environ.get("INSJOB_DOCKER", "docker")


class DockerError(RuntimeError):
    pass


def docker(*args: str, input: str | None = None, check: bool = True,
           log: IO | None = None) -> subprocess.CompletedProcess:
    if log is not None:
        log.write(f"[insjob] $ docker {' '.join(shlex.quote(a) for a in args)}\n")
        log.flush()
        p = subprocess.run([DOCKER, *args], input=input, text=True, stdout=log,
                           stderr=subprocess.STDOUT)
    else:
        p = subprocess.run([DOCKER, *args], input=input, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise DockerError(f"docker {args[0]} failed ({p.returncode}): {(p.stderr or '').strip()}")
    return p


def image_exists(tag: str) -> bool:
    return docker("image", "inspect", tag, check=False).returncode == 0


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", s.lower()).strip("-.") or "x"


def image_tags(job: dict, ident: Identity) -> tuple[str, str]:
    spec = ContainerSpec.from_dict(job["spec"])
    repo = f"insjob/{_sanitize(ident.user)}/{_sanitize(job['repo'])}"
    base = spec.image if spec.image else f"{repo}:{spec.image_hash}"
    return base, f"{repo}:{spec.image_hash}-uid{ident.uid}"


def ensure_image(job: dict, ident: Identity, log: IO) -> str:
    spec = ContainerSpec.from_dict(job["spec"])
    base, final = image_tags(job, ident)
    if image_exists(final):
        log.write(f"[insjob] image {final} (cached)\n")
        return final
    code = Path(job["code_dir"])
    if spec.image:
        if not image_exists(base):
            docker("pull", base, log=log)
    else:
        args = ["build", "-f", str(code / spec.dockerfile), "-t", base,
                "--label", f"insjob.user={ident.user}", "--label", f"insjob.repo={job['repo']}"]
        for k, v in spec.build_args.items():
            args += ["--build-arg", f"{k}={v}"]
        if spec.target:
            args += ["--target", spec.target]
        args.append(str(code / spec.context))
        log.write(f"[insjob] building {base} from {spec.dockerfile}\n")
        docker(*args, log=log)
    dockerfile = uid_layer_dockerfile(base, ident.uid, ident.gid, ident.user, spec.remote_user)
    log.write(f"[insjob] adding uid layer for {ident.user} ({ident.uid}:{ident.gid})\n")
    docker("build", "-t", final, "-", input=dockerfile, log=log)
    return final


def command_string(cmd: list[str]) -> str:
    """One argument is taken as a shell string (`"a && b"`), several are quoted."""
    if len(cmd) == 1:
        return cmd[0]
    return shlex.join(cmd)


def job_script(job: dict, ident: Identity, *, tee: bool = True) -> str:
    spec = ContainerSpec.from_dict(job["spec"])
    gpus = ",".join(map(str, job.get("gpu_ids") or [])) or "none"
    lines = [
        f'echo "[insjob] job {job["id"]} ({job.get("kind", "run")}) on {job.get("host")} '
        f'gpus={gpus} as $(id -un 2>/dev/null || echo "?") uid=$(id -u)"',
        f'if [ "$(id -u)" != "{ident.uid}" ]; then echo "[insjob] ERROR: running as uid '
        f'$(id -u), expected {ident.uid}"; exit 97; fi',
        f"cd {shlex.quote(spec.workspace_folder)} || exit 96",
    ]
    if job.get("setup"):
        lines.append(f"printf '%s\\n' {shlex.quote('[insjob] setup: ' + job['setup'])}")
        lines.append(f"{{ {job['setup']} ; }} || {{ echo '[insjob] setup failed'; exit 98; }}")
    cmd = command_string(job["cmd"])
    lines.append(f"printf '%s\\n' {shlex.quote('[insjob] run: ' + cmd)}")
    lines.append(cmd)
    body = "\n".join(lines)
    if not tee:
        return body
    return "set -o pipefail\n{\n" + body + "\n} 2>&1 | tee -a /insjob-run/log"


def _mount_arg(spec: str, ident: Identity) -> str:
    """'src:dst[:ro|rw]' -> --mount value. Fails early when the source is missing."""
    parts = ident.expand(spec).split(":")
    src, dst = parts[0], parts[1] if len(parts) > 1 else parts[0]
    mode = parts[2] if len(parts) > 2 else "ro"
    own = (f"/datahome/{ident.user}/", f"{ident.home}/", f"{ident.state_root}/")
    if mode == "rw" and not os.path.exists(src) and src.startswith(own):
        os.makedirs(src, exist_ok=True)  # e.g. a cache dir in the user's own space
    if not os.path.exists(src):  # also triggers autofs
        raise DockerError(f"mount source does not exist on this host: {src}")
    m = f"type=bind,src={src},dst={dst}"
    if mode == "ro":
        m += ",readonly"
    return m


def run_args(job: dict, ident: Identity, *, image: str, detach: bool = True,
             interactive: bool = False, tty: bool = False, script: str | None = None) -> list[str]:
    spec = ContainerSpec.from_dict(job["spec"])
    jid = job["id"]
    gpu_ids = job.get("gpu_ids") or []
    run_dir = Path(job["run_dir"])
    args = ["run"]
    if detach:
        args.append("-d")
    else:
        args.append("--rm")
    if interactive:
        args.append("-i")
    if tty:
        args.append("-t")
    args += [
        "--name", f"job-{ident.user}-{jid}",
        "--hostname", f"job-{jid}",
        "--label", f"insjob.user={ident.user}",
        "--label", f"insjob.id={jid}",
        "--label", f"insjob.kind={job.get('kind', 'run')}",
        "--label", f"insjob.repo={job['repo']}",
        "--label", f"insjob.sha={job.get('sha', '')[:12]}",
        "--label", f"insjob.cmd={command_string(job['cmd'])[:200]}",
        "--user", f"{ident.uid}:{ident.gid}",
        "--init", "-e", "TINI_KILL_PROCESS_GROUP=1",
        "--cpus", str(job["cpus"]),
        "--memory", str(job["memory"]),
        "--memory-reservation", reservation(job),
    ]
    if gpu_ids:
        ids = ",".join(map(str, gpu_ids))
        args += ["--gpus", f'"device={ids}"', "-e", f"NVIDIA_VISIBLE_DEVICES={ids}"]
    else:
        args += ["-e", "NVIDIA_VISIBLE_DEVICES=void"]
    if not any(a.startswith("--shm-size") for a in spec.run_args):
        args += ["--shm-size", str(job.get("shm_size", "16g"))]
    args += spec.run_args
    args += [
        "--mount", f"type=bind,src={job['code_dir']},dst={spec.workspace_folder}",
        "--mount", f"type=bind,src={run_dir},dst=/insjob-run",
        "--mount", f"type=bind,src={run_dir / 'outputs'},dst=/insjob-out",
    ]
    for root in ("/datahome", "/data-project"):
        if os.path.isdir(root):
            args += ["--mount", f"type=bind,src={root},dst={root},readonly,"
                                "bind-propagation=rslave"]
    for m in spec.mounts:
        args += ["--mount", m]
    for m in job.get("mounts") or []:
        args += ["--mount", _mount_arg(m, ident)]
    secrets = Path(ident.state_root) / "secrets.env"
    if secrets.exists():
        args += ["--env-file", str(secrets)]
    env = {"PYTHONUNBUFFERED": "1", **spec.env, **(job.get("env") or {}),
           "INSJOB_ID": str(jid), "INSJOB_HOST": str(job.get("host")),
           "INSJOB_OUT": "/insjob-out"}
    for k, v in env.items():
        args += ["-e", f"{k}={ident.expand(str(v))}"]
    if job.get("kind") == "notebook":
        args += ["-p", "127.0.0.1::8888"]
    args += ["-w", spec.workspace_folder, image, "bash", "-lc",
             script if script is not None else job_script(job, ident)]
    return args


def run_job(job: dict, ident: Identity, image: str, log: IO) -> str:
    name = f"job-{ident.user}-{job['id']}"
    docker("rm", "-f", name, check=False)
    p = docker(*run_args(job, ident, image=image))
    log.write(f"[insjob] started container {name} ({p.stdout.strip()[:12]})\n")
    return name


def list_job_containers(user: str) -> dict[str, dict]:
    p = docker("ps", "-a", "--filter", f"label=insjob.user={user}", "--filter",
               "label=insjob.id", "--format", "{{.Names}}", check=False)
    names = [n for n in p.stdout.split() if n]
    if not names:
        return {}
    p = docker("inspect", *names, check=False)
    out = {}
    try:
        for c in json.loads(p.stdout or "[]"):
            out[c["Name"].lstrip("/")] = {
                "id": c["Id"],
                "status": c["State"]["Status"],
                "exit_code": c["State"]["ExitCode"],
                "labels": c["Config"].get("Labels") or {},
                "ports": c["NetworkSettings"].get("Ports") or {},
            }
    except json.JSONDecodeError:
        pass
    return out


def all_containers() -> dict[str, dict]:
    """Full container id -> name + insjob labels, for naming GPU processes."""
    fmt = '{{.ID}}\t{{.Names}}\t{{.Label "insjob.user"}}\t{{.Label "insjob.id"}}'
    p = docker("ps", "--no-trunc", "--format", fmt, check=False)
    out = {}
    for line in p.stdout.splitlines():
        cid, name, user, jid = (line.split("\t") + ["", "", "", ""])[:4]
        out[cid] = {"name": name, "insjob_user": user or None, "insjob_id": jid or None}
    return out


def host_port(info: dict, cport: str = "8888/tcp") -> int | None:
    b = (info.get("ports") or {}).get(cport)
    if b:
        return int(b[0]["HostPort"])
    return None


def stop(name: str, timeout: int = 30) -> None:
    docker("stop", "-t", str(timeout), name, check=False)


def remove(name: str) -> None:
    docker("rm", "-f", name, check=False)

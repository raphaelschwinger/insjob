"""SSH plumbing for the laptop CLI: commands, agent ops, code snapshots, tunnels, install."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import tarfile
import time
from pathlib import Path

from . import __version__
from .config import LaptopConfig
from .policy import fqdn

CACHE = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "insjob"


class RemoteError(RuntimeError):
    pass


class Remote:
    def __init__(self, cfg: LaptopConfig):
        self.cfg = cfg
        self.ident = cfg.identity

    # ---- ssh ------------------------------------------------------------------------
    def target(self, host: str) -> str:
        return f"{self.ident.user}@{fqdn(self.cfg.lab, host)}"

    def ssh_args(self, host: str, *, tty: bool = False, forward_agent: bool = False,
                 batch: bool = True) -> list[str]:
        ctl = Path("~/.ssh").expanduser() / "insjob-%C"
        a = ["ssh", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
             "-o", "ControlMaster=auto", "-o", f"ControlPath={ctl}", "-o", "ControlPersist=10m"]
        if batch:
            a += ["-o", "BatchMode=yes"]
        if tty:
            a.append("-t")
        if forward_agent:
            a.append("-A")
        return [*a, self.target(host)]

    def run(self, host: str, command: str, *, input: str | bytes | None = None,
            check: bool = True, capture: bool = True, tty: bool = False,
            forward_agent: bool = False) -> subprocess.CompletedProcess:
        args = [*self.ssh_args(host, tty=tty, forward_agent=forward_agent), command]
        p = subprocess.run(args, input=input, capture_output=capture,
                           text=not isinstance(input, bytes))
        if check and p.returncode != 0:
            err = (p.stderr or "").strip() if capture else ""
            raise RemoteError(f"{host}: `{command[:80]}` failed ({p.returncode}) {err}")
        return p

    def reachable(self, host: str) -> bool:
        return self.run(host, "true", check=False).returncode == 0

    # ---- agent ops ----------------------------------------------------------------------
    def docker_exec(self, *, tty: bool = False) -> str:
        flags = "-it" if tty else "-i"
        return f"docker exec {flags} {self.ident.agent_name} insjob-remote"

    def agent_op(self, op: str, *args: str, payload: dict | None = None,
                 host: str | None = None) -> dict | list | None:
        hosts = [host] if host else self.api_hosts()
        last = None
        for h in hosts:
            cmd = f"{self.docker_exec()} {op} {' '.join(shlex.quote(a) for a in args)}"
            p = self.run(h, cmd, input=json.dumps(payload or {}), check=False)
            if p.returncode == 0:
                return json.loads(p.stdout or "null")
            last = (p.stderr or p.stdout).strip()
            retry = ("No such container", "is not running", "Connection", "timed out",
                     "resolve hostname")
            if not any(r in last for r in retry):
                break  # a real error from the op itself — don't hide it behind another host
        raise RemoteError(f"agent op {op} failed: {last}\n"
                          "Is the agent installed? `job agent status` / `job agent install`")

    def sync_users(self, host: str | None = None) -> int:
        """Copy the server's account list (LDAP via getent) into the lab-wide uid -> name
        cache, so dashboards can name the owner of any GPU process. Best effort."""
        host = host or self.cfg.host
        cmd = "getent passwd | awk -F: '$3>=1000 && $3<65534 {print $3\":\"$1}'"
        p = self.run(host, cmd, check=False)
        names = dict(ln.split(":", 1) for ln in (p.stdout or "").splitlines() if ":" in ln)
        if not names:
            return 0
        try:
            self.agent_op("add_users", payload=names, host=host)
        except RemoteError:
            return 0
        return len(names)

    def api_hosts(self) -> list[str]:
        p = self.cfg.policy
        return [self.cfg.host] + [h for h in p["hosts"] if h != self.cfg.host]

    # ---- code snapshot ---------------------------------------------------------------
    def prepare_code(self, host: str, *, repo: str, url: str, sha: str, code_dir: str) -> None:
        mirror = f"{self.ident.state_root}/repos/{repo}.git"
        script = f"""
set -euo pipefail
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
export GIT_TERMINAL_PROMPT=0
MIRROR={shlex.quote(mirror)}; URL={shlex.quote(url)}; SHA={shlex.quote(sha)}
DST={shlex.quote(code_dir)}
mkdir -p "$(dirname "$MIRROR")"
if [ ! -d "$MIRROR" ]; then git clone -q --mirror "$URL" "$MIRROR"
else git -C "$MIRROR" remote set-url origin "$URL"; git -C "$MIRROR" fetch -q --prune origin; fi
git -C "$MIRROR" cat-file -e "$SHA^{{commit}}" || {{ echo "commit $SHA not on remote" >&2; exit 3; }}
rm -rf "$DST"
git clone -q --shared --no-checkout "$MIRROR" "$DST"
git -C "$DST" remote set-url origin "$URL"
git -C "$DST" -c advice.detachedHead=false checkout -q "$SHA"
if [ -f "$DST/.gitmodules" ]; then git -C "$DST" submodule update -q --init --recursive; fi
echo ok
"""
        p = self.run(host, "bash -s", input=script, check=False, forward_agent=True)
        if p.returncode != 0:
            raise RemoteError(f"preparing code on {host} failed:\n{p.stderr.strip()}")

    # ---- tunnels -------------------------------------------------------------------------
    def tunnel(self, host: str, remote_port: int, key: str) -> int:
        """Background `ssh -N -L` to 127.0.0.1:<remote_port> on host; reused while alive."""
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / "tunnels.json"
        try:
            tunnels = json.loads(f.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            tunnels = {}
        t = tunnels.get(key)
        if t and t.get("host") == host and t.get("rport") == remote_port and _alive(t["pid"]) \
                and _port_open(t["lport"]):
            return t["lport"]
        if t and _alive(t["pid"]):
            os.kill(t["pid"], signal.SIGTERM)
        lport = _free_port()
        args = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes",
                "-o", "ServerAliveInterval=15", "-o", "ControlMaster=no", "-o", "ControlPath=none",
                "-L", f"{lport}:127.0.0.1:{remote_port}", self.target(host)]
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(50):
            if _port_open(lport):
                break
            if proc.poll() is not None:
                raise RemoteError(f"ssh tunnel to {host}:{remote_port} failed")
            time.sleep(0.2)
        tunnels[key] = {"pid": proc.pid, "lport": lport, "host": host, "rport": remote_port}
        f.write_text(json.dumps(tunnels))
        return lport

    def close_tunnels(self) -> int:
        f = CACHE / "tunnels.json"
        try:
            tunnels = json.loads(f.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return 0
        n = 0
        for t in tunnels.values():
            if _alive(t["pid"]):
                os.kill(t["pid"], signal.SIGTERM)
                n += 1
        f.unlink()
        return n

    # ---- agent install -------------------------------------------------------------
    def agent_image(self) -> tuple[str, bytes]:
        pkg = Path(__file__).parent
        buf = io.BytesIO()
        h = hashlib.sha256()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(pkg / "agent" / "Dockerfile", arcname="Dockerfile")
            for p in sorted(pkg.rglob("*")):
                if "__pycache__" in p.parts or p.suffix == ".pyc" or p.is_dir():
                    continue
                rel = Path("insjob") / p.relative_to(pkg)
                tar.add(p, arcname=str(rel))
                h.update(str(rel).encode())
                h.update(p.read_bytes())
        return f"insjob/agent:{__version__}-{h.hexdigest()[:10]}", buf.getvalue()

    def install_agent(self, host: str, log=print) -> None:
        tag, ctx = self.agent_image()
        have = self.run(host, f"docker image inspect {tag} >/dev/null 2>&1 && echo yes",
                        check=False).stdout.strip()
        if have != "yes":
            log(f"  building {tag} on {host} …")
            p = self.run(host, f"docker build -q -t {tag} -", input=ctx, check=False)
            if p.returncode != 0:
                raise RemoteError(f"agent build failed on {host}:\n"
                                  f"{p.stderr.decode(errors='replace')[-2000:]}")
        i = self.ident
        env = {
            "INSJOB_HOST": host, "INSJOB_USER": i.user, "INSJOB_UID": i.uid,
            "INSJOB_GID": i.gid, "INSJOB_STATE": i.state_root, "INSJOB_PORT": i.port,
            "INSJOB_GROUPS": ",".join(i.groups), "INSJOB_DOCKER_GID": i.docker_gid or "",
            "HOME": i.home,
        }
        env_args = " ".join(f"-e {k}={shlex.quote(str(v))}" for k, v in env.items())
        group_add = f"--group-add {i.docker_gid}" if i.docker_gid else ""
        script = f"""
set -e
mkdir -p /var/tmp/insjob && chmod 1777 /var/tmp/insjob 2>/dev/null || true
ls -d /datahome/{i.user} >/dev/null 2>&1 || true
mkdir -p {shlex.quote(i.state_root)} && chmod 700 {shlex.quote(i.state_root)}
EXTRA=""
if [ -d /data-project ]; then
  EXTRA="--mount type=bind,src=/data-project,dst=/data-project,readonly,bind-propagation=rslave"
fi
GPUS=""
if command -v nvidia-smi >/dev/null 2>&1; then GPUS="--gpus all"; fi
docker rm -f {i.agent_name} >/dev/null 2>&1 || true
docker run -d --name {i.agent_name} --restart unless-stopped \\
  --label insjob.user={i.user} --label insjob.kind=agent \\
  --user {i.uid}:{i.gid} {group_add} --pid=host $GPUS \\
  --memory 2g --cpus 1 {env_args} \\
  -v /var/run/docker.sock:/var/run/docker.sock \\
  --mount type=bind,src={i.home},dst={i.home} \\
  --mount type=bind,src=/datahome,dst=/datahome,bind-propagation=rslave \\
  --mount type=bind,src=/var/tmp/insjob,dst=/var/tmp/insjob \\
  $EXTRA -p 127.0.0.1:{i.port}:{i.port} {tag} >/dev/null
echo started
"""
        self.run(host, "bash -s", input=script)

    def agent_status(self, host: str) -> dict:
        p = self.run(host, f"docker inspect -f '{{{{.State.Status}}}} {{{{.Config.Image}}}}' "
                           f"{self.ident.agent_name}", check=False)
        if p.returncode == 255:  # ssh itself failed
            return {"host": host, "status": "unreachable"}
        if p.returncode != 0:
            return {"host": host, "status": "not installed"}
        status, image = (p.stdout.strip().split() + ["", ""])[:2]
        out = {"host": host, "status": status, "image": image}
        if status == "running":
            try:
                out["ping"] = self.agent_op("ping", host=host)
            except RemoteError as e:
                out["error"] = str(e)
        return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None

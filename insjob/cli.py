"""`job` — the laptop CLI."""

from __future__ import annotations

import json
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import time
import tomllib
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from . import devcontainer as dcmod
from .config import CONFIG_PATH, LaptopConfig, NotSetUp, RepoConfig, git, repo_template
from .identity import DISCOVER_SCRIPT, parse_discover
from .mount import is_stale, mount_one, mounted, register_with_claude, unmount
from .policy import DEFAULT_LAB, clamp_resources, deep_merge, fqdn, user_policy
from .remote import Remote, RemoteError

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Run training, data processing and notebooks on the INS GPU servers.")
agent_app = typer.Typer(no_args_is_help=True, help="Manage your agent on the servers.")
data_app = typer.Typer(no_args_is_help=True, help="Look at / copy data on the servers.")
app.add_typer(agent_app, name="agent")
app.add_typer(data_app, name="data")

con = Console()
err = Console(stderr=True)
PASS = {"allow_extra_args": True, "ignore_unknown_options": True}


def die(msg: str, code: int = 1):
    err.print(f"[red]error:[/] {msg}")
    raise typer.Exit(code)


def ctx() -> tuple[LaptopConfig, Remote]:
    try:
        cfg = LaptopConfig.load()
    except NotSetUp as e:
        die(str(e))
    return cfg, Remote(cfg)


def ago(ts: float | None) -> str:
    if not ts:
        return ""
    s = int(time.time() - ts)
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= n:
            return f"{s // n}{unit}"
    return f"{s}s"


def dur(a: float | None, b: float | None) -> str:
    if not a:
        return ""
    s = int((b or time.time()) - a)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ===================================================================================
# building a job
# ===================================================================================
def git_commit(root: Path, *, push: bool, interactive: bool) -> tuple[str, str, str]:
    """Return (sha, branch, url) of what will run — the code must be on the remote."""
    dirty = git("status", "--porcelain", "-uno", "--ignore-submodules=none", cwd=root)
    if dirty and not interactive:
        die("uncommitted changes — commit them first (the server checks out your commit):\n"
            + dirty)
    if dirty:
        err.print("[yellow]note:[/] uncommitted changes are not on the server")
    url = git("remote", "get-url", "origin", cwd=root)

    def unpushed() -> str:
        own = git("rev-list", "-n1", "HEAD", "--not", "--remotes", cwd=root)
        subs = git("submodule", "foreach", "--quiet", "--recursive",
                   "git rev-list -n1 HEAD --not --remotes", cwd=root, check=False)
        return own or subs.strip()

    if unpushed():
        if push:
            has_up = git("rev-parse", "--abbrev-ref", "@{u}", cwd=root, check=False)
            err.print("pushing …")
            subprocess.run(["git", "push", "--recurse-submodules=on-demand"]
                           + ([] if has_up else ["-u", "origin", "HEAD"]), cwd=root, check=True)
        elif interactive and git("rev-parse", "@{u}", cwd=root, check=False):
            sha = git("rev-parse", "@{u}", cwd=root)
            err.print(f"[yellow]note:[/] HEAD is not pushed — using the pushed {sha[:8]}")
            return sha, git("rev-parse", "--abbrev-ref", "HEAD", cwd=root), url
        else:
            die("HEAD (or a submodule) is not pushed — `git push`, or pass --push")
    return (git("rev-parse", "HEAD", cwd=root),
            git("rev-parse", "--abbrev-ref", "HEAD", cwd=root), url)


def resolve_spec(cfg: LaptopConfig, repo: RepoConfig, sha: str, code_dir: str
                 ) -> dcmod.ContainerSpec:
    rel = repo.devcontainer
    try:
        text = git("show", f"{sha}:{rel}", cwd=repo.root)
    except RuntimeError:
        die(f"{rel} not found in commit {sha[:8]} — the job image is built from the repo's "
            "devcontainer (start from github.com/raphaelschwinger/cuda-python-devcontainer)")
    i = cfg.identity
    local_env = {"UID": str(i.uid), "GID": str(i.gid), "USER": i.user, "USERNAME": i.user,
                 "HOME": i.home}
    try:
        spec = dcmod.resolve(dcmod.json5.loads(text), devcontainer_rel=rel, repo_name=repo.name,
                             host_workspace=code_dir, local_env=local_env)
    except ValueError as e:
        die(str(e))
    trees = [git("rev-parse", f"{sha}:{Path(rel).parent}", cwd=repo.root)]
    if spec.dockerfile:
        df = git("show", f"{sha}:{spec.dockerfile}", cwd=repo.root, check=False)
        dcmod.apply_uid_build_args(spec, df, i.uid, i.gid)
        ctx_rel = str(Path(spec.context))
        if re.search(r"^\s*(COPY|ADD)\s+(?!--from)", df, re.M | re.I) and \
                ctx_rel != str(Path(rel).parent):
            p = "" if ctx_rel == "." else ctx_rel
            trees.append(git("rev-parse", f"{sha}:{p}", cwd=repo.root))
    spec.image_hash = dcmod.compute_hash(spec, trees)
    return spec


def make_job(cfg: LaptopConfig, remote: Remote, *, kind: str, cmd: list[str], gpus: int | None,
             cpus: int | None, memory: str | None, min_mem: float | None, host: str | None,
             after: list[int], push: bool, api_host: str | None = None,
             gpu_ids: list[int] | None = None) -> tuple[dict, str]:
    repo = RepoConfig.load()
    interactive = kind in ("exec", "shell")
    sha, branch, url = git_commit(repo.root, push=push, interactive=interactive)
    r = {**cfg.lab["resources"], **repo.resources}
    if gpu_ids:
        if gpus is not None and gpus != len(gpu_ids):
            die(f"-g {gpus} does not match --gpu-ids {','.join(map(str, gpu_ids))}")
        gpus = len(gpu_ids)
    gpus = int(r.get("gpus", 1) if gpus is None else gpus)
    cpus_, mem_ = clamp_resources(cfg.lab, cpus or r.get("cpus", 8), memory or r.get("memory"))
    hosts = [host] if host and host != "any" else (repo.hosts or ["any"])
    pol = cfg.policy
    bad = [h for h in hosts if h != "any" and h not in pol["hosts"]]
    if bad:
        die(f"host {bad[0]} is not allowed for group '{pol['group']}' "
            f"(allowed: {', '.join(pol['hosts'])})")
    h = api_host or (hosts[0] if hosts[0] != "any" else cfg.host)
    ids = remote.agent_op("new_id", host=h)
    err.print(f"[dim]job {ids['id']}: preparing {repo.name}@{sha[:8]} on {h} …[/]")
    remote.prepare_code(h, repo=repo.name, url=url, sha=sha, code_dir=ids["code_dir"])
    spec = resolve_spec(cfg, repo, sha, ids["code_dir"])
    for n in spec.notes:
        err.print(f"[dim]  devcontainer: {n}[/]")
    job = {
        "id": ids["id"], "user": cfg.identity.user, "repo": repo.name, "kind": kind,
        "cmd": cmd, "setup": repo.setup, "env": repo.env, "mounts": repo.mounts,
        "gpus": gpus, "cpus": cpus_, "memory": mem_,
        "memory_reservation": r.get("memory_reservation", "32g"),
        "shm_size": r.get("shm_size", "16g"),
        "min_gpu_mem_gb": float(min_mem if min_mem is not None else r.get("min_gpu_mem_gb", 0)),
        "gpu_ids_req": list(gpu_ids or []),
        "hosts": hosts, "after": after, "sha": sha, "branch": branch, "remote_url": url,
        "code_dir": ids["code_dir"], "run_dir": ids["run_dir"], "spec": spec.to_dict(),
        "submitted_at": time.time(), "submitted_from": socket.gethostname(),
    }
    return job, h


def parse_gpu_ids(spec: str | None) -> list[int]:
    """`--gpu-ids 1` / `0,1` -> [1] / [0, 1]; empty or None -> []."""
    if not spec:
        return []
    try:
        ids = sorted({int(x) for x in spec.split(",") if x.strip()})
    except ValueError:
        die(f"--gpu-ids takes comma-separated GPU indices, got {spec!r}")
    if any(i < 0 for i in ids):
        die(f"--gpu-ids: negative index in {spec!r}")
    return ids


def fits_now(ov: dict, job: dict) -> str | None:
    """Name of a host where the job could start right now, if any."""
    for h in ov["hosts"]:
        if not h["allowed"] or not h.get("online"):
            continue
        if "any" not in job["hosts"] and h["host"] not in job["hosts"]:
            continue
        if job["gpus"] == 0:
            return h["host"]
        req = job.get("gpu_ids_req") or []
        free = [g for g in h.get("gpus", []) if not g.get("claim") and not g.get("procs")
                and (not req or g["index"] in req)
                and g["mem_used_mb"] < 1024
                and g["mem_total_mb"] >= job["min_gpu_mem_gb"] * 1024 * 0.95]
        if len(free) >= job["gpus"]:
            return h["host"]
    return None


def resolve_id(remote: Remote, jid: str) -> int:
    if jid == "last":
        jobs = remote.agent_op("list", "--limit", "1")
        if not jobs:
            die("no jobs yet")
        return int(jobs[0]["id"])
    try:
        return int(jid)
    except ValueError:
        die(f"not a job id: {jid}")


# ===================================================================================
# commands
# ===================================================================================
@app.command(context_settings=PASS)
def run(
    ctx_: typer.Context,
    gpus: int = typer.Option(None, "-g", "--gpus", help="GPUs (0 = CPU-only data job)"),
    cpus: int = typer.Option(None, "-c", "--cpus"),
    memory: str = typer.Option(None, "-m", "--memory", help="e.g. 64g"),
    min_mem: float = typer.Option(None, "--min-mem", help="min GPU memory in GB"),
    host: str = typer.Option(None, "-H", "--host", help="host name or 'any'"),
    wait: bool = typer.Option(False, "-w", "--wait", help="queue until resources are free"),
    after: list[int] = typer.Option([], "--after", help="start after job ID succeeded"),
    push: bool = typer.Option(False, "--push", help="git push before submitting"),
    gpu_ids: str = typer.Option(None, "--gpu-ids",
                                help="pin GPU indices on the host, e.g. 1 or 0,1"),
):
    """Queue a batch job:  job run [-g 1] -- python train.py experiment=x"""
    cmd = list(ctx_.args)
    if not cmd:
        die("no command — usage: job run [options] -- <command>")
    ids = parse_gpu_ids(gpu_ids)
    if ids and (host is None or host == "any"):
        die("--gpu-ids needs -H <host>: GPU indices only mean something on one server")
    cfg, remote = ctx()
    job, h = make_job(cfg, remote, kind="run", cmd=cmd, gpus=gpus, cpus=cpus, memory=memory,
                      min_mem=min_mem, host=host, after=after, push=push, gpu_ids=ids)
    if not wait and not after:
        ov = remote.agent_op("overview", host=h)
        if not fits_now(ov, job):
            show_gpus(ov)
            die(f"no host has {job['gpus']} free GPU(s) for this job right now — "
                "pass --wait to queue it")
    remote.agent_op("submit", payload=job, host=h)
    con.print(f"[green]queued job {job['id']}[/] ({job['repo']}@{job['sha'][:8]}, "
              f"{job['gpus']} GPU, {job['cpus']} CPU)  →  `job logs {job['id']} -f`")


@app.command(name="exec", context_settings=PASS)
def exec_(
    ctx_: typer.Context,
    gpus: int = typer.Option(0, "-g", "--gpus"),
    cpus: int = typer.Option(None, "-c", "--cpus"),
    memory: str = typer.Option(None, "-m", "--memory"),
    host: str = typer.Option(None, "-H", "--host"),
):
    """Run a one-off command in the repo's container next to the data; output streams back."""
    cmd = list(ctx_.args)
    if not cmd:
        die("usage: job exec [-g 0] -- <command>")
    sys.exit(_interactive("exec", cmd, gpus, cpus, memory, host))


@app.command()
def shell(
    gpus: int = typer.Option(0, "-g", "--gpus"),
    cpus: int = typer.Option(None, "-c", "--cpus"),
    memory: str = typer.Option(None, "-m", "--memory"),
    host: str = typer.Option(None, "-H", "--host"),
):
    """Interactive bash inside the repo's container on a server."""
    sys.exit(_interactive("shell", ["bash"], gpus, cpus, memory, host))


def _pick_host(cfg: LaptopConfig, remote: Remote, gpus: int, host: str | None) -> str:
    if host:
        return host
    if gpus == 0:
        return cfg.host
    ov = remote.agent_op("overview")
    order = sorted(ov["hosts"], key=lambda h: h["host"] != cfg.host)
    found = fits_now({**ov, "hosts": order}, {"gpus": gpus, "hosts": ["any"], "min_gpu_mem_gb": 0})
    if not found:
        show_gpus(ov)
        die(f"no host with {gpus} free GPU(s) right now")
    return found


def _interactive(kind, cmd, gpus, cpus, memory, host) -> int:
    cfg, remote = ctx()
    h = _pick_host(cfg, remote, gpus, host)
    job, h = make_job(cfg, remote, kind=kind, cmd=cmd, gpus=gpus, cpus=cpus, memory=memory,
                      min_mem=None, host=h, after=[], push=False, api_host=h)
    remote.agent_op("prepare", payload=job, host=h)
    tty = sys.stdin.isatty() and sys.stdout.isatty()
    flag = " --tty" if tty else ""
    args = [*remote.ssh_args(h, tty=tty), f"{remote.docker_exec(tty=tty)} interactive "
                                           f"{job['id']}{flag}"]
    return subprocess.call(args)


@app.command()
def notebook(
    gpus: int = typer.Option(1, "-g", "--gpus"),
    cpus: int = typer.Option(None, "-c", "--cpus"),
    memory: str = typer.Option(None, "-m", "--memory"),
    host: str = typer.Option(None, "-H", "--host"),
    timeout: int = typer.Option(1800, help="seconds to wait for the image build / a GPU"),
):
    """Start Jupyter in the repo's container on a server GPU and tunnel it to this laptop."""
    cfg, remote = ctx()
    repo = RepoConfig.load()
    token = secrets.token_hex(16)
    ws = "$PWD"
    default = (
        'if command -v jupyter >/dev/null 2>&1; then J="jupyter"; '
        'elif command -v uv >/dev/null 2>&1; then J="uv run --with jupyterlab jupyter"; '
        'else python -m pip install -q --user jupyterlab && J="python -m jupyter"; fi; '
        '$J lab --no-browser --ip=0.0.0.0 --port=8888 --ServerApp.token="$INSJOB_NB_TOKEN" '
        f"--ServerApp.allow_origin='*' --ServerApp.root_dir={ws}"
    )
    # the token travels as an env var, so it never shows up in labels or the lab status
    cmd = [repo.notebook.get("command", default)]
    job, h = make_job(cfg, remote, kind="notebook", cmd=cmd, gpus=gpus, cpus=cpus,
                      memory=memory, min_mem=None, host=host, after=[], push=False)
    job["env"] = {**job["env"], "INSJOB_NB_TOKEN": token}
    job["notebook"] = {}
    remote.agent_op("submit", payload=job, host=h)
    con.print(f"[green]notebook job {job['id']} queued[/] — waiting for it to start "
              "(first time: the image is built) …")
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        j = remote.agent_op("get", str(job["id"]), host=h)
        if j["state"] != last:
            con.print(f"  {j['state']} {('on ' + j['host']) if j.get('host') else ''}")
            last = j["state"]
        if j["state"] in ("failed", "cancelled", "skipped", "succeeded"):
            die(f"notebook job ended: {j.get('reason', '')} — `job logs {job['id']}`")
        port = (j.get("notebook") or {}).get("port")
        if j["state"] == "running" and port:
            lport = remote.tunnel(j["host"], port, f"nb{job['id']}")
            time.sleep(4)
            url = f"http://127.0.0.1:{lport}/?token={token}"
            con.print(f"\n[bold green]Jupyter is up[/] on {j['host']} GPU {j['gpu_ids']}\n"
                      f"  VS Code: Select Kernel → Existing Jupyter Server → paste:\n"
                      f"  [bold]{url}[/]\n  browser: http://127.0.0.1:{lport}/lab?token={token}\n"
                      f"  stop it with `job stop {job['id']}`")
            return
        time.sleep(3)
    die("timed out waiting for the notebook; it stays queued — `job ls`")


@app.command(name="ls")
def ls_(
    all_: bool = typer.Option(False, "-a", "--all", help="include finished jobs"),
    n: int = typer.Option(20, "-n", help="how many finished jobs to show"),
    as_json: bool = typer.Option(False, "--json"),
):
    """List your jobs."""
    cfg, remote = ctx()
    jobs = remote.agent_op("list", "--limit", str(200 if all_ else 50))
    if not all_:
        active = [j for j in jobs if j["state"] in ("queued", "starting", "running")]
        recent = [j for j in jobs if j not in active][: min(n, 5)]
        jobs = active + recent
    if as_json:
        print(json.dumps(jobs, indent=1))
        return
    t = Table(box=None, header_style="bold dim")
    for c in ("id", "kind", "state", "host", "gpu", "repo", "sha", "command", "time"):
        t.add_column(c)
    colors = {"running": "green", "queued": "yellow", "starting": "cyan", "failed": "red",
              "succeeded": "dim", "cancelled": "dim", "skipped": "dim"}
    for j in jobs:
        st = j["state"]
        when = dur(j.get("started_at"), j.get("ended_at")) if j.get("started_at") else \
            ago(j.get("submitted_at"))
        t.add_row(str(j["id"]), j.get("kind", "run"), f"[{colors.get(st, '')}]{st}[/]",
                  j.get("host") or ",".join(j.get("hosts") or []),
                  ",".join(map(str, j.get("gpu_ids") or [])) or str(j.get("gpus", "")),
                  j.get("repo", ""), (j.get("sha") or "")[:7],
                  " ".join(j.get("cmd") or [])[:50], when)
    con.print(t)


@app.command()
def logs(
    jid: str = typer.Argument("last"),
    follow: bool = typer.Option(False, "-f", "--follow"),
    lines: int = typer.Option(200, "-n"),
):
    """Show (or follow) a job's log."""
    cfg, remote = ctx()
    i = resolve_id(remote, jid)
    j = remote.agent_op("get", str(i))
    if not j:
        die(f"no job {i}")
    host = j.get("host") or cfg.host
    path = shlex.quote(f"{j['run_dir']}/log")
    cmd = f"tail -n {lines} {'-F' if follow else ''} {path} 2>/dev/null || echo '(no log yet)'"
    try:
        subprocess.call([*remote.ssh_args(host), cmd])
    except KeyboardInterrupt:
        pass
    if not follow:
        if j.get("wandb_url"):
            con.print(f"[dim]W&B: {j['wandb_url']}[/]")
        if j.get("reason"):
            con.print(f"[dim]{j['state']}: {j['reason']}[/]")


@app.command()
def stop(jids: list[str] = typer.Argument(..., help="job id(s) or 'last'")):
    """Cancel a queued job or stop a running one."""
    cfg, remote = ctx()
    for jid in jids:
        i = resolve_id(remote, jid)
        r = remote.agent_op("cancel", str(i))
        con.print(f"job {i}: {r['result']}")


def short_cmd(cmd: str | None) -> str:
    """'/opt/venv/bin/python3 -u src/train.py …' -> 'python3 -u src/train.py …'"""
    if not cmd:
        return ""
    exe, _, rest = cmd.partition(" ")
    return (exe.rsplit("/", 1)[-1] + (" " + rest if rest else "")).strip()


def show_gpus(ov: dict, lab_view: bool = True) -> None:
    me = ov["me"]["user"]
    for h in ov["hosts"]:
        title = f"[bold]{h['host']}[/] [dim]{h['gpus_desc']}[/]"
        if not h["allowed"]:
            title += " [dim](not in your policy)[/]"
        if not h.get("online"):
            con.print(f"{title}  [dim]no data (no agent on this host yet)[/]")
            continue
        con.print(f"{title}  [dim]cpu {h['load1']:.0f}/{h['cpus']} load, "
                  f"{h['mem_avail_mb'] // 1024}G free RAM, {ago(h.get('updated'))} ago[/]")
        for g in h.get("gpus", []):
            owners = []
            for p in g.get("procs", []):
                who = p.get("insjob_user") or p.get("user") or "?"
                what = f" job {p['insjob_id']}" if p.get("insjob_id") else ""
                cmd = short_cmd(p.get("cmd")) or p.get("container_name") or ""
                owners.append(f"{who}{what}" + (f" [dim]{escape(cmd[:50])}[/]" if cmd else ""))
            c = g.get("claim")
            if c and not owners:
                owners.append(f"{c['user']} job {c['job']} (starting)")
            owners = sorted(set(owners))
            busy = owners or g["mem_used_mb"] > 1024 or g["util"] > 5
            mark = "[red]●[/]" if busy else "[green]○[/]"
            owners = [f"[bold]{me}[/]{o[len(me):]}" if o.split(" ")[0] == me else o
                      for o in owners]
            first = owners[0] if owners else ("busy" if busy else "free")
            con.print(f"  {mark} {g['index']} {g['name'][:24]:24} "
                      f"{g['mem_used_mb'] / 1024:5.1f}/{g['mem_total_mb'] / 1024:4.0f}G "
                      f"{g['util']:3d}%  {first}")
            for o in owners[1:]:
                con.print(f"{'':50}{o}")
    if lab_view and ov.get("lab"):
        others = [u for u in ov["lab"] if u["jobs"]]
        if others:
            con.print("\n[bold]insjob jobs in the lab[/]")
            for u in others:
                for j in u["jobs"]:
                    con.print(f"  {u['host']:6} {u['user']:12} job {j['id']:<4} {j['kind']:8} "
                              f"gpu {','.join(map(str, j['gpu_ids'])) or '-':4} {j['repo']}  "
                              f"[dim]{j['cmd'][:40]}[/]")


@app.command()
def gpus(as_json: bool = typer.Option(False, "--json")):
    """GPU usage on all servers, and who uses them."""
    _, remote = ctx()
    ov = remote.agent_op("overview")
    unknown = any(str(p.get("user", "")).startswith("uid ") for h in ov["hosts"]
                  for g in h.get("gpus", []) for p in g.get("procs", []))
    if unknown and remote.sync_users(ov.get("served_by")):  # names are cached lab-wide
        ov = remote.agent_op("overview")
    if as_json:
        print(json.dumps(ov["hosts"], indent=1))
        return
    show_gpus(ov)


@app.command()
def ui():
    """Open the dashboard (GPUs, queue, notifications) in the browser."""
    cfg, remote = ctx()
    for h in remote.api_hosts():
        if remote.agent_status(h).get("status") == "running":
            break
    else:
        die("no running agent found — `job agent install`")
    remote.sync_users(h)
    token = remote.run(h, f"cat {shlex.quote(cfg.identity.state_root)}/token").stdout.strip()
    lport = remote.tunnel(h, cfg.identity.port, "ui")
    url = f"http://127.0.0.1:{lport}/#token={token}"
    con.print(f"dashboard (via {h}): {url}")
    webbrowser.open(url)


@app.command()
def pull(jid: str = typer.Argument("last"), path: str = typer.Argument(""),
         dest: Path = typer.Option(None, "--dest")):
    """Copy a job's /insjob-out outputs to the laptop."""
    cfg, remote = ctx()
    i = resolve_id(remote, jid)
    j = remote.agent_op("get", str(i))
    dest = dest or Path("insjob-out") / str(i)
    dest.mkdir(parents=True, exist_ok=True)
    src = f"{remote.target(j.get('host') or cfg.host)}:{j['run_dir']}/outputs/{path}"
    sys.exit(subprocess.call(["rsync", "-ah", "--info=progress2", src.rstrip("/") + "/",
                              str(dest)]))


@data_app.command("ls")
def data_ls(path: str = typer.Argument(None)):
    """List a data directory on the servers (default: /datahome/<you>)."""
    cfg, remote = ctx()
    path = path or f"/datahome/{cfg.identity.user}"
    subprocess.call([*remote.ssh_args(cfg.host), f"ls -lah {shlex.quote(path)}"])


@data_app.command("pull")
def data_pull(path: str, dest: Path = typer.Argument(Path(".")),
              force: bool = typer.Option(False, help="allow more than 2 GB")):
    """Copy a (small!) piece of server data to the laptop."""
    cfg, remote = ctx()
    p = remote.run(cfg.host, f"du -sb {shlex.quote(path)} | cut -f1")
    size = int(p.stdout.strip() or 0)
    if size > 2 * 1024**3 and not force:
        die(f"{path} is {size / 1024**3:.1f} GB — keep data on the server "
            "(use `job mount`, `job exec`), or pass --force")
    sys.exit(subprocess.call(["rsync", "-ah", "--info=progress2",
                              f"{remote.target(cfg.host)}:{path}", str(dest)]))


@app.command()
def mount(
    what: str = typer.Argument(None, help="name from .insjob.toml [data], or a server path"),
    rw: bool = typer.Option(False, "--rw", help="writable (default read-only)"),
    status: bool = typer.Option(False, "--status"),
):
    """Mount project data from the servers on this laptop (sshfs) and tell Claude Code."""
    cfg, remote = ctx()
    if status:
        ms = mounted()
        if not ms:
            con.print("no insjob mounts")
        for mp, src in ms.items():
            stale = is_stale(Path(mp))
            if stale:
                unmount(Path(mp))
            con.print(f"{mp}  ←  {src}" + ("  [red]stale → unmounted, run `job mount` "
                                           "again[/]" if stale else ""))
        return
    try:
        repo = RepoConfig.load()
    except RuntimeError:
        repo = None
    targets: dict[str, str] = {}
    if what and what.startswith("/"):
        targets[Path(what).name] = what
    elif repo and repo.data:
        targets = {k: cfg.identity.expand(v) for k, v in repo.data.items()
                   if not what or k == what}
    if not targets:
        die("nothing to mount — give a server path, or add a [data] table to .insjob.toml")
    done = {}
    for name, path in targets.items():
        try:
            mp = mount_one(remote, name, path, rw=rw)
        except RuntimeError as e:
            die(str(e))
        con.print(f"[green]mounted[/] {path} → {mp}{'' if rw else ' (read-only)'}")
        done[name] = (mp, path)
    if repo:
        register_with_claude(repo, done)
        con.print("[dim]added to .claude/settings.local.json and CLAUDE.local.md[/]")


@app.command()
def umount(name: str = typer.Argument(None), all_: bool = typer.Option(False, "--all")):
    """Unmount data mounted by `job mount`."""
    ms = mounted()
    for mp in ms:
        if all_ or not name or Path(mp).name == name:
            ok = unmount(Path(mp))
            con.print(f"{'unmounted' if ok else '[red]could not unmount[/]'} {mp}")


TASKS = {
    "version": "2.0.0",
    "tasks": [
        {"label": "job: run", "type": "shell", "command": "job run --wait -- ${input:jobCmd}",
         "problemMatcher": []},
        {"label": "job: gpus", "type": "shell", "command": "job gpus", "problemMatcher": []},
        {"label": "job: ls", "type": "shell", "command": "job ls", "problemMatcher": []},
        {"label": "job: logs (last)", "type": "shell", "command": "job logs last -f",
         "problemMatcher": []},
        {"label": "job: dashboard", "type": "shell", "command": "job ui", "problemMatcher": []},
        {"label": "job: shell (server)", "type": "shell", "command": "job shell",
         "problemMatcher": []},
        {"label": "job: notebook", "type": "shell", "command": "job notebook",
         "problemMatcher": []},
        {"label": "job: mount data", "type": "shell", "command": "job mount",
         "problemMatcher": []},
    ],
    "inputs": [{"id": "jobCmd", "type": "promptString",
                "description": "command to run on the server", "default": "python src/train.py"}],
}


@app.command()
def init(force: bool = typer.Option(False, help="overwrite an existing .insjob.toml")):
    """Add .insjob.toml and VS Code tasks to the current repo."""
    from .config import REPO_FILE, repo_root

    root = repo_root()
    f = root / REPO_FILE
    dc = next((str(p.relative_to(root)) for p in
               [root / ".devcontainer/devcontainer.json", root / ".devcontainer.json"]
               if p.exists()), ".devcontainer/devcontainer.json")
    if f.exists() and not force:
        con.print(f"{REPO_FILE} exists (use --force to overwrite)")
    else:
        f.write_text(repo_template(root.name.lower(), (root / "uv.lock").exists(), dc))
        con.print(f"[green]wrote[/] {f}")
    if not (root / dc).exists():
        err.print(f"[yellow]note:[/] {dc} missing — jobs need a devcontainer "
                  "(template: github.com/raphaelschwinger/cuda-python-devcontainer)")
    tasks = root / ".vscode" / "tasks.json"
    if tasks.exists():
        try:
            cur = dcmod.json5.loads(tasks.read_text())
        except ValueError:
            con.print("[yellow].vscode/tasks.json not parseable — add the tasks by hand:[/]")
            print(json.dumps(TASKS, indent=2))
            return
        labels = {t.get("label") for t in cur.get("tasks", [])}
        cur.setdefault("tasks", []).extend(t for t in TASKS["tasks"] if t["label"] not in labels)
        ids = {i.get("id") for i in cur.get("inputs", [])}
        cur.setdefault("inputs", []).extend(i for i in TASKS["inputs"] if i["id"] not in ids)
        tasks.write_text(json.dumps(cur, indent=2) + "\n")
    else:
        tasks.parent.mkdir(exist_ok=True)
        tasks.write_text(json.dumps(TASKS, indent=2) + "\n")
    con.print(f"[green]VS Code tasks[/] in {tasks} (Terminal → Run Task → job: …)")


# ---- agent management -------------------------------------------------------------
def _hosts(cfg: LaptopConfig, host: str | None) -> list[str]:
    if host and host != "all":
        if host not in cfg.policy["hosts"]:
            die(f"{host} is not in your policy ({', '.join(cfg.policy['hosts'])})")
        return [host]
    return cfg.policy["hosts"]


@agent_app.command("install")
def agent_install(host: str = typer.Option("all", "-H", "--host")):
    """Build and (re)start your agent on the server(s)."""
    cfg, remote = ctx()
    for h in _hosts(cfg, host):
        con.print(f"[bold]{h}[/]")
        try:
            remote.install_agent(h, log=con.print)
            time.sleep(2)
            s = remote.agent_status(h)
            ping = s.get("ping") or {}
            con.print(f"  [green]{s['status']}[/] uid={ping.get('uid')} gpus={ping.get('gpus')} "
                      f"docker={ping.get('docker')}" if ping else f"  {s}")
        except RemoteError as e:
            err.print(f"  [red]{e}[/]")
    remote.sync_users()


@agent_app.command("upgrade")
def agent_upgrade(host: str = typer.Option("all", "-H", "--host")):
    """Same as install: rebuilds the agent from this version of insjob."""
    agent_install(host)


@agent_app.command("status")
def agent_status(host: str = typer.Option("all", "-H", "--host")):
    """Is your agent running on each server?"""
    cfg, remote = ctx()
    tag, _ = remote.agent_image()
    for h in _hosts(cfg, host):
        s = remote.agent_status(h)
        outdated = s.get("image") and s["image"] != tag
        con.print(f"{h:6} {s['status']}"
                  + ("  [yellow](outdated: `job agent upgrade`)[/]" if outdated else "")
                  + (f"  [red]{s['error'][:80]}[/]" if s.get("error") else ""))


@agent_app.command("up")
def agent_up(host: str = typer.Option("all", "-H", "--host")):
    """Start or install your agent wherever it isn't running (healthy ones are left alone)."""
    cfg, remote = ctx()
    tag, _ = remote.agent_image()
    for h in _hosts(cfg, host):
        s = remote.agent_status(h)
        st, current = s["status"], s.get("image") == tag
        try:
            if st == "unreachable":
                err.print(f"{h:6} [red]unreachable[/] (ssh failed)")
                continue
            if st == "running" and current and not s.get("error"):
                con.print(f"{h:6} [green]running[/]")
                continue
            if st in ("exited", "created", "paused") and current:
                con.print(f"{h:6} {st} → starting")
                remote.run(h, f"docker start {cfg.identity.agent_name} >/dev/null || "
                              f"docker unpause {cfg.identity.agent_name}")
            else:
                why = "outdated" if st == "running" and not current else (
                    "not answering" if st == "running" else st)
                con.print(f"{h:6} {why} → installing")
                remote.install_agent(h, log=con.print)
            time.sleep(2)
            s = remote.agent_status(h)
            ok = s["status"] == "running" and not s.get("error")
            con.print(f"{h:6} " + (f"[green]{s['status']}[/]" if ok else
                                   f"[red]{s['status']} {s.get('error', '')[:80]}[/]"))
        except RemoteError as e:
            err.print(f"{h:6} [red]{e}[/]")
    remote.sync_users()


@agent_app.command("logs")
def agent_logs(host: str = typer.Option(None, "-H", "--host"),
               follow: bool = typer.Option(False, "-f")):
    """Show the agent's own log on a server."""
    cfg, remote = ctx()
    h = host or cfg.host
    subprocess.call([*remote.ssh_args(h), f"docker logs --tail 200 {'-f' if follow else ''} "
                                          f"{cfg.identity.agent_name}"])


@agent_app.command("uninstall")
def agent_uninstall(host: str = typer.Option("all", "-H", "--host")):
    """Stop and remove your agent (running jobs keep running)."""
    cfg, remote = ctx()
    for h in _hosts(cfg, host):
        remote.run(h, f"docker rm -f {cfg.identity.agent_name}", check=False)
        con.print(f"{h}: removed")


@app.command("tunnels-close")
def tunnels_close():
    """Close background SSH tunnels (dashboard, notebooks)."""
    cfg, remote = ctx()
    con.print(f"closed {remote.close_tunnels()} tunnel(s)")


# ---- setup ----------------------------------------------------------------------------
SSH_BLOCK_START = "# >>> insjob >>>"
SSH_BLOCK_END = "# <<< insjob <<<"


def _ssh_ok(target: str) -> bool:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", target,
                           "true"], capture_output=True).returncode == 0


@app.command()
def setup(
    user: str = typer.Option(None, "-u", "--user", help="your login on the INS servers"),
    host: str = typer.Option(None, "-H", "--host", help="server to discover your account on"),
    yes: bool = typer.Option(False, "-y", "--yes", help="accept defaults, no questions"),
    install: bool = typer.Option(True, "--install/--no-install", help="install the agents"),
    skill: bool = typer.Option(True, "--skill/--no-skill", help="install the Claude Code skill"),
    ssh_config: bool = typer.Option(None, "--ssh-config/--no-ssh-config",
                                    help="add short host aliases to ~/.ssh/config"),
):
    """One-time setup: find your account, install your agents, check everything."""
    import getpass

    lab = DEFAULT_LAB
    try:
        old = LaptopConfig.load()
        user = user or old.identity.user
    except NotSetUp:
        old = None
    if not user:
        user = getpass.getuser() if yes else typer.prompt("Your login on the INS servers",
                                                          default=getpass.getuser())
    order = [host] if host else ["amur", "ploen", "stohl", "kiel", "stein"]
    found = None
    for h in order:
        t = f"{user}@{fqdn(lab, h)}"
        con.print(f"checking ssh {t} …")
        if _ssh_ok(t):
            found = h
            break
    if not found:
        h = order[0]
        t = f"{user}@{fqdn(lab, h)}"
        if yes or not typer.confirm(f"No key-based login yet. Run ssh-copy-id {t} now?",
                                    default=True):
            die("cannot log in with an SSH key — run `ssh-copy-id " + t + "` first")
        subprocess.call(["ssh-copy-id", t])
        if not _ssh_ok(t):
            die("still cannot log in with a key")
        found = h
    t = f"{user}@{fqdn(lab, found)}"
    out = subprocess.run(["ssh", t, "bash -s"], input=DISCOVER_SCRIPT, capture_output=True,
                         text=True, check=True).stdout
    ident = parse_discover(out)
    shared = subprocess.run(["ssh", t, "cat /datahome/insjob/lab.toml 2>/dev/null"],
                            capture_output=True, text=True).stdout
    if shared.strip():
        try:
            lab = deep_merge(lab, tomllib.loads(shared))
            con.print("[dim]using the lab policy from /datahome/insjob/lab.toml[/]")
        except tomllib.TOMLDecodeError as e:
            err.print(f"[yellow]ignoring broken /datahome/insjob/lab.toml: {e}[/]")
    pol = user_policy(lab, ident.groups)
    con.print(f"\nyou are [bold]{ident.user}[/] (uid {ident.uid}, gid {ident.gid}, groups "
              f"{','.join(ident.groups)}) → policy [bold]{pol['group']}[/]: "
              f"hosts {', '.join(pol['hosts'])}, default {pol['default_host']}, "
              f"≤{pol['max_gpus_per_host']} GPUs/host")
    con.print(f"state: {ident.state_root}   agent port: {ident.port}")
    if "docker" not in ident.groups:
        die("you are not in the `docker` group on the servers — ask the admin "
            "(torge.storm@informatik.uni-kiel.de) to add you, then re-run `job setup`")
    cfg = LaptopConfig(identity=ident, lab=lab, default_host=pol["default_host"],
                       narrow=old.narrow if old else {})
    cfg.save()
    con.print(f"[green]saved[/] {CONFIG_PATH}")
    remote = Remote(cfg)
    remote.run(found, f"mkdir -p {shlex.quote(ident.state_root)} && chmod 700 "
                      f"{shlex.quote(ident.state_root)}")

    has_secrets = remote.run(found, f"test -s {shlex.quote(ident.state_root)}/secrets.env "
                                    "&& echo y", check=False).stdout.strip() == "y"
    if not has_secrets and not yes:
        key = typer.prompt("W&B API key for your jobs (empty to skip)", default="",
                           hide_input=True, show_default=False)
        if key.strip():
            remote.run(found, f"umask 077; cat >> {shlex.quote(ident.state_root)}/secrets.env",
                       input=f"WANDB_API_KEY={key.strip()}\n")
            con.print("[green]stored[/] in secrets.env (only readable by you)")

    if ssh_config is None:
        ssh_config = (not yes) and typer.confirm(
            f"Add short aliases ({', '.join(pol['hosts'])}) to ~/.ssh/config?", default=False)
    if ssh_config:
        _write_ssh_block(cfg)

    if install:
        con.print("\n[bold]installing agents[/]")
        agent_install("all")
    if skill:
        _install_skill()
    con.print("\n[bold green]done.[/] Try: `job gpus`, then in a repo `job init` and "
              "`job run -- python -c 'import torch; print(torch.cuda.get_device_name())'`")


def _write_ssh_block(cfg: LaptopConfig) -> None:
    f = Path("~/.ssh/config").expanduser()
    text = f.read_text() if f.exists() else ""
    if SSH_BLOCK_START in text:
        pre, rest = text.split(SSH_BLOCK_START, 1)
        text = pre + rest.split(SSH_BLOCK_END, 1)[-1].lstrip("\n")
    lines = [SSH_BLOCK_START]
    for h in cfg.policy["hosts"]:
        lines += [f"Host {h}", f"  HostName {fqdn(cfg.lab, h)}", f"  User {cfg.identity.user}",
                  "  ServerAliveInterval 15", ""]
    lines.append(SSH_BLOCK_END)
    if f.exists():
        shutil.copy(f, f.with_suffix(".insjob-backup"))
    f.parent.mkdir(mode=0o700, exist_ok=True)
    f.write_text(text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n")
    con.print(f"[green]updated[/] {f} (backup: {f.with_suffix('.insjob-backup')})")


def _install_skill() -> None:
    here = Path(__file__).parent
    src = next((p for p in (here / "_skills" / "insjob", here.parent / "skills" / "insjob")
                if (p / "SKILL.md").exists()), None)
    if not src:
        err.print("[yellow]skill files not found in this install[/]")
        return
    dst = Path("~/.claude/skills/insjob").expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "SKILL.md", dst / "SKILL.md")
    con.print(f"[green]Claude Code skill[/] → {dst}")


@app.command()
def version():
    """Print the insjob version."""
    con.print(__version__)


def main() -> None:
    try:
        app()
    except RemoteError as e:
        err.print(f"[red]error:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

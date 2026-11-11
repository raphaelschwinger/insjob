"""Operations the laptop CLI runs inside the agent container:

    ssh <host> docker exec -i insjob-agent-<user> insjob-remote <op> [args] < request.json

SSH is the authentication; the JSON answer goes to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from ..identity import Identity
from ..locks import GpuLocks, LabStatus
from ..policy import load_lab, user_policy
from ..state import State
from . import docker_ops, gpu, notify
from .views import overview


def _ctx():
    ident = Identity.from_env()
    lab = load_lab()
    cfg = notify.load_user_config(ident.state_root)
    policy = user_policy(lab, ident.groups, cfg.get("policy"))
    state = State(ident.state_root)
    state.ensure()
    return ident, os.environ["INSJOB_HOST"], lab, policy, state


def _read_stdin() -> dict:
    data = sys.stdin.read()
    return json.loads(data) if data.strip() else {}


def _out(obj) -> None:
    json.dump(obj, sys.stdout)
    sys.stdout.write("\n")


def op_ping(args) -> None:
    ident, host, lab, policy, state = _ctx()
    try:
        n = len(gpu.read_gpus())
    except Exception as e:  # noqa: BLE001
        n = f"error: {e}"
    dv = docker_ops.docker("version", "--format", "{{.Server.Version}}", check=False)
    _out({"host": host, "user": ident.user, "uid": ident.uid, "gid": ident.gid,
          "gpus": n, "docker": dv.stdout.strip() or dv.stderr.strip(), "policy": policy,
          "state_root": ident.state_root})


def op_new_id(args) -> None:
    *_, state = _ctx()
    jid = state.new_job_id()
    rd = state.run_dir(jid)
    _out({"id": jid, "run_dir": str(rd), "code_dir": str(rd / "code")})


def op_submit(args) -> None:
    *_, state = _ctx()
    job = _read_stdin()
    state.submit(job)
    _out({"id": job["id"], "state": "queued"})


def op_prepare(args) -> None:
    *_, state = _ctx()
    job = _read_stdin()
    (state.run_dir(job["id"]) / "job.json").write_text(json.dumps(job, indent=1))
    _out({"id": job["id"]})


def op_get(args) -> None:
    *_, state = _ctx()
    _out(state.get(args.id))


def op_list(args) -> None:
    *_, state = _ctx()
    _out(state.list_jobs(args.limit))


def op_cancel(args) -> None:
    *_, state = _ctx()
    _out({"id": args.id, "result": state.request_cancel(args.id)})


def op_overview(args) -> None:
    ident, host, lab, policy, state = _ctx()
    d = overview(state, LabStatus(lab["paths"]["lab_dir"]), ident, policy, lab, args.limit)
    d["served_by"] = host
    _out(d)


def op_add_users(args) -> None:
    ident, _, lab, *_ = _ctx()
    LabStatus(lab["paths"]["lab_dir"]).add_users(_read_stdin(), ident.user)
    _out({"ok": True})


def op_events(args) -> None:
    *_, state = _ctx()
    _out(state.events(args.since))


def op_interactive(args) -> None:
    """exec / shell: run in the foreground, attached to the caller's terminal."""
    ident, host, lab, policy, state = _ctx()
    rd = state.run_dir(args.id)
    job = json.loads((rd / "job.json").read_text())
    locks = GpuLocks(lab["paths"]["lock_dir"], ident.user)
    n = int(job.get("gpus", 0))
    ids: list[int] = []
    if host not in policy["hosts"]:
        sys.exit(f"insjob: your policy ({policy['group']}) does not allow {host}")
    if n:
        gpus = gpu.read_gpus()
        th = lab.get("gpu_free", {})
        free = gpu.free_gpu_ids(gpus, set(locks.claims()), th.get("max_mem_used_mb", 1024),
                                th.get("max_util", 5))
        picked = gpu.pick_gpus(gpus, free, n, float(job.get("min_gpu_mem_gb") or 0))
        if picked is None:
            sys.exit(f"insjob: no {n} free GPU(s) on {host} right now — see `job gpus`")
        ids = picked
    job.update(state="running", host=host, gpu_ids=ids, started_at=time.time())
    (state.claimed / host).mkdir(parents=True, exist_ok=True)
    state.update_claimed(host, job)
    if ids and not locks.acquire(ids, job["id"], {"repo": job.get("repo")}):
        state.finish(job, host, "failed", reason="GPU taken concurrently")
        sys.exit("insjob: GPU was just taken by someone else, try again")
    code = 1
    try:
        image = docker_ops.ensure_image(job, ident, sys.stderr)
        script = docker_ops.job_script(job, ident, tee=job.get("kind") == "exec")
        if job.get("kind") == "shell":
            script = docker_ops.job_script({**job, "cmd": ["true"], "setup": ""}, ident,
                                           tee=False) + "\nexec bash -l"
        rargs = docker_ops.run_args(job, ident, image=image, detach=False, interactive=True,
                                    tty=args.tty, script=script)
        docker_ops.docker("rm", "-f", f"job-{ident.user}-{job['id']}", check=False)
        code = subprocess.call([docker_ops.DOCKER, *rargs])
    finally:
        locks.release(job["id"])
        state.finish(job, host, "succeeded" if code == 0 else "failed", exit_code=code)
    sys.exit(code)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="insjob-remote")
    sub = p.add_subparsers(dest="op", required=True)
    sub.add_parser("ping").set_defaults(fn=op_ping)
    sub.add_parser("new_id").set_defaults(fn=op_new_id)
    sub.add_parser("submit").set_defaults(fn=op_submit)
    sub.add_parser("prepare").set_defaults(fn=op_prepare)
    s = sub.add_parser("get")
    s.add_argument("id", type=int)
    s.set_defaults(fn=op_get)
    s = sub.add_parser("list")
    s.add_argument("--limit", type=int, default=200)
    s.set_defaults(fn=op_list)
    s = sub.add_parser("cancel")
    s.add_argument("id", type=int)
    s.set_defaults(fn=op_cancel)
    s = sub.add_parser("overview")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(fn=op_overview)
    sub.add_parser("add_users").set_defaults(fn=op_add_users)
    s = sub.add_parser("events")
    s.add_argument("--since", type=float, default=0)
    s.set_defaults(fn=op_events)
    s = sub.add_parser("interactive")
    s.add_argument("id", type=int)
    s.add_argument("--tty", action="store_true")
    s.set_defaults(fn=op_interactive)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

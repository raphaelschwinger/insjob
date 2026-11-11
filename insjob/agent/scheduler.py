"""The per-user, per-host agent loop: claim → build → run → reap, plus status publishing."""

from __future__ import annotations

import re
import threading
import time
import traceback
from pathlib import Path

from ..identity import Identity
from ..locks import GpuLocks, LabStatus
from ..policy import reservation, size_gb, user_policy
from ..state import State
from . import docker_ops, gpu, notify

WANDB_RE = re.compile(r"https://wandb\.ai/\S+/runs/[A-Za-z0-9]+")
INTERACTIVE = {"exec", "shell"}


def tail(path: Path, n: int = 20) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            return f.read().decode(errors="replace").splitlines()[-n:]
    except FileNotFoundError:
        return []


def find_wandb_url(path: Path) -> str | None:
    urls = WANDB_RE.findall("\n".join(tail(path, 400)))
    return urls[-1] if urls else None


def public_job(j: dict) -> dict:
    return {
        "id": j["id"],
        "repo": j.get("repo"),
        "kind": j.get("kind", "run"),
        "state": j.get("state"),
        "gpu_ids": j.get("gpu_ids") or [],
        "started_at": j.get("started_at"),
        "cmd": docker_ops.command_string(j.get("cmd") or [])[:80],
    }


class Agent:
    def __init__(self, ident: Identity, host: str, lab: dict, *, state: State | None = None,
                 locks: GpuLocks | None = None, lab_status: LabStatus | None = None,
                 docker=docker_ops, read_gpus=gpu.read_gpus, read_host=gpu.read_host,
                 interval: float = 10):
        self.ident = ident
        self.host = host
        self.lab = lab
        self.state = state or State(ident.state_root)
        self.state.ensure()
        self.user_cfg = notify.load_user_config(ident.state_root)
        self.policy = user_policy(lab, ident.groups, self.user_cfg.get("policy"))
        self.locks = locks or GpuLocks(lab["paths"]["lock_dir"], ident.user)
        self.lab_status = lab_status or LabStatus(lab["paths"]["lab_dir"])
        self.docker = docker
        self.read_gpus = read_gpus
        self.read_host = read_host
        self.interval = interval
        self.launching: dict[int, threading.Thread] = {}
        self.last_gpus: list[gpu.GpuInfo] = []
        self.last_host: dict = {}
        self.write_lock = threading.Lock()
        self.lab_status.add_users({str(ident.uid): ident.user}, ident.user)

    def log(self, msg: str) -> None:
        print(f"[insjob-agent {self.host}] {msg}", flush=True)

    # ---- main loop ---------------------------------------------------------------
    def run_forever(self) -> None:
        self.log(f"started for {self.ident.user} (uid {self.ident.uid}), policy {self.policy}")
        while True:
            t0 = time.time()
            try:
                self.tick()
            except Exception:
                self.log("tick failed:\n" + traceback.format_exc())
            time.sleep(max(1.0, self.interval - (time.time() - t0)))

    def tick(self) -> None:
        try:
            gpus = self.read_gpus()
        except Exception as e:
            self.log(f"cannot read GPUs: {e}")
            gpus = []
        host_info = self.read_host()
        self.last_gpus, self.last_host = gpus, host_info
        containers = self.docker.list_job_containers(self.ident.user)
        self.reconcile(containers)
        live = {j["id"] for j in self.state.claimed_on(self.host)}
        self.locks.heartbeat(live)
        self.schedule(gpus, host_info)
        self.publish(gpus, host_info)

    # ---- reconcile running jobs ----------------------------------------------------
    def reconcile(self, containers: dict[str, dict]) -> None:
        for job in self.state.claimed_on(self.host):
            jid = job["id"]
            t = self.launching.get(jid)
            if t and t.is_alive():
                continue
            self.launching.pop(jid, None)
            name = f"job-{self.ident.user}-{jid}"
            c = containers.get(name)

            if job.get("kind") in INTERACTIVE:
                # the interactive op records the result itself; only clean up leftovers
                if c is None and time.time() - job.get("started_at", 0) > 60:
                    self.locks.release(jid)
                    self.state.finish(job, self.host, "succeeded", reason="interactive session ended")
                continue

            if job.get("state") == "starting":
                if c is None:  # agent restarted in the middle of a launch: try again
                    self.locks.release(jid)
                    self.state.unclaim(jid, self.host)
                    self.state.emit("requeued", job, "agent restarted during launch")
                    continue
                job.update(state="running", container=name,
                           started_at=job.get("started_at") or time.time())
                self.state.update_claimed(self.host, job)

            if c is None:
                self.locks.release(jid)
                self._finish(job, "failed", reason="container disappeared")
                continue
            if job.get("kind") == "notebook" and not (job.get("notebook") or {}).get("port"):
                port = docker_ops.host_port(c)
                if port:
                    job["notebook"] = {**(job.get("notebook") or {}), "port": port}
                    self.state.update_claimed(self.host, job)
            if c["status"] in ("exited", "dead"):
                self.reap(job, c)
            elif self.state.cancel_requested(jid):
                self.log(f"stopping job {jid}")
                self.docker.stop(name)
            elif job.get("gpu_ids"):
                self.locks.acquire(job["gpu_ids"], jid, {"repo": job.get("repo")})

    def reap(self, job: dict, c: dict) -> None:
        jid = job["id"]
        code = c.get("exit_code")
        if self.state.cancel_requested(jid):
            st = "cancelled"
        else:
            st = "succeeded" if code == 0 else "failed"
        log = self.state.run_dir(jid) / "log"
        self.docker.remove(f"job-{self.ident.user}-{jid}")
        self.locks.release(jid)
        self._finish(job, st, exit_code=code, wandb_url=find_wandb_url(log),
                     tail=tail(log, 20) if st == "failed" else [])

    def _finish(self, job: dict, st: str, **fields) -> None:
        with self.write_lock:
            job = self.state.finish(job, self.host, st, **fields)
        self.log(f"job {job['id']} {st}")
        notify.send(self.user_cfg, job)

    # ---- scheduling ----------------------------------------------------------------
    def host_ok(self, job: dict) -> bool:
        hosts = job.get("hosts") or ["any"]
        return self.host in self.policy["hosts"] and ("any" in hosts or self.host in hosts)

    def cpu_ok(self, job: dict, host_info: dict) -> bool:
        need_cpu = min(float(job.get("cpus", 1)), 4.0)
        need_mem = size_gb(reservation(job, "8g")) * 1024
        return (host_info.get("cpus_free", 0) >= need_cpu
                and host_info.get("mem_avail_mb", 0) >= need_mem)

    def best_cpu_host(self, job: dict, host_info: dict) -> bool:
        """For unpinned CPU jobs, leave the job to a clearly less loaded host."""
        hosts = job.get("hosts") or ["any"]
        if "any" not in hosts and len(hosts) == 1:
            return True
        mine = host_info.get("cpus_free", 0)
        for h, info in self.state.read_hosts(max_age=60).items():
            if h == self.host or h not in self.policy["hosts"]:
                continue
            if "any" not in hosts and h not in hosts:
                continue
            if info.get("cpus_free", 0) > mine + 4:
                return False
        return True

    def schedule(self, gpus: list[gpu.GpuInfo], host_info: dict) -> None:
        th = self.lab.get("gpu_free", {})
        claims = self.locks.claims()
        free = gpu.free_gpu_ids(gpus, set(claims), th.get("max_mem_used_mb", 1024),
                                th.get("max_util", 5))
        mine_here = sum(len(j.get("gpu_ids") or []) for j in self.state.claimed_on(self.host))
        for job in self.state.queued():
            if not self.host_ok(job):
                continue
            dep = self.state.dep_status(job)
            if dep == "skip":
                self.state.cancel_queued(job["id"], "skipped", "a dependency did not succeed")
                continue
            if dep == "wait":
                continue
            n = int(job.get("gpus", 0))
            ids: list[int] = []
            if n > 0:
                if mine_here + n > self.policy["max_gpus_per_host"]:
                    continue
                req = job.get("gpu_ids_req") or []
                pool = [i for i in free if i in req] if req else free
                picked = gpu.pick_gpus(gpus, pool, n, float(job.get("min_gpu_mem_gb") or 0))
                if picked is None:
                    continue
                ids = picked
            elif not (self.cpu_ok(job, host_info) and self.best_cpu_host(job, host_info)):
                continue
            if not self.state.claim(job["id"], self.host):
                continue  # another of my agents was faster, or it was cancelled
            if ids and not self.locks.acquire(ids, job["id"], {"repo": job.get("repo")}):
                self.state.unclaim(job["id"], self.host)
                continue
            job.update(state="starting", host=self.host, gpu_ids=ids, claimed_at=time.time())
            self.state.update_claimed(self.host, job)
            self.state.emit("starting", job)
            self.log(f"claimed job {job['id']} gpus={ids}")
            free = [i for i in free if i not in ids]
            mine_here += n
            t = threading.Thread(target=self._launch, args=(job,), daemon=True)
            self.launching[job["id"]] = t
            t.start()

    def _launch(self, job: dict) -> None:
        jid = job["id"]
        logp = self.state.run_dir(jid) / "log"
        with open(logp, "a") as log:
            try:
                image = self.docker.ensure_image(job, self.ident, log)
                if self.state.cancel_requested(jid):
                    raise RuntimeError("cancelled before start")
                name = self.docker.run_job(job, self.ident, image, log)
                job.update(state="running", container=name, image=image, started_at=time.time())
                with self.write_lock:
                    self.state.update_claimed(self.host, job)
                self.state.emit("running", job)
            except Exception as e:
                log.write(f"[insjob] launch failed: {e}\n")
                self.locks.release(jid)
                st = "cancelled" if self.state.cancel_requested(jid) else "failed"
                self._finish(job, st, reason=str(e)[:500], tail=tail(logp, 20))

    # ---- publishing ----------------------------------------------------------------
    def publish(self, gpus: list[gpu.GpuInfo], host_info: dict) -> None:
        mine = self.state.claimed_on(self.host)
        self.lab_status.publish(self.host, self.ident.user, self.ident.uid,
                                [public_job(j) for j in mine])
        try:
            conts = self.docker.all_containers()
        except Exception:
            conts = {}
        gd = []
        for g in gpus:
            d = g.to_dict()
            for p in d["procs"]:
                c = conts.get(p.get("container") or "")
                if c:
                    p.update(container_name=c["name"], insjob_user=c["insjob_user"],
                             insjob_id=c["insjob_id"])
            gd.append(d)
        claims = {str(k): v for k, v in self.locks.claims().items()}
        info = {"by": self.ident.user, "gpus": gd, "claims": claims, **host_info}
        self.lab_status.publish_gpus(self.host, info)
        self.state.write_host(self.host, info)

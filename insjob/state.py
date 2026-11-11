"""Per-user job state as plain files on NFS.

Layout under the user's state root (e.g. /datahome/<user>/.insjob):

    queue/<id>.json            submitted, waiting for a GPU / CPU / dependency
    claimed/<host>/<id>.json   claimed by the agent on <host> (starting or running)
    done/<id>.json             finished: succeeded | failed | cancelled | skipped
    runs/<id>/                 code/, log, outputs/, job.json, cancel (flag)
    hosts/<host>.json          heartbeat + resources of each of the user's agents
    events.jsonl               notifications feed
    token, secrets.env, config.toml

Moving a record between directories is a rename(), which is atomic on NFSv4 — that is
what makes "only one host claims a job" and "cancel vs. claim" race-free.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

TERMINAL = {"succeeded", "failed", "cancelled", "skipped"}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class State:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.queue = self.root / "queue"
        self.claimed = self.root / "claimed"
        self.done = self.root / "done"
        self.runs = self.root / "runs"
        self.hosts = self.root / "hosts"
        self.events_file = self.root / "events.jsonl"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        for d in (self.queue, self.claimed, self.done, self.runs, self.hosts):
            d.mkdir(exist_ok=True)

    # ---- ids & runs ---------------------------------------------------------------
    def new_job_id(self) -> int:
        """Allocate the next id by mkdir (atomic on NFS) — no lock needed."""
        existing = [int(p.name) for p in self.runs.iterdir() if p.name.isdigit()]
        n = max(existing, default=0) + 1
        while True:
            try:
                (self.runs / str(n)).mkdir()
                (self.runs / str(n) / "outputs").mkdir()
                return n
            except FileExistsError:
                n += 1

    def run_dir(self, job_id: int) -> Path:
        return self.runs / str(job_id)

    # ---- records ------------------------------------------------------------------
    def submit(self, job: dict) -> None:
        job = {**job, "state": "queued", "submitted_at": job.get("submitted_at", time.time())}
        _write_json(self.run_dir(job["id"]) / "job.json", job)
        _write_json(self.queue / f"{job['id']}.json", job)
        self.emit("queued", job)

    def queued(self) -> list[dict]:
        jobs = [_read_json(p) for p in self.queue.glob("*.json")]
        return sorted((j for j in jobs if j), key=lambda j: j["id"])

    def claimed_on(self, host: str) -> list[dict]:
        d = self.claimed / host
        if not d.exists():
            return []
        jobs = [_read_json(p) for p in d.glob("*.json")]
        return sorted((j for j in jobs if j), key=lambda j: j["id"])

    def all_claimed(self) -> list[dict]:
        out: list[dict] = []
        if self.claimed.exists():
            for d in self.claimed.iterdir():
                if d.is_dir():
                    out += self.claimed_on(d.name)
        return out

    def finished(self, limit: int | None = None) -> list[dict]:
        paths = sorted(self.done.glob("*.json"), key=lambda p: int(p.stem), reverse=True)
        if limit:
            paths = paths[:limit]
        return [j for j in (_read_json(p) for p in paths) if j]

    def get(self, job_id: int) -> dict | None:
        for p in [self.queue / f"{job_id}.json", self.done / f"{job_id}.json"]:
            j = _read_json(p)
            if j:
                return j
        if self.claimed.exists():
            for d in self.claimed.iterdir():
                j = _read_json(d / f"{job_id}.json")
                if j:
                    return j
        return _read_json(self.run_dir(job_id) / "job.json")

    def list_jobs(self, limit: int = 200) -> list[dict]:
        jobs = self.queued() + self.all_claimed() + self.finished(limit)
        return sorted(jobs, key=lambda j: j["id"], reverse=True)[:limit]

    def update_claimed(self, host: str, job: dict) -> None:
        _write_json(self.claimed / host / f"{job['id']}.json", job)
        _write_json(self.run_dir(job["id"]) / "job.json", job)

    # ---- transitions ----------------------------------------------------------------
    def claim(self, job_id: int, host: str) -> bool:
        (self.claimed / host).mkdir(parents=True, exist_ok=True)
        try:
            os.rename(self.queue / f"{job_id}.json", self.claimed / host / f"{job_id}.json")
            return True
        except FileNotFoundError:
            return False

    def unclaim(self, job_id: int, host: str) -> None:
        src = self.claimed / host / f"{job_id}.json"
        j = _read_json(src)
        if j:
            j.update(state="queued", host=None, gpu_ids=[])
            _write_json(src, j)
            os.rename(src, self.queue / f"{job_id}.json")

    def finish(self, job: dict, host: str | None, state: str, **fields) -> dict:
        job = {**job, **fields, "state": state, "ended_at": fields.get("ended_at", time.time())}
        _write_json(self.done / f"{job['id']}.json", job)
        _write_json(self.run_dir(job["id"]) / "job.json", job)
        if host:
            try:
                (self.claimed / host / f"{job['id']}.json").unlink()
            except FileNotFoundError:
                pass
        self.emit(state, job)
        return job

    def cancel_queued(self, job_id: int, state: str = "cancelled", reason: str = "") -> bool:
        """Move a queued job to done/. Loses cleanly against a concurrent claim()."""
        dst = self.done / f"{job_id}.json"
        try:
            os.rename(self.queue / f"{job_id}.json", dst)
        except FileNotFoundError:
            return False
        j = _read_json(dst) or {"id": job_id}
        self.finish(j, None, state, reason=reason)
        return True

    def request_cancel(self, job_id: int) -> str:
        if self.cancel_queued(job_id):
            return "cancelled"
        j = self.get(job_id)
        if not j:
            return "unknown"
        if j.get("state") in TERMINAL:
            return j["state"]
        (self.run_dir(job_id) / "cancel").touch()
        return "stopping"

    def cancel_requested(self, job_id: int) -> bool:
        return (self.run_dir(job_id) / "cancel").exists()

    def dep_status(self, job: dict) -> str:
        """'ok' when all `after` jobs succeeded, 'wait' while pending, 'skip' if one failed."""
        for dep in job.get("after") or []:
            d = self.get(int(dep))
            if d is None:
                return "skip"
            s = d.get("state")
            if s == "succeeded":
                continue
            if s in TERMINAL:
                return "skip"
            return "wait"
        return "ok"

    # ---- events -------------------------------------------------------------------
    def emit(self, kind: str, job: dict, msg: str = "") -> dict:
        ev = {
            "ts": time.time(),
            "kind": kind,
            "job": job.get("id"),
            "repo": job.get("repo"),
            "host": job.get("host"),
            "msg": msg or job.get("reason", ""),
        }
        with open(self.events_file, "a") as f:
            f.write(json.dumps(ev) + "\n")
        return ev

    def events(self, since: float = 0, limit: int = 100) -> list[dict]:
        try:
            lines = self.events_file.read_text().splitlines()[-1000:]
        except FileNotFoundError:
            return []
        evs = []
        for line in lines:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e["ts"] > since:
                evs.append(e)
        return evs[-limit:]

    # ---- host heartbeats ---------------------------------------------------------------
    def write_host(self, host: str, info: dict) -> None:
        _write_json(self.hosts / f"{host}.json", {**info, "host": host, "updated": time.time()})

    def read_hosts(self, max_age: float = 120) -> dict[str, dict]:
        out = {}
        for p in self.hosts.glob("*.json"):
            j = _read_json(p)
            if j and time.time() - j.get("updated", 0) < max_age:
                out[p.stem] = j
        return out

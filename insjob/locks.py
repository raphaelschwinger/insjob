"""Cross-user GPU claims on one host, and the public lab-wide status.

Every user's agent on a host bind-mounts the same local directory (/var/tmp/insjob,
mode 1777). A GPU is claimed by writing `gpu<N>.<user>.json`; each user only ever
writes and deletes their own files (the sticky bit forbids touching others').
Claim decisions happen under an exclusive flock on `claim.lock`, so two users' agents
cannot grab the same free GPU at the same moment. A claim whose file has not been
touched for `stale_s` seconds (its agent is gone) is ignored.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

STALE_S = 180


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except PermissionError:
        pass


class GpuLocks:
    def __init__(self, lock_dir: str | Path, user: str, stale_s: float = STALE_S):
        self.dir = Path(lock_dir)
        self.user = user
        self.stale_s = stale_s

    def ensure(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _chmod(self.dir, 0o1777)
        lock = self.dir / "claim.lock"
        if not lock.exists():
            lock.touch()
            _chmod(lock, 0o666)

    @contextmanager
    def _locked(self):
        self.ensure()
        fd = os.open(self.dir / "claim.lock", os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def claims(self) -> dict[int, dict]:
        """GPU index -> live claim (any user)."""
        out: dict[int, dict] = {}
        now = time.time()
        for p in self.dir.glob("gpu*.json"):
            try:
                st = p.stat()
                if now - st.st_mtime > self.stale_s:
                    continue
                c = json.loads(p.read_text())
            except (FileNotFoundError, json.JSONDecodeError, PermissionError):
                continue
            out[int(c["gpu"])] = c
        return out

    def _file(self, gpu: int) -> Path:
        return self.dir / f"gpu{gpu}.{self.user}.json"

    def acquire(self, gpu_ids: list[int], job_id: int, extra: dict | None = None) -> bool:
        with self._locked():
            current = self.claims()
            for g in gpu_ids:
                c = current.get(g)
                if c and not (c["user"] == self.user and c["job"] == job_id):
                    return False
            for g in gpu_ids:
                data = {"gpu": g, "user": self.user, "job": job_id, "since": time.time(),
                        **(extra or {})}
                f = self._file(g)
                f.write_text(json.dumps(data))
                _chmod(f, 0o644)
            return True

    def release(self, job_id: int) -> None:
        for p in self.dir.glob(f"gpu*.{self.user}.json"):
            try:
                if json.loads(p.read_text()).get("job") == job_id:
                    p.unlink()
            except (FileNotFoundError, json.JSONDecodeError):
                pass

    def heartbeat(self, job_ids: set[int]) -> None:
        """Keep own claims fresh for running jobs; drop own claims of jobs not running."""
        for p in self.dir.glob(f"gpu*.{self.user}.json"):
            try:
                job = json.loads(p.read_text()).get("job")
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if job in job_ids:
                os.utime(p)
            else:
                p.unlink(missing_ok=True)


class LabStatus:
    """Public, non-secret status shared by everybody: <lab_dir>/status/<host>/<user>.json."""

    def __init__(self, lab_dir: str | Path):
        self.dir = Path(lab_dir)

    def _host_dir(self, host: str) -> Path | None:
        d = self.dir / "status" / host
        try:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                for p in (self.dir, self.dir / "status", d):
                    _chmod(p, 0o1777)
            return d
        except PermissionError:
            return None

    def publish(self, host: str, user: str, uid: int, jobs: list[dict]) -> None:
        d = self._host_dir(host)
        if not d:
            return
        data = {"host": host, "user": user, "uid": uid, "updated": time.time(), "jobs": jobs}
        self._write(d / f"{user}.json", data)

    def publish_gpus(self, host: str, info: dict) -> None:
        d = self._host_dir(host)
        if d:
            self._write(d / f"_gpus.{info.get('by', 'x')}.json",
                        {**info, "host": host, "updated": time.time()})

    @staticmethod
    def _write(path: Path, data: dict) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(data))
            _chmod(tmp, 0o644)
            os.replace(tmp, path)
        except PermissionError:
            pass

    def read(self, max_age: float = 180) -> dict:
        """{'hosts': {host: gpus-info}, 'users': [status...]}, freshest GPU info per host."""
        out: dict = {"hosts": {}, "users": []}
        now = time.time()
        root = self.dir / "status"
        if not root.exists():
            return out
        for hd in root.iterdir():
            if not hd.is_dir():
                continue
            for p in hd.glob("*.json"):
                try:
                    j = json.loads(p.read_text())
                except (json.JSONDecodeError, PermissionError, FileNotFoundError):
                    continue
                if now - j.get("updated", 0) > max_age:
                    continue
                if p.name.startswith("_gpus."):
                    cur = out["hosts"].get(hd.name)
                    if not cur or j["updated"] > cur["updated"]:
                        out["hosts"][hd.name] = j
                else:
                    out["users"].append(j)
        return out

    def users_map(self) -> dict[str, str]:
        """uid -> name, merged from every user's contribution."""
        out: dict[str, str] = {}
        d = self.dir / "users"
        if d.exists():
            for p in d.glob("*.json"):
                try:
                    out.update(json.loads(p.read_text()))
                except (json.JSONDecodeError, PermissionError, FileNotFoundError):
                    pass
        return out

    def add_users(self, mapping: dict[str, str], by_user: str) -> None:
        """Extend the uid -> name cache (each user writes only their own file)."""
        d = self.dir / "users"
        try:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                _chmod(self.dir, 0o1777)
                _chmod(d, 0o1777)
        except PermissionError:
            return
        own = d / f"{by_user}.json"
        try:
            cur = json.loads(own.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            cur = {}
        if all(cur.get(k) == v for k, v in mapping.items()):
            return
        cur.update(mapping)
        self._write(own, cur)

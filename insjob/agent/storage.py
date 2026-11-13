"""Storage readings: the shared network drives (/datahome, /home) and each server's disks.

The agent runs with --pid=host, so /proc/1/mountinfo is the host's mount table and
/sys/block lists the host's block devices. To measure a host filesystem it must also see
it: `job agent install` bind-mounts every local filesystem read-only (and non-recursively)
at HOSTFS/<target with / replaced by _>, e.g. /data1 -> /hostfs/_data1, / -> /hostfs/_.
Without that (older installs) usage is shown only for paths the container sees itself.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

HOSTFS = "/hostfs"
LOCAL_FS = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "f2fs", "jfs", "reiserfs", "vfat",
            "exfat", "ntfs", "ntfs3", "fuseblk"}
SKIP_PREFIXES = ("/boot", "/var/lib/docker", "/var/lib/containers", "/snap", "/run", "/proc",
                 "/sys", "/dev")
NET_FS = {"nfs", "nfs4", "cifs", "smb3", "ceph", "glusterfs", "lustre", "beegfs"}


def hostfs_path(target: str, root: str = HOSTFS) -> str:
    return f"{root}/{target.replace('/', '_')}"


def _unescape(s: str) -> str:
    """mountinfo escapes space, tab, newline and backslash as octal (\\040 …)."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and len(s[i + 1:i + 4]) == 3 and s[i + 1:i + 4].isdigit():
            out.append(chr(int(s[i + 1:i + 4], 8)))
            i += 4
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def parse_mountinfo(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        parts = line.split()
        if " - " not in line or len(parts) < 10:
            continue
        sep = parts.index("-")
        out.append({
            "dev": parts[2], "root": _unescape(parts[3]), "target": _unescape(parts[4]),
            "fstype": parts[sep + 1], "source": _unescape(parts[sep + 2]),
        })
    return out


def usage(path: str) -> dict | None:
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    if not st.f_blocks:
        return None  # autofs placeholder etc.
    return {"size_b": st.f_blocks * st.f_frsize,
            "used_b": (st.f_blocks - st.f_bfree) * st.f_frsize,
            "avail_b": st.f_bavail * st.f_frsize}


def _read(p: Path, default: str = "") -> str:
    try:
        return p.read_text().strip()
    except OSError:
        return default


class BlockDevices:
    """Host disks from sysfs; maps any (partition, LVM, md) device to its physical disks."""

    def __init__(self, sys: str = "/sys"):
        self.sys = Path(sys)

    def disks(self) -> list[dict]:
        out = []
        try:
            names = sorted(os.listdir(self.sys / "block"))
        except OSError:
            return out
        for name in names:
            d = self.sys / "block" / name
            if name.startswith(("loop", "ram", "zram", "sr", "fd")) or not (d / "device").exists():
                continue
            if _read(d / "hidden") == "1":  # NVMe multipath controller path (nvme0c0n1)
                continue
            size = int(_read(d / "size", "0") or 0) * 512
            if not size:
                continue
            rota = _read(d / "queue" / "rotational", "0") == "1"
            model = " ".join(_read(d / "device" / "model").split())
            kind = "NVMe" if name.startswith("nvme") else "HDD" if rota else "SSD"
            if not model and _read(d / "device" / "vendor"):
                model = " ".join(_read(d / "device" / "vendor").split())
            out.append({"name": name, "model": model, "size_b": size, "kind": kind,
                        "virtual": "virtual" in model.lower()})
        return out

    def name_of(self, majmin: str) -> str | None:
        p = self.sys / "dev" / "block" / majmin
        try:
            return Path(os.path.realpath(p)).name if p.exists() else None
        except OSError:
            return None

    def disks_of(self, dev: str, depth: int = 0) -> list[str]:
        if depth > 8:
            return []
        p = self.sys / "class" / "block" / dev
        if (p / "partition").exists():
            return self.disks_of(Path(os.path.realpath(p)).parent.name, depth + 1)
        try:
            slaves = sorted(os.listdir(p / "slaves"))
        except OSError:
            slaves = []
        if slaves:
            out: list[str] = []
            for s in slaves:
                out += [d for d in self.disks_of(s, depth + 1) if d not in out]
            return out
        return [dev]


def local_mounts(mounts: list[dict]) -> list[dict]:
    """Block-device filesystems of the host, one entry per device (bind mounts folded)."""
    best: dict[str, dict] = {}
    for m in mounts:
        if m["fstype"] not in LOCAL_FS:
            continue
        t = m["target"]
        if any(t == p or t.startswith(p + "/") for p in SKIP_PREFIXES):
            continue
        cur = best.get(m["dev"])
        key = (m["root"] != "/", len(t))
        if cur is None or key < (cur["root"] != "/", len(cur["target"])):
            best[m["dev"]] = m
    return sorted(best.values(), key=lambda m: m["target"])


def _same_dev(path: str, majmin: str) -> bool:
    try:
        ma, mi = (int(x) for x in majmin.split(":"))
        st = os.stat(path)
        return os.major(st.st_dev) == ma and os.minor(st.st_dev) == mi
    except (OSError, ValueError):
        return False


def read_disks(proc: str = "/proc", sys: str = "/sys", hostfs: str = HOSTFS) -> list[dict]:
    bd = BlockDevices(sys)
    disks = bd.disks()
    by_name = {d["name"]: {**d, "mounts": []} for d in disks}
    try:
        host_mounts = parse_mountinfo(Path(proc, "1", "mountinfo").read_text())
    except OSError:
        host_mounts = []
    other = []
    for m in local_mounts(host_mounts):
        u = None
        for cand in (hostfs_path(m["target"], hostfs), m["target"]):
            if _same_dev(cand, m["dev"]):  # a bind mount keeps the device number
                u = usage(cand)
                if u:
                    break
        dev = bd.name_of(m["dev"]) or m["source"].removeprefix("/dev/")
        size_b = int(_read(Path(sys, "class", "block", dev, "size"), "0") or 0) * 512
        entry = {"target": m["target"], "fstype": m["fstype"], "dev": dev,
                 "part_size_b": size_b or None, **(u or {})}
        owners = [d for d in bd.disks_of(dev) if d in by_name]
        if owners:
            for d in owners:
                by_name[d]["mounts"].append(entry)
        else:
            other.append(entry)
    out = list(by_name.values())
    if other:
        out.append({"name": "other", "model": "", "size_b": 0, "kind": "", "mounts": other})
    return out


def _covering_mount(path: str, mounts: list[dict]) -> dict | None:
    best = None
    for m in mounts:
        t = m["target"].rstrip("/") or "/"
        covers = path == t or path.startswith(t.rstrip("/") + "/")
        if covers and (best is None or len(t) > len(best["target"])):
            best = m
    return best


def read_network(paths: dict[str, str], proc: str = "/proc") -> list[dict]:
    """Shared drives, measured through paths this container sees ({label: probe path})."""
    try:
        own = parse_mountinfo(Path(proc, "self", "mountinfo").read_text())
    except OSError:
        own = []
    out = []
    for label, probe in paths.items():
        u = usage(probe)
        if not u:
            continue
        m = _covering_mount(probe, own) or {}
        src = m.get("source", "")
        base = probe.rsplit("/", 1)[-1]
        if probe != label and src.endswith("/" + base):  # per-user mount: show the export
            src = src[: -len(base) - 1]
        out.append({"mount": label, "source": src, "fstype": m.get("fstype", ""), **u})
    return out


def network_probes(home: str, state_root: str) -> dict[str, str]:
    """/datahome is often automounted per user, so measure it at the user's own subdir."""
    probes = {}
    dh = [p for p in (state_root.rsplit("/.insjob", 1)[0], "/datahome")
          if p.startswith("/datahome")]
    for p in dh:
        if os.path.isdir(p):
            probes["/datahome"] = p
            break
    if os.path.isdir(home):
        probes["/home"] = home
    return probes


def read_storage(home: str, state_root: str, proc: str = "/proc", sys: str = "/sys",
                 hostfs: str = HOSTFS) -> dict:
    return {"network": read_network(network_probes(home, state_root), proc),
            "disks": read_disks(proc, sys, hostfs), "updated": time.time()}


class StorageMonitor:
    """Reads storage in the background: statvfs on a hung NFS mount must not stop the agent."""

    def __init__(self, read, interval: float = 60):
        self.read = read
        self.interval = interval
        self._latest: dict = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while True:
            try:
                self._latest = self.read()
            except Exception as e:  # never kill the loop
                print(f"[insjob-agent] storage read failed: {e}", flush=True)
            time.sleep(self.interval)

    def latest(self) -> dict:
        return self._latest

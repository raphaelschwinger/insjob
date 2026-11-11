"""GPU / host resource readings (NVML + /proc) and the pure "which GPU is free" logic."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field


@dataclass
class GpuProc:
    pid: int
    mem_mb: int
    uid: int | None = None
    container: str | None = None  # full container id, if the process runs in docker
    cmd: str | None = None  # command line (secrets masked, truncated)


@dataclass
class GpuInfo:
    index: int
    name: str
    mem_total_mb: int
    mem_used_mb: int
    util: int
    temp: int | None = None
    procs: list[GpuProc] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_CGROUP_ID = re.compile(r"(?:docker-|/docker/)([0-9a-f]{64})")


CMD_MAX = 1000
_SECRET = re.compile(
    r"(?i)(?<!\S)(-{0,2}[\w.-]*(?:api[_-]?key|token|secret|passw(?:or)?d)[\w.-]*)(=|\s+)(\S+)")


def redact_cmd(cmd: str) -> str:
    """Mask values of secret-looking arguments: --token X, WANDB_API_KEY=X, password=X."""
    return _SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}***", cmd)


def proc_info(pid: int, proc: str = "/proc") -> tuple[int | None, str | None, str | None]:
    """(uid, container id, command line) of a host process; the agent runs with --pid=host."""
    uid = container = cmd = None
    try:
        with open(f"{proc}/{pid}/status") as f:
            for line in f:
                if line.startswith("Uid:"):
                    uid = int(line.split()[1])
                    break
        with open(f"{proc}/{pid}/cgroup") as f:
            m = _CGROUP_ID.search(f.read())
            if m:
                container = m.group(1)
        with open(f"{proc}/{pid}/cmdline", "rb") as f:
            raw = f.read(64 * 1024).rstrip(b"\0").split(b"\0")
        cmd = " ".join(a.decode(errors="replace") for a in raw if a) or None
        if cmd:
            cmd = redact_cmd(cmd)
            if len(cmd) > CMD_MAX:
                cmd = cmd[:CMD_MAX] + " …"
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return uid, container, cmd


def read_gpus() -> list[GpuInfo]:
    import pynvml  # only available inside the agent image

    pynvml.nvmlInit()
    try:
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except pynvml.NVMLError:
                util = 0
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temp = None
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            procs = []
            try:
                plist = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            except pynvml.NVMLError:
                plist = []
            for p in plist:
                uid, cont, cmd = proc_info(p.pid)
                used = (p.usedGpuMemory or 0) // 2**20 if p.usedGpuMemory else 0
                procs.append(GpuProc(pid=p.pid, mem_mb=int(used), uid=uid, container=cont,
                                     cmd=cmd))
            out.append(
                GpuInfo(
                    index=i,
                    name=name,
                    mem_total_mb=mem.total // 2**20,
                    mem_used_mb=mem.used // 2**20,
                    util=int(util),
                    temp=temp,
                    procs=procs,
                )
            )
        return out
    finally:
        pynvml.nvmlShutdown()


def read_host() -> dict:
    """Host-wide CPU and memory (loadavg and meminfo are not namespaced in docker)."""
    ncpu = os.cpu_count() or 1
    try:
        load1 = float(open("/proc/loadavg").read().split()[0])
    except OSError:
        load1 = 0.0
    mem_total = mem_avail = 0
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            if k == "MemTotal":
                mem_total = int(v.split()[0]) // 1024
            elif k == "MemAvailable":
                mem_avail = int(v.split()[0]) // 1024
    except OSError:
        pass
    return {
        "cpus": ncpu,
        "load1": load1,
        "cpus_free": max(0.0, ncpu - load1),
        "mem_total_mb": mem_total,
        "mem_avail_mb": mem_avail,
    }


def free_gpu_ids(
    gpus: list[GpuInfo], claimed: set[int], max_mem_used_mb: int = 1024, max_util: int = 5
) -> list[int]:
    return [
        g.index
        for g in gpus
        if g.index not in claimed
        and g.mem_used_mb <= max_mem_used_mb
        and g.util <= max_util
        and not g.procs
    ]


def pick_gpus(
    gpus: list[GpuInfo], free: list[int], n: int, min_mem_gb: float = 0
) -> list[int] | None:
    by_idx = {g.index: g for g in gpus}
    # 5% slack: a "24GB" card reports 24564 MiB
    ok = [i for i in free if by_idx[i].mem_total_mb >= min_mem_gb * 1024 * 0.95]
    if len(ok) < n:
        return None
    return sorted(ok)[:n]

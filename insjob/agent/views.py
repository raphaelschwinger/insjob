"""Read-only views shared by the web API and the CLI's remote ops."""

from __future__ import annotations

import time

from ..identity import Identity
from ..locks import LabStatus
from ..state import State


def overview(state: State, lab_status: LabStatus, ident: Identity, policy: dict,
             lab: dict, limit: int = 100) -> dict:
    lab_data = lab_status.read()
    own_hosts = state.read_hosts()
    users = lab_status.users_map()
    hosts = []
    for name, meta in lab["hosts"].items():
        info = lab_data["hosts"].get(name) or own_hosts.get(name)
        entry = {"host": name, "gpus_desc": meta.get("gpus", ""),
                 "allowed": name in policy["hosts"], "online": info is not None}
        if info:
            gpus = info.get("gpus", [])
            for g in gpus:
                for p in g.get("procs", []):
                    if p.get("uid") is not None:
                        p["user"] = users.get(str(p["uid"]), f"uid {p['uid']}")
                claim = (info.get("claims") or {}).get(str(g["index"]))
                g["claim"] = claim
            entry.update({k: info.get(k) for k in
                          ("cpus", "load1", "cpus_free", "mem_total_mb", "mem_avail_mb",
                           "updated")})
            entry["gpus"] = gpus
        hosts.append(entry)
    return {
        "me": {"user": ident.user, "uid": ident.uid, "group": policy["group"]},
        "policy": policy,
        "now": time.time(),
        "hosts": hosts,
        "lab": sorted(lab_data["users"], key=lambda u: (u["host"], u["user"])),
        "jobs": state.list_jobs(limit),
    }

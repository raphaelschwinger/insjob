"""Who is running insjob: discovered from the server, never hard-coded."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field

PORT_BASE = 20000


def agent_port(uid: int) -> int:
    """Per-user agent port, derived from the UID so users never collide on a host."""
    return PORT_BASE + uid % 10000


@dataclass
class Identity:
    user: str
    uid: int
    gid: int
    home: str
    state_root: str
    groups: list[str] = field(default_factory=list)
    docker_gid: int | None = None

    @property
    def port(self) -> int:
        return agent_port(self.uid)

    @property
    def agent_name(self) -> str:
        return f"insjob-agent-{self.user}"

    def expand(self, s: str) -> str:
        """Expand {user}, {uid}, {gid}, {home}, {state} placeholders in repo configs."""
        return (
            s.replace("{user}", self.user)
            .replace("{uid}", str(self.uid))
            .replace("{gid}", str(self.gid))
            .replace("{home}", self.home)
            .replace("{state}", self.state_root)
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["docker_gid"] is None:
            d.pop("docker_gid")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Identity:
        return cls(
            user=d["user"],
            uid=int(d["uid"]),
            gid=int(d["gid"]),
            home=d["home"],
            state_root=d["state_root"],
            groups=list(d.get("groups", [])),
            docker_gid=int(d["docker_gid"]) if d.get("docker_gid") is not None else None,
        )

    @classmethod
    def from_env(cls) -> Identity:
        """Inside the agent container, identity comes from INSJOB_* env vars."""
        e = os.environ
        return cls(
            user=e["INSJOB_USER"],
            uid=int(e["INSJOB_UID"]),
            gid=int(e["INSJOB_GID"]),
            home=e.get("HOME", f"/home/{e['INSJOB_USER']}"),
            state_root=e["INSJOB_STATE"],
            groups=[g for g in e.get("INSJOB_GROUPS", "").split(",") if g],
            docker_gid=int(e["INSJOB_DOCKER_GID"]) if e.get("INSJOB_DOCKER_GID") else None,
        )


# Script run once over SSH to discover the identity on the server.
DISCOVER_SCRIPT = r"""
set -e
echo "user=$(id -un)"
echo "uid=$(id -u)"
echo "gid=$(id -g)"
echo "groups=$(id -Gn | tr ' ' ',')"
echo "docker_gid=$(getent group docker | cut -d: -f3)"
echo "home=$HOME"
if [ -d "/datahome/$(id -un)" ] && [ -w "/datahome/$(id -un)" ]; then
  echo "state_root=/datahome/$(id -un)/.insjob"
else
  echo "state_root=$HOME/.insjob"
fi
"""


def parse_discover(output: str) -> Identity:
    kv = dict(
        line.split("=", 1) for line in output.strip().splitlines() if re.match(r"^\w+=", line)
    )
    return Identity(
        user=kv["user"],
        uid=int(kv["uid"]),
        gid=int(kv["gid"]),
        home=kv["home"],
        state_root=kv["state_root"],
        groups=[g for g in kv.get("groups", "").split(",") if g],
        docker_gid=int(kv["docker_gid"]) if kv.get("docker_gid") else None,
    )

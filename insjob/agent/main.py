"""Agent entry point (runs inside the insjob-agent-<user> container on each server)."""

from __future__ import annotations

import os
import secrets
import threading
from pathlib import Path

from ..identity import Identity
from ..policy import load_lab
from .scheduler import Agent


def ensure_token(state_root: str) -> str:
    p = Path(state_root) / "token"
    if not p.exists():
        p.write_text(secrets.token_urlsafe(24))
        os.chmod(p, 0o600)
    return p.read_text().strip()


def main() -> None:
    ident = Identity.from_env()
    host = os.environ["INSJOB_HOST"]
    agent = Agent(ident, host, load_lab())
    token = ensure_token(ident.state_root)

    import uvicorn

    from .api import make_app

    app = make_app(agent, token)
    port = int(os.environ.get("INSJOB_PORT", ident.port))
    threading.Thread(
        target=uvicorn.run, args=(app,),
        kwargs={"host": "0.0.0.0", "port": port, "log_level": "warning"}, daemon=True,
    ).start()
    agent.run_forever()


if __name__ == "__main__":
    main()

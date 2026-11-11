"""Optional push notifications (ntfy or a Mattermost incoming webhook).

Configured per user in <state>/config.toml:

    [notify]
    kind = "ntfy"                      # or "mattermost"
    url = "https://ntfy.sh/<secret-topic>"
    on = ["succeeded", "failed"]       # default
"""

from __future__ import annotations

import base64
import json
import tomllib
import urllib.request
from pathlib import Path


def load_user_config(state_root: str | Path) -> dict:
    try:
        with open(Path(state_root) / "config.toml", "rb") as f:
            return tomllib.load(f)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def message(job: dict) -> tuple[str, str]:
    icon = {"succeeded": "✅", "failed": "❌", "cancelled": "⏹", "skipped": "⏭"}.get(
        job.get("state", ""), "ℹ️"
    )
    title = f"{icon} job {job['id']} {job.get('state')} — {job.get('repo')} on {job.get('host')}"
    body = " ".join(job.get("cmd") or [])[:200]
    if job.get("exit_code") not in (None, 0):
        body += f"\nexit code {job['exit_code']}"
    if job.get("reason"):
        body += f"\n{job['reason']}"
    if job.get("wandb_url"):
        body += f"\n{job['wandb_url']}"
    return title, body


def send(cfg: dict, job: dict) -> None:
    n = cfg.get("notify") or {}
    if not n.get("url") or job.get("state") not in n.get("on", ["succeeded", "failed"]):
        return
    title, body = message(job)
    try:
        if n.get("kind", "ntfy") == "mattermost":
            data = json.dumps({"text": f"**{title}**\n{body}"}).encode()
            req = urllib.request.Request(n["url"], data=data,
                                         headers={"Content-Type": "application/json"})
        else:
            # RFC 2047 so emoji survive the latin-1 header encoding
            enc = "=?UTF-8?B?" + base64.b64encode(title.encode()).decode() + "?="
            req = urllib.request.Request(n["url"], data=body.encode(), headers={"Title": enc})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:  # never let a notification break the agent
        print(f"[insjob] notification failed: {e}", flush=True)

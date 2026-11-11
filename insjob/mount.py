"""Mount project data from the servers on the laptop (sshfs), and tell Claude Code about it."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

from .config import RepoConfig
from .policy import fqdn
from .remote import Remote, have

MNT_ROOT = Path(os.environ.get("INSJOB_MNT", "~/mnt/ins")).expanduser()
PREFERRED = ["ploen", "stohl"]  # mount /datahome straight from the file server

CLAUDE_NOTE_START = "<!-- insjob:data -->"
CLAUDE_NOTE_END = "<!-- /insjob:data -->"


def mounted() -> dict[str, str]:
    """mountpoint -> source, for sshfs mounts under MNT_ROOT."""
    out = subprocess.run(["mount"], capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        if " on " not in line:
            continue
        src, rest = line.split(" on ", 1)
        mp = rest.split(" (")[0].split(" type ")[0]
        if mp.startswith(str(MNT_ROOT)):
            res[mp] = src
    return res


def is_stale(mp: Path) -> bool:
    try:
        p = subprocess.run(["ls", str(mp)], capture_output=True, timeout=8)
        return p.returncode != 0
    except subprocess.TimeoutExpired:
        return True


def unmount(mp: Path) -> bool:
    for cmd in (["umount", str(mp)], ["diskutil", "unmount", "force", str(mp)],
                ["fusermount", "-uz", str(mp)]):
        if have(cmd[0]) and subprocess.run(cmd, capture_output=True).returncode == 0:
            return True
    return False


def mount_one(remote: Remote, name: str, path: str, *, rw: bool = False) -> Path:
    if not have("sshfs"):
        hint = ("brew install --cask macfuse && brew install gromgit/fuse/sshfs-mac"
                if platform.system() == "Darwin" else "sudo apt install sshfs")
        raise RuntimeError(f"sshfs not found — install it: {hint}")
    pol = remote.cfg.policy
    host = next((h for h in PREFERRED if h in pol["hosts"]), remote.cfg.host)
    mp = MNT_ROOT / name
    if str(mp) in mounted():
        if not is_stale(mp):
            return mp
        unmount(mp)
    mp.mkdir(parents=True, exist_ok=True)
    opts = ["reconnect", "ServerAliveInterval=15", "ServerAliveCountMax=4", "auto_cache",
            "follow_symlinks"]
    if not rw:
        opts.append("ro")
    if platform.system() == "Darwin":
        opts += ["defer_permissions", "noappledouble", "noapplexattr", f"volname={name}"]
        if _macfuse_major() >= 5 and _macos_major() >= 15:
            opts.append("backend=fskit")  # no kernel extension needed
    src = f"{remote.ident.user}@{fqdn(remote.cfg.lab, host)}:{path}"
    p = subprocess.run(["sshfs", src, str(mp), "-o", ",".join(opts)], capture_output=True,
                       text=True)
    if p.returncode != 0:
        msg = p.stderr.strip()
        if "file system is not available" in msg and _macfuse_major() < 5:
            msg += (f"\nmacFUSE {_macfuse_version()} has no kernel extension for macOS "
                    f"{platform.mac_ver()[0]}. Upgrade to macFUSE ≥ 5 (FSKit, no kext):\n"
                    "  brew upgrade --cask macfuse   # then enable it in System Settings → "
                    "General → Login Items & Extensions → File System Extensions")
        raise RuntimeError(f"sshfs failed: {msg}")
    return mp


def _macfuse_version() -> str:
    p = subprocess.run(["defaults", "read", "/Library/Filesystems/macfuse.fs/Contents/Info.plist",
                        "CFBundleVersion"], capture_output=True, text=True)
    return p.stdout.strip() or "0"


def _macfuse_major() -> int:
    try:
        return int(_macfuse_version().split(".")[0])
    except ValueError:
        return 0


def _macos_major() -> int:
    try:
        return int(platform.mac_ver()[0].split(".")[0])
    except ValueError:
        return 0


def register_with_claude(repo: RepoConfig, mounts: dict[str, tuple[Path, str]]) -> None:
    """Allow Claude Code to read the mounts, and leave it a note on how to use them."""
    settings = repo.root / ".claude" / "settings.local.json"
    settings.parent.mkdir(exist_ok=True)
    try:
        d = json.loads(settings.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        d = {}
    perms = d.setdefault("permissions", {})
    dirs = perms.setdefault("additionalDirectories", [])
    for mp, _ in mounts.values():
        if str(mp) not in dirs:
            dirs.append(str(mp))
    settings.write_text(json.dumps(d, indent=2) + "\n")

    lines = [CLAUDE_NOTE_START, "## Data (mounted by `job mount`)", ""]
    for name, (mp, src) in mounts.items():
        lines.append(f"- `{name}`: `{mp}` on this laptop = `{src}` on the INS servers "
                     "(same path inside job containers)")
    lines += [
        "",
        "The mounts are sshfs over the network: use them for light reads only (ls, configs,",
        "a few samples, small CSV/JSON). For anything that scans many files or GBs, run it",
        "next to the data instead: `job exec -g 0 -- python scripts/inspect.py` (streams",
        "output back) or a queued `job run -g 0 -- ...` for long jobs.",
        CLAUDE_NOTE_END,
    ]
    note = "\n".join(lines) + "\n"
    f = repo.root / "CLAUDE.local.md"
    text = f.read_text() if f.exists() else ""
    if CLAUDE_NOTE_START in text:
        pre, rest = text.split(CLAUDE_NOTE_START, 1)
        post = rest.split(CLAUDE_NOTE_END, 1)[1] if CLAUDE_NOTE_END in rest else ""
        text = pre + note + post.lstrip("\n")
    else:
        text = (text.rstrip() + "\n\n" if text.strip() else "") + note
    f.write_text(text)
    _gitignore_local(repo.root)


def _gitignore_local(root: Path) -> None:
    """CLAUDE.local.md / settings.local.json are personal — keep them out of git."""
    exclude = root / ".git" / "info" / "exclude"
    if not exclude.parent.exists():
        return
    text = exclude.read_text() if exclude.exists() else ""
    add = [p for p in ("CLAUDE.local.md", ".claude/settings.local.json") if p not in text]
    if add:
        exclude.write_text(text.rstrip("\n") + "\n" + "\n".join(add) + "\n")

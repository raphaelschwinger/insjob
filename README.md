# insjob — `job`

Run training, data processing, shells and notebooks on the **INS GPU servers** (ploen,
amur, kiel, stein, stohl) **from your laptop**. You don't need tmux, a server checkout or
"Reopen in Container" for every run.

```sh
job gpus                                             # who uses which GPU, on every server
job run -- python src/train.py experiment=x          # queue a run on a free GPU
job run -g 0 -c 16 -- python scripts/preprocess.py  # CPU-only data step
job run --after 12 --wait -- python src/train.py     # ... then train once 12 succeeded
job logs last -f                                     # follow the log
job notebook                                         # Jupyter on a server GPU, opened in VS Code
job mount                                            # project data on your laptop (read-only)
job ui                                               # dashboard: GPUs, queue, notifications
```

## How it works

- **Your laptop** only runs the `job` CLI, which talks to the servers over SSH.
- **Each server** runs a small *agent* container per user (`insjob-agent-<you>`,
  `--restart unless-stopped`). It:
  - takes your queued jobs when a GPU is free,
  - builds the image,
  - starts the job with `docker run -d`, so it keeps running when you close the laptop,
  - records the result,
  - serves the dashboard.
- **Code**: a job runs a **pushed git commit**. The server keeps a mirror of your repo and
  checks out a frozen copy of that commit for every run, under `<state>/runs/<id>/code`.
- **Image**: always the project repo's own `.devcontainer/` (devcontainer.json +
  Dockerfile), the same one VS Code uses.
  - Kept from devcontainer.json: build args, `--shm-size`, mounts, env.
  - Dropped: `--gpus all`, extensions, lifecycle hooks.
  - insjob sets the GPUs, CPUs and memory itself, following the INS Docker guidelines.
- **You, not root**: the container runs as **your server UID**, so `ps`, `nvtop` and file
  owners show who started what. A thin layer on top of the project image takes care of
  this:
  - if the Dockerfile has `ARG USER_UID/USER_GID`, insjob builds it with your UID;
  - otherwise it remaps the devcontainer's `remoteUser` to your UID, or creates you as a
    user.

  Containers are named `job-<you>-<id>` and labeled with repo, commit and command.
- **Fair sharing**:
  - Agents of different users coordinate through claim files in `/var/tmp/insjob`, so two
    people never get the same GPU.
  - GPUs busy with work started outside insjob are left alone too.
  - Everyone's agent publishes a small public status to `/datahome/insjob/status/`, so
    every dashboard shows the whole lab.
- **Policy**: `/datahome/insjob/lab.toml` (defaults are in `insjob/policy.py`) decides who
  may use which server.
  - Students (unix group `stud`): amur, at most 2 GPUs per host.
  - Staff (`ins`): all servers.

  You can narrow it for yourself in `<state>/config.toml` (`[policy] hosts = [...]`), but
  not widen it.
- **State** lives in your own server space: `/datahome/<you>/.insjob`, or `~/.insjob`.
  - `queue/`, `claimed/<host>/`, `done/`: the job records
  - `runs/<id>/`: `code/`, `log`, `outputs/`
  - `events.jsonl`: the notification feed
  - `token`: for the dashboard
  - `secrets.env`: env vars for your jobs, e.g. `WANDB_API_KEY`
  - `config.toml`: notifications and policy narrowing

## Install (students: start here)

You need:
- an INS server account (ask Torge);
- membership in the `docker` group;
- SSH key login (`ssh-copy-id <you>@amur.ins.informatik.uni-kiel.de`);
- macOS or Linux. On Windows, use WSL2.

```sh
uv tool install git+https://github.com/raphaelschwinger/insjob
job setup
```

`job setup`:
1. finds your account (user, UID, groups);
2. asks for an optional W&B key;
3. installs your agent on every server your group may use;
4. installs the Claude Code skill (`~/.claude/skills/insjob`).

Re-run it any time. `job agent up` starts or installs your agent on every server where it
isn't running (e.g. after a server reboot); `job agent upgrade` rebuilds all agents after
updating insjob.

## Use it in a project

```sh
cd my-project          # must have .devcontainer/ (template: raphaelschwinger/cuda-python-devcontainer)
job init               # writes .insjob.toml and VS Code tasks (Terminal → Run Task → job: …)
git add .insjob.toml && git commit -m "Add insjob config" && git push
job run -- python -c "import torch; print(torch.cuda.get_device_name())"
```

`.insjob.toml` is committed and works for everyone: `{user}` expands to whoever runs the
job.

```toml
name = "fastrl"
devcontainer = ".devcontainer/devcontainer.json"
setup = "uv sync --frozen"          # before every command, inside the container
[resources]
gpus = 1
cpus = 8
memory = "64g"
min_gpu_mem_gb = 0
[env]
UV_CACHE_DIR = "/datahome/{user}/.cache/uv"
[data]                              # for `job mount`
episodicrl = "/datahome/{user}/episodicrl"
[mounts]                            # extra job mounts; /datahome is always mounted read-only
extra = ["/datahome/{user}/episodicrl/processed:/data/processed:rw"]
```

## Commands

| | |
|---|---|
| `job gpus` | GPUs on all servers and who uses them (`--json`) |
| `job run [-g N] [-c CPUS] [-m 64g] [--min-mem GB] [-H host\|any] [--gpu-ids 1] [--wait] [--after ID] [--push] -- CMD` | queue a batch job. Without `--wait` it fails if nothing is free now. `-g 0` = CPU-only data job. `--gpu-ids` (needs `-H`) pins GPU indices on that host; the job waits for exactly those |
| `job exec [-g 0] -- CMD` | one-off command in the container, output streamed back (heavy data inspection) |
| `job shell [-g 0]` | interactive bash in the container on a server |
| `job notebook [-g 1]` | Jupyter in the container plus an SSH tunnel; paste the URL into VS Code → *Select Kernel → Existing Jupyter Server* |
| `job ls [-a] [--json]` · `job logs ID\|last [-f]` · `job stop ID` | manage jobs |
| `job pull ID` | copy the job's `/insjob-out` to `./insjob-out/ID` |
| `job mount [name\|/server/path] [--rw]` · `job umount` · `job mount --status` | sshfs-mount data on the laptop, at `~/mnt/ins/<name>`. Claude Code is allowed to read it (`.claude/settings.local.json`) and gets a note in `CLAUDE.local.md` |
| `job data ls [path]` · `job data pull PATH [DEST]` | browse data, or copy a small piece of it |
| `job ui` | dashboard in the browser (over an SSH tunnel) |
| `job agent up\|status\|install\|upgrade\|logs\|uninstall [-H host]` | your agents. `up` fixes only the servers where the agent isn't running |

In jobs:
- `$INSJOB_ID`, `$INSJOB_HOST` and `$INSJOB_OUT` (= `/insjob-out`) are set.
- `CUDA_VISIBLE_DEVICES` covers only your GPUs.
- Logs are kept in `runs/<id>/log`.
- A W&B run URL in the log is picked up and shown.

## Notifications

The dashboard shows browser notifications while it is open. For pushes to your phone,
add this to `<state>/config.toml` on the server:

```toml
[notify]
kind = "ntfy"                                  # or "mattermost" (incoming webhook URL)
url = "https://ntfy.sh/<long-random-topic>"
on = ["succeeded", "failed"]
```

## Troubleshooting

- **"HEAD is not pushed" / "uncommitted changes"**: the server runs exactly your pushed
  commit. Commit and push, or use `--push`.
- **exit 97**: the container didn't run as your UID. Check the uid-layer lines at the top
  of `job logs ID`.
- **exit 98**: the `setup` command failed.
- **"mount source does not exist"**: a path in `.insjob.toml` is wrong for this host or user.
- **Agent problems**: `job agent up`, then `job agent logs -H <host>`, then
  `job agent install -H <host>`.
- **Stale mount after sleep**: `job mount --status`, then `job mount`.
- **Security note**: anyone in the `docker` group can inspect or enter any container on a
  host, so don't treat containers as a boundary between lab members. Keep secrets in
  `secrets.env` (mode 600), never in `.insjob.toml`.

## Development

```sh
uv sync && uv run pytest
uv tool install -e .     # use the working copy as `job`
```

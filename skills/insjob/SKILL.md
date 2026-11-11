---
name: insjob
description: Run and monitor experiments, data-processing steps, shells and notebooks on the INS GPU servers (ploen, amur, kiel, stein, stohl) from the laptop with the `job` CLI, and access server data. Use whenever the user wants to train/launch/queue an experiment, check free GPUs, look at job logs or failures, preprocess or inspect a dataset that lives on /datahome, open a notebook on a GPU, or mount project data.
---

# insjob — the `job` CLI for the INS GPU servers

Jobs run on the servers inside the **project repo's own devcontainer image**
(`.devcontainer/`), as the user's own server UID, from a **pushed git commit**. The
laptop never needs the data or a GPU. Repo settings live in `.insjob.toml`
(`job init` creates it).

## Commands

| Goal | Command |
|---|---|
| Who uses which GPU right now | `job gpus` (`--json` for parsing) |
| Queue a training run | `job run [-g 1] [-H ploen\|any] [--min-mem 40] [--wait] -- python src/train.py experiment=x` |
| CPU-only data processing | `job run -g 0 -c 16 -- python scripts/preprocess.py ...` |
| Chain steps | `job run --after <id> -- ...` (starts only if `<id>` succeeded) |
| Quick command next to the data (streams back) | `job exec [-g 0] -- python -c "..."` |
| My jobs | `job ls` (`-a` all, `--json`) |
| Logs | `job logs <id\|last>` (`-f` to follow, `-n 500`) |
| Stop / cancel | `job stop <id>` |
| Notebook on a server GPU | `job notebook` → prints a Jupyter URL for VS Code "Existing Jupyter Server" |
| Copy outputs back | `job pull <id>` (everything the job wrote to `/insjob-out`) |
| Mount project data on the laptop | `job mount` (names from `[data]` in `.insjob.toml`, read-only) |
| Dashboard | `job ui` |
| Agent down / "no running agent" | `job agent up` (starts or installs it only where needed) |

## Rules

1. **Commit and push first.** `job run` refuses to run with uncommitted changes or an
   unpushed HEAD/submodule, because the server checks out that exact commit. Ask the
   user before committing. `--push` pushes for them. `job exec` and `job shell` fall
   back to the last pushed commit, with a warning.
2. **A single quoted argument is a shell string**: `job run -- "python a.py && python b.py"`.
   Several arguments are quoted individually.
3. Without `--wait`, `job run` fails if no GPU is free right now and prints `job gpus`. When
   the user wants to queue, use `--wait`. Don't loop or poll to grab GPUs.
4. Lab etiquette: don't ask for more GPUs than needed. The lab policy caps GPUs per host and
   the hosts a group may use (students: amur). Never try to get around it.
5. **Data access:**
   - `/datahome` is mounted read-only into every job at the same path.
   - Writable paths come from `[mounts] extra` in `.insjob.toml`. Outputs go to `$INSJOB_OUT`
     (`/insjob-out`).
   - If the data is mounted on the laptop (`~/mnt/ins/<name>`, see `CLAUDE.local.md`), use it
     only for **light reads**: ls, configs, a few samples.
   - Anything that scans many files or GBs runs on the server with `job exec -g 0 -- ...`
     (short, streams back) or `job run -g 0 -- ...` (long, queued).
6. To debug a failed job, run `job logs <id> -n 300`. The `[insjob]` lines show the host,
   the GPUs, the UID check, setup and the command.
   - Exit 97: wrong UID in the container.
   - Exit 98: the `setup` step failed.
   - "mount source does not exist": a path in `.insjob.toml` is wrong.
7. W&B: when the log contains a run URL, `job ls --json` / `job logs` show `wandb_url`.

## Typical flows

- "Train X on a free 4090":
  1. `job gpus`
  2. commit and push (after asking)
  3. `job run -H ploen -- python src/train.py experiment=X`
  4. report the job id, plus `job logs <id> -f`
- "Preprocess, then train":
  1. `id=$(job run -g 0 -c 16 --wait -- python scripts/prep.py | grep -o 'job [0-9]*' | head -1 | cut -d' ' -f2)`
  2. `job run --after $id --wait -- python src/train.py`
- "How big is the dataset / what's in it": `job exec -g 0 -- du -sh /datahome/<user>/<name>`
  and `job exec -g 0 -- python -c "..."`, rather than walking the sshfs mount.

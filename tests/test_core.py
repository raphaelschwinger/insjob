import json
import threading
import time
import tomllib

import pytest

from insjob import devcontainer as dc
from insjob.agent import gpu
from insjob.agent.docker_ops import command_string, job_script, run_args
from insjob.agent.scheduler import Agent
from insjob.config import dump_toml
from insjob.identity import Identity, agent_port, parse_discover
from insjob.locks import GpuLocks, LabStatus
from insjob.policy import DEFAULT_LAB, deep_merge, size_gb, user_policy
from insjob.state import State


def ident(tmp_path, user="alice", uid=5017, groups=("ins", "docker")):
    return Identity(user=user, uid=uid, gid=5001, home=f"/home/{user}",
                    state_root=str(tmp_path / f"state-{user}"), groups=list(groups),
                    docker_gid=6001)


# ---- identity / policy --------------------------------------------------------------
def test_ports_differ_per_user():
    assert agent_port(5017) == 25017
    assert agent_port(1055) != agent_port(5042)


def test_expand_placeholders(tmp_path):
    i = ident(tmp_path)
    assert i.expand("/datahome/{user}/x:{uid}") == "/datahome/alice/x:5017"


def test_parse_discover():
    out = "user=bob\nuid=5042\ngid=5002\ngroups=stud,docker\ndocker_gid=6001\nhome=/home/bob\n" \
          "state_root=/datahome/bob/.insjob\n"
    i = parse_discover(out)
    assert (i.user, i.uid, i.groups, i.docker_gid) == ("bob", 5042, ["stud", "docker"], 6001)


def test_policy_groups_and_narrowing():
    staff = user_policy(DEFAULT_LAB, ["ins", "docker"])
    assert staff["group"] == "staff" and "kiel" in staff["hosts"]
    stud = user_policy(DEFAULT_LAB, ["stud", "docker"])
    assert stud["hosts"] == ["amur"] and stud["max_gpus_per_host"] == 2
    unknown = user_policy(DEFAULT_LAB, ["other"])
    assert unknown["group"] == "student"
    # narrowing can restrict but never widen
    n = user_policy(DEFAULT_LAB, ["stud"], {"hosts": ["amur", "kiel"], "max_gpus_per_host": 9})
    assert n["hosts"] == ["amur"] and n["max_gpus_per_host"] == 2


def test_shared_lab_override():
    lab = deep_merge(DEFAULT_LAB, {"groups": {"student": {"hosts": ["amur", "ploen"]}}})
    assert user_policy(lab, ["stud"])["hosts"] == ["amur", "ploen"]
    assert DEFAULT_LAB["groups"]["student"]["hosts"] == ["amur"]  # not mutated


def test_size_gb():
    assert size_gb("32g") == 32 and size_gb("126GB") == 126 and size_gb("512m") == 0.5


def test_dump_toml_roundtrip():
    d = {"default_host": "ploen", "identity": {"user": "a", "uid": 1, "groups": ["x", "y"]},
         "lab": {"hosts": {"ploen": {"fqdn": "p.example", "gpus": "4x"}}}}
    assert tomllib.loads(dump_toml(d)) == d


# ---- devcontainer ----------------------------------------------------------------------
DC = {
    "build": {"dockerfile": "Dockerfile", "args": {"USER_UID": "${localEnv:UID:1000}",
                                                    "USER_GID": "${localEnv:GID}"}},
    "workspaceFolder": "/workspaces/${localWorkspaceFolderBasename}",
    "runArgs": ["--gpus", "all", "--shm-size=32gb", "--name", "x", "--ipc", "host"],
    "mounts": [
        "source=/datahome/${localEnv:USER}/H2Smart,target=${containerWorkspaceFolder}/data,type=bind",
        "source=claude-code-config,target=/home/vscode/.claude,type=volume",
        "source=${localEnv:HOME}/.ssh,target=/home/vscode/.ssh-host,type=bind,readonly",
    ],
    "remoteEnv": {"ROS1_MOUNTS": "${localWorkspaceFolder}:${containerWorkspaceFolder}"},
    "remoteUser": "vscode",
}
ENV = {"UID": "5017", "GID": "5001", "USER": "alice", "HOME": "/home/alice"}


def test_resolve_devcontainer():
    s = dc.resolve(DC, devcontainer_rel=".devcontainer/devcontainer.json", repo_name="proj",
                   host_workspace="/st/runs/3/code", local_env=ENV)
    assert s.dockerfile == ".devcontainer/Dockerfile" and s.context == ".devcontainer"
    assert s.build_args == {"USER_UID": "5017", "USER_GID": "5001"}  # server uid, not laptop
    assert s.workspace_folder == "/workspaces/proj"
    assert s.run_args == ["--shm-size=32gb", "--ipc", "host"]  # --gpus/--name dropped
    assert s.mounts == ["type=bind,src=/datahome/alice/H2Smart,dst=/workspaces/proj/data"]
    assert s.env["ROS1_MOUNTS"] == "/st/runs/3/code:/workspaces/proj"
    assert s.remote_user == "vscode"
    assert any("volume" in n for n in s.notes) and any(".ssh" in n for n in s.notes)


def test_localenv_default():
    assert dc.substitute("${localEnv:NOPE:1000}", {}, {}) == "1000"


def test_uid_args_only_if_declared():
    s = dc.ContainerSpec(dockerfile="Dockerfile")
    dc.apply_uid_build_args(s, "FROM x\nRUN true\n", 5017, 5001)
    assert "USER_UID" not in s.build_args
    dc.apply_uid_build_args(s, "FROM x\nARG USER_UID=1000\nARG USER_GID=1000\n", 5017, 5001)
    assert s.build_args == {"USER_UID": "5017", "USER_GID": "5001"}


def test_uid_layer_covers_three_cases():
    import base64
    import re

    df = dc.uid_layer_dockerfile("base:1", 5017, 5001, "alice", "vscode")
    assert df.startswith("FROM base:1\nUSER root\n") and "USER 5017:5001" in df
    script = base64.b64decode(re.search(r"echo (\S+) \| base64", df).group(1)).decode()
    assert "getent passwd \"$UID_\"" in script  # case 1: already present
    assert "usermod -u" in script               # case 2: remap remoteUser
    assert "useradd -m -u" in script            # case 3: create user


def test_uid_layer_runs_in_a_shell(tmp_path):
    """The generated script must at least be valid POSIX sh."""
    import base64
    import re
    import subprocess

    df = dc.uid_layer_dockerfile("b", 1, 1, "a", None)
    script = base64.b64decode(re.search(r"echo (\S+) \| base64", df).group(1)).decode()
    assert subprocess.run(["sh", "-n"], input=script, text=True).returncode == 0


# ---- state ------------------------------------------------------------------------------
def make_job(state: State, **kw) -> dict:
    jid = state.new_job_id()
    job = {"id": jid, "repo": "proj", "kind": "run", "cmd": ["python", "t.py"], "gpus": 1,
           "cpus": 8, "memory": "64g", "hosts": ["any"], "after": [], "min_gpu_mem_gb": 0,
           "spec": dc.ContainerSpec(dockerfile="Dockerfile", image_hash="abc").to_dict(),
           "code_dir": str(state.run_dir(jid) / "code"), "run_dir": str(state.run_dir(jid)),
           **kw}
    state.submit(job)
    return job


def test_ids_are_unique_under_concurrency(tmp_path):
    s = State(tmp_path / "st")
    s.ensure()
    ids, threads = [], [threading.Thread(target=lambda: ids.append(s.new_job_id()))
                        for _ in range(30)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(ids) == list(range(1, 31))


def test_only_one_host_claims(tmp_path):
    s = State(tmp_path / "st")
    s.ensure()
    j = make_job(s)
    wins = []
    threads = [threading.Thread(target=lambda h=h: wins.append(h) if s.claim(j["id"], h) else 0)
               for h in ("ploen", "kiel", "stein", "amur")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(wins) == 1
    assert s.get(j["id"]) is not None


def test_cancel_vs_claim(tmp_path):
    s = State(tmp_path / "st")
    s.ensure()
    j = make_job(s)
    assert s.request_cancel(j["id"]) == "cancelled"
    assert not s.claim(j["id"], "ploen")
    assert s.get(j["id"])["state"] == "cancelled"


def test_dependencies(tmp_path):
    s = State(tmp_path / "st")
    s.ensure()
    a = make_job(s)
    b = make_job(s, after=[a["id"]])
    assert s.dep_status(b) == "wait"
    s.claim(a["id"], "ploen")
    s.finish(s.get(a["id"]), "ploen", "succeeded")
    assert s.dep_status(b) == "ok"
    c = make_job(s)
    d = make_job(s, after=[c["id"]])
    s.request_cancel(c["id"])
    assert s.dep_status(d) == "skip"


def test_events(tmp_path):
    s = State(tmp_path / "st")
    s.ensure()
    make_job(s)
    t = time.time()
    assert [e["kind"] for e in s.events()] == ["queued"]
    assert s.events(since=t + 1) == []


# ---- locks (two different users on one host) ----------------------------------------------
def test_cross_user_claims(tmp_path):
    a = GpuLocks(tmp_path / "locks", "alice")
    b = GpuLocks(tmp_path / "locks", "bob")
    assert a.acquire([0, 1], job_id=1)
    assert not b.acquire([1], job_id=7)       # taken by alice
    assert b.acquire([2], job_id=7)
    assert set(a.claims()) == {0, 1, 2}
    a.release(1)
    assert b.acquire([1], job_id=8)


def test_cross_user_race(tmp_path):
    users = [GpuLocks(tmp_path / "locks", f"u{i}") for i in range(8)]
    wins = []
    ts = [threading.Thread(target=lambda u=u: wins.append(u.user) if u.acquire([3], 1) else 0)
          for u in users]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(wins) == 1


def test_stale_claims_ignored(tmp_path):
    a = GpuLocks(tmp_path / "locks", "alice", stale_s=0.05)
    b = GpuLocks(tmp_path / "locks", "bob", stale_s=0.05)
    a.acquire([0], 1)
    time.sleep(0.1)  # alice's agent died: no heartbeat
    assert b.acquire([0], 2)


def test_heartbeat_drops_finished(tmp_path):
    a = GpuLocks(tmp_path / "locks", "alice")
    a.acquire([0], 1)
    a.acquire([1], 2)
    a.heartbeat({2})
    assert set(a.claims()) == {1}


def test_lab_status_and_users(tmp_path):
    ls = LabStatus(tmp_path / "lab")
    ls.publish("ploen", "alice", 5017, [{"id": 1}])
    ls.publish_gpus("ploen", {"by": "alice", "gpus": []})
    ls.add_users({"5017": "alice"}, "alice")
    ls.add_users({"5042": "bob"}, "bob")
    r = ls.read()
    assert "ploen" in r["hosts"] and r["users"][0]["user"] == "alice"
    assert ls.users_map() == {"5017": "alice", "5042": "bob"}


# ---- gpu selection ----------------------------------------------------------------------
def G(i, used=0, util=0, total=24564, procs=()):
    return gpu.GpuInfo(index=i, name="RTX 4090", mem_total_mb=total, mem_used_mb=used,
                       util=util, procs=[gpu.GpuProc(pid=1, mem_mb=1) for _ in procs])


def test_free_and_pick():
    gs = [G(0, used=20000, util=90, procs=[1]), G(1), G(2, used=300), G(3, util=50)]
    free = gpu.free_gpu_ids(gs, claimed={2})
    assert free == [1]
    assert gpu.pick_gpus(gs, [1, 2], 2) == [1, 2]
    assert gpu.pick_gpus(gs, [1], 2) is None
    assert gpu.pick_gpus(gs, [1], 1, min_mem_gb=24) == [1]   # 24564 MiB counts as 24 GB
    assert gpu.pick_gpus(gs, [1], 1, min_mem_gb=40) is None


# ---- container args ----------------------------------------------------------------------
def test_command_string():
    assert command_string(["python", "a b.py"]) == "python 'a b.py'"
    assert command_string(["python a.py && python b.py"]) == "python a.py && python b.py"


def test_run_args(tmp_path):
    i = ident(tmp_path)
    s = State(i.state_root)
    s.ensure()
    j = make_job(s, gpu_ids=[1, 3], host="ploen", setup="uv sync",
                 env={"UV_CACHE_DIR": "/datahome/{user}/uv"})
    a = run_args(j, i, image="img:1")
    joined = " ".join(a)
    assert a[a.index("--user") + 1] == "5017:5001"
    assert a[a.index("--gpus") + 1] == '"device=1,3"'
    assert "NVIDIA_VISIBLE_DEVICES=1,3" in a
    assert f"--name job-alice-{j['id']}" in joined and "insjob.user=alice" in joined
    assert "UV_CACHE_DIR=/datahome/alice/uv" in a
    cpu = run_args({**j, "gpu_ids": []}, i, image="img:1")
    assert "--gpus" not in cpu and "NVIDIA_VISIBLE_DEVICES=void" in cpu


def test_reservation_capped_at_memory(tmp_path):
    i = ident(tmp_path)
    s = State(i.state_root)
    s.ensure()
    j = make_job(s, memory="16g", memory_reservation="32g")
    a = run_args(j, i, image="img:1")
    assert a[a.index("--memory-reservation") + 1] == "16g"
    j2 = make_job(s, memory="64g", memory_reservation="32g")
    a2 = run_args(j2, i, image="img:1")
    assert a2[a2.index("--memory-reservation") + 1] == "32g"


def test_job_script_checks_uid(tmp_path):
    i = ident(tmp_path)
    s = State(i.state_root)
    s.ensure()
    j = make_job(s, setup="uv sync", cmd=["echo", "$HOME"])
    sc = job_script(j, i)
    assert '!= "5017"' in sc and "exit 97" in sc and "tee -a /insjob-run/log" in sc
    import subprocess

    assert subprocess.run(["bash", "-n"], input=sc, text=True).returncode == 0


# ---- scheduler with fakes -------------------------------------------------------------------
class FakeDocker:
    def __init__(self):
        self.containers: dict[str, dict] = {}
        self.started: list[dict] = []

    def list_job_containers(self, user):
        return dict(self.containers)

    def all_containers(self):
        return {}

    def ensure_image(self, job, ident, log):
        return "img"

    def run_job(self, job, ident, image, log):
        name = f"job-{ident.user}-{job['id']}"
        self.containers[name] = {"status": "running", "exit_code": 0, "ports": {}}
        self.started.append(job)
        return name

    def stop(self, name):
        self.containers[name]["status"] = "exited"
        self.containers[name]["exit_code"] = 137

    def remove(self, name):
        self.containers.pop(name, None)


def make_agent(tmp_path, user="alice", groups=("ins",), host="ploen", gpus=None, docker=None):
    i = ident(tmp_path, user=user, groups=groups)
    lab = deep_merge(DEFAULT_LAB, {"paths": {"lab_dir": str(tmp_path / "lab"),
                                             "lock_dir": str(tmp_path / "locks")}})
    gs = gpus if gpus is not None else [G(0), G(1), G(2), G(3)]
    return Agent(i, host, lab, docker=docker or FakeDocker(), read_gpus=lambda: gs,
                 read_host=lambda: {"cpus": 64, "load1": 1, "cpus_free": 63,
                                    "mem_total_mb": 128000, "mem_avail_mb": 100000})


def wait_launch(agent):
    for t in list(agent.launching.values()):
        t.join(5)


def test_agent_full_lifecycle(tmp_path):
    ag = make_agent(tmp_path)
    j = make_job(ag.state, gpus=2)
    ag.tick()
    wait_launch(ag)
    rec = ag.state.get(j["id"])
    assert rec["state"] == "running" and rec["gpu_ids"] == [0, 1] and rec["host"] == "ploen"
    assert set(ag.locks.claims()) == {0, 1}
    ag.docker.containers[f"job-alice-{j['id']}"].update(status="exited", exit_code=0)
    ag.tick()
    assert ag.state.get(j["id"])["state"] == "succeeded"
    assert ag.locks.claims() == {}
    kinds = [e["kind"] for e in ag.state.events()]
    assert kinds[:2] == ["queued", "starting"] and "succeeded" in kinds


def test_agents_of_two_users_share_gpus(tmp_path):
    a = make_agent(tmp_path, user="alice")
    b = make_agent(tmp_path, user="bob", groups=("ins",))
    ja = make_job(a.state, gpus=3)
    jb = make_job(b.state, gpus=2)
    a.tick()
    wait_launch(a)
    b.tick()
    wait_launch(b)
    assert a.state.get(ja["id"])["state"] == "running"
    assert b.state.get(jb["id"])["state"] == "queued"   # only 1 GPU left for bob
    jb1 = make_job(b.state, gpus=1)
    b.tick()
    wait_launch(b)
    assert b.state.get(jb1["id"])["gpu_ids"] == [3]


def test_pinned_gpu_ids(tmp_path):
    ag = make_agent(tmp_path, gpus=[G(0), G(1)])
    j = make_job(ag.state, gpus=1, gpu_ids_req=[1])
    ag.tick()
    wait_launch(ag)
    assert ag.state.get(j["id"])["gpu_ids"] == [1]      # not the lowest free one (0)


def test_pinned_gpu_waits_while_claimed(tmp_path):
    a = make_agent(tmp_path, user="alice", gpus=[G(0), G(1)])
    b = make_agent(tmp_path, user="bob", gpus=[G(0), G(1)])
    ja = make_job(a.state, gpus=1, gpu_ids_req=[1])
    a.tick()
    wait_launch(a)
    jb = make_job(b.state, gpus=1, gpu_ids_req=[1])
    b.tick()
    assert b.state.get(jb["id"])["state"] == "queued"    # GPU 0 is free, but not requested
    assert a.state.get(ja["id"])["gpu_ids"] == [1]


def test_parse_gpu_ids():
    from insjob.cli import parse_gpu_ids
    assert parse_gpu_ids(None) == [] and parse_gpu_ids("") == []
    assert parse_gpu_ids("1") == [1] and parse_gpu_ids("3, 1,1") == [1, 3]


def test_fits_now_respects_pinned_ids():
    from insjob.cli import fits_now
    gpu0 = {"index": 0, "mem_used_mb": 0, "mem_total_mb": 97887}
    gpu1 = {"index": 1, "mem_used_mb": 0, "mem_total_mb": 97887, "claim": {"user": "x"}}
    ov = {"hosts": [{"host": "stohl", "allowed": True, "online": True, "gpus": [gpu0, gpu1]}]}
    job = {"hosts": ["stohl"], "gpus": 1, "min_gpu_mem_gb": 0}
    assert fits_now(ov, job) == "stohl"
    assert fits_now(ov, {**job, "gpu_ids_req": [1]}) is None
    assert fits_now(ov, {**job, "gpu_ids_req": [0]}) == "stohl"


def test_student_policy_blocks_other_hosts(tmp_path):
    ag = make_agent(tmp_path, user="stud", groups=("stud",), host="ploen")
    j = make_job(ag.state, gpus=1)
    ag.tick()
    assert ag.state.get(j["id"])["state"] == "queued"   # ploen is not for students
    ag2 = make_agent(tmp_path, user="stud", groups=("stud",), host="amur")
    ag2.tick()
    wait_launch(ag2)
    assert ag2.state.get(j["id"])["state"] == "running"


def test_max_gpus_per_host(tmp_path):
    ag = make_agent(tmp_path, user="stud", groups=("stud",), host="amur")
    j1 = make_job(ag.state, gpus=2)
    j2 = make_job(ag.state, gpus=1)
    ag.tick()
    wait_launch(ag)
    assert ag.state.get(j1["id"])["state"] == "running"
    assert ag.state.get(j2["id"])["state"] == "queued"   # students: max 2 per host


def test_cancel_running_and_dependency_skip(tmp_path):
    ag = make_agent(tmp_path)
    a = make_job(ag.state, gpus=1)
    b = make_job(ag.state, gpus=1, after=[a["id"]])
    ag.tick()
    wait_launch(ag)
    assert ag.state.get(b["id"])["state"] == "queued"
    assert ag.state.request_cancel(a["id"]) == "stopping"
    ag.tick()  # stops
    ag.tick()  # reaps, then skips b
    assert ag.state.get(a["id"])["state"] == "cancelled"
    assert ag.state.get(b["id"])["state"] == "skipped"


def test_cpu_job_takes_no_gpu(tmp_path):
    ag = make_agent(tmp_path)
    j = make_job(ag.state, gpus=0, cpus=16)
    ag.tick()
    wait_launch(ag)
    rec = ag.state.get(j["id"])
    assert rec["state"] == "running" and rec["gpu_ids"] == []


def test_restart_requeues_unstarted(tmp_path):
    ag = make_agent(tmp_path)
    j = make_job(ag.state, gpus=1)
    ag.state.claim(j["id"], "ploen")
    rec = {**ag.state.get(j["id"]), "state": "starting", "host": "ploen", "gpu_ids": [0]}
    ag.state.update_claimed("ploen", rec)
    ag2 = make_agent(tmp_path)  # fresh agent, no container exists
    ag2.reconcile({})
    assert ag2.state.get(j["id"])["state"] == "queued"


def test_launch_failure_is_reported(tmp_path):
    class Broken(FakeDocker):
        def ensure_image(self, job, ident, log):
            raise RuntimeError("Dockerfile not found")

    ag = make_agent(tmp_path, docker=Broken())
    j = make_job(ag.state, gpus=1)
    ag.tick()
    wait_launch(ag)
    rec = ag.state.get(j["id"])
    assert rec["state"] == "failed" and "Dockerfile" in rec["reason"]
    assert ag.locks.claims() == {}


def test_api(tmp_path):
    from fastapi.testclient import TestClient

    from insjob.agent.api import make_app

    ag = make_agent(tmp_path)
    make_job(ag.state, gpus=1)
    ag.tick()
    wait_launch(ag)
    c = TestClient(make_app(ag, "secret"))
    assert c.get("/api/overview").status_code == 401
    d = c.get("/api/overview", headers={"Authorization": "Bearer secret"}).json()
    assert d["me"]["user"] == "alice" and d["jobs"][0]["state"] == "running"
    ploen = next(h for h in d["hosts"] if h["host"] == "ploen")
    assert ploen["online"] and ploen["gpus"][0]["claim"]["user"] == "alice"
    assert c.get("/").status_code == 200
    r = c.post("/api/jobs/1/stop?token=secret").json()
    assert r["result"] == "stopping"
    assert isinstance(json.loads(c.get("/api/events?token=secret").text), list)


@pytest.mark.parametrize("bad", [{"dockerComposeFile": "x.yml"}, {}])
def test_unsupported_devcontainers(bad):
    with pytest.raises(ValueError):
        dc.resolve(bad, devcontainer_rel=".devcontainer/devcontainer.json", repo_name="p",
                   host_workspace="/x", local_env={})


# ---- GPU process command lines ---------------------------------------------------------
def test_redact_cmd():
    from insjob.agent.gpu import redact_cmd
    assert redact_cmd("python a.py --token abc x=1") == "python a.py --token *** x=1"
    assert redact_cmd("env WANDB_API_KEY=k123 python t.py") == "env WANDB_API_KEY=*** python t.py"
    assert redact_cmd("t.py --db-password=hunter2") == "t.py --db-password=***"
    # values that merely contain a keyword are left alone
    assert redact_cmd("train.py exp_name=token_test lr=1e-3") == "train.py exp_name=token_test lr=1e-3"


def test_proc_info_reads_fake_proc(tmp_path):
    from insjob.agent.gpu import CMD_MAX, proc_info
    d = tmp_path / "4242"
    d.mkdir()
    (d / "status").write_text("Name:\tpython\nUid:\t1042\t1042\t1042\t1042\n")
    cid = "a" * 64
    (d / "cgroup").write_text(f"0::/system.slice/docker-{cid}.scope\n")
    (d / "cmdline").write_bytes(b"/opt/venv/bin/python\0src/train.py\0--api_key\0s3cr3t\0")
    uid, cont, cmd = proc_info(4242, proc=str(tmp_path))
    assert (uid, cont) == (1042, cid)
    assert cmd == "/opt/venv/bin/python src/train.py --api_key ***"
    (d / "cmdline").write_bytes(b"python\0" + b"x" * 5000)
    assert proc_info(4242, proc=str(tmp_path))[2].endswith(" …")
    assert len(proc_info(4242, proc=str(tmp_path))[2]) <= CMD_MAX + 2
    assert proc_info(999, proc=str(tmp_path)) == (None, None, None)


def test_short_cmd():
    from insjob.cli import short_cmd
    assert short_cmd("/opt/venv/bin/python3 -u src/train.py x=1") == "python3 -u src/train.py x=1"
    assert short_cmd("nvtop") == "nvtop"
    assert short_cmd(None) == ""

"""Release CI: run the GPU test suite on rented RunPod GPUs.

The suite only exercises the kernels native to the GPU it runs on, so the
generations no local machine has (Hopper wgmma, datacenter-Blackwell
tcgen05) are covered here. Run it from a checkout with the key injected:

    kw run -e RUNPOD_API_KEY=RUNPOD_API_KEY -- python3 ci/runpod_ci.py
    kw run -e RUNPOD_API_KEY=RUNPOD_API_KEY -- python3 ci/runpod_ci.py --only sm90
    kw run -e RUNPOD_API_KEY=RUNPOD_API_KEY -- python3 ci/runpod_ci.py --cleanup

Tests ``git archive HEAD`` (refuses a dirty tree unless ``--dirty``, which
ships the working tree's tracked files instead). Per target, in parallel:
create a pod (falling back across GPU variants and clouds), wait for SSH,
install uv, ship the tree, ``uv sync --extra cu13``, run pytest, pull the
log and JUnit XML into ci-results/<run>/<target>/, terminate the pod.

Cost guards: pods are named ``fa4-ci-*`` and nothing else is ever
terminated; termination runs in ``finally``; each pod also arms a dead-man
timer that kills its container after --max-minutes (an exited pod bills no
GPU time), so a crashed or disconnected runner can't leave a B200 running.
``--cleanup`` terminates any stray ``fa4-ci-*`` pods.

Stdlib only. Needs ssh, tar and git locally; the SSH public key matching
--ssh-key must be registered on the RunPod account.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

API = "https://api.runpod.io/v2"
IMAGE = "runpod/base:1.0.2-ubuntu2404"
PREFIX = "fa4-ci-"
# jax[cuda13] wheels bundle the CUDA userspace; the host driver must do 13.
MIN_CUDA = "13.0"


@dataclass(frozen=True)
class Target:
    name: str
    gpus: tuple[str, ...]  # RunPod GPU type ids, preferred first


TARGETS = {
    "sm90": Target("sm90", ("NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe")),
    "sm100": Target("sm100", ("NVIDIA B200",)),
    # Not in the release matrix (dantalion covers sm_86): a cheap target for
    # debugging this script itself.
    "sm86": Target("sm86", ("NVIDIA A40", "NVIDIA RTX A6000")),
}
DEFAULT_TARGETS = ("sm90", "sm100")


# ---------------------------------------------------------------- RunPod API


def api(method: str, path: str, body=None):
    req = urllib.request.Request(
        API + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Cloudflare in front of api.runpod.io bans urllib's default UA (error 1010).
            "User-Agent": "fa4-jax-ci/0.1 (+https://github.com/nshepperd/fa4-jax)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            return json.loads(data) if data else None
    except urllib.error.HTTPError as e:
        raise ApiError(e.code, f"{method} {path}: {e.read().decode()[:1000]}") from None


class ApiError(RuntimeError):
    def __init__(self, code: int, msg: str):
        super().__init__(f"HTTP {code} {msg}")
        self.code = code


def list_ci_pods():
    pods = api("GET", "/pods")
    pods = pods.get("pods", pods) if isinstance(pods, dict) else pods
    return [p for p in pods or [] if p["name"].startswith(PREFIX)]


def terminate(pod_id: str):
    pod = api("GET", f"/pods/{pod_id}")
    if not pod["name"].startswith(PREFIX):
        raise RuntimeError(f"refusing to terminate non-CI pod {pod['name']!r}")
    api("DELETE", f"/pods/{pod_id}")


# ---------------------------------------------------------------- one target


@dataclass
class Result:
    target: str
    gpu: str = "-"
    status: str = "not started"
    tests: dict = field(default_factory=dict)
    minutes: float = 0.0
    cost: float = 0.0
    note: str = ""


class Runner:
    def __init__(self, target: Target, run_id: str, tarball: bytes, args, out_dir: Path):
        self.t, self.run_id, self.tarball, self.args = target, run_id, tarball, args
        self.out = out_dir / target.name
        self.out.mkdir(parents=True, exist_ok=True)
        self.known_hosts = self.out / "known_hosts"
        self.res = Result(target.name)
        self.pod_id: str | None = None
        self.rate = 0.0

    def log(self, msg: str):
        print(f"[{self.t.name}] {msg}", flush=True)

    # -- pod lifecycle

    def create(self):
        name = f"{PREFIX}{self.run_id}-{self.t.name}"
        errors = []
        for attempt in range(self.args.placement_rounds):
            for cloud in ("SECURE", "COMMUNITY"):
                for gpu in self.t.gpus:
                    try:
                        pod = api("POST", "/pods", {
                            "name": name,
                            "image": IMAGE,
                            "cloud": cloud,
                            "gpu": {"id": gpu, "count": 1, "minCudaVersion": MIN_CUDA},
                            "disk": 60,
                            "ports": ["22/tcp"],
                            "startSsh": True,
                        })
                    except ApiError as e:
                        if e.code in (400, 409, 503):  # no capacity for this spec
                            errors.append(f"{gpu}/{cloud}: {e}")
                            continue
                        raise
                    self.pod_id, self.rate = pod["id"], float(pod.get("cost") or 0)
                    self.res.gpu = f"{gpu} ({cloud.lower()})"
                    self.log(f"created {name} id={self.pod_id} on {gpu} {cloud} ${self.rate}/h")
                    return
            self.log(f"no capacity (round {attempt + 1}/{self.args.placement_rounds}); retrying in 60s")
            time.sleep(60)
        raise RuntimeError("no capacity for any GPU variant: " + " | ".join(errors[-3:]))

    def wait_ssh(self):
        t0 = time.time()
        while time.time() - t0 < 900:
            pod = api("GET", f"/pods/{self.pod_id}")
            direct = (pod.get("ssh") or {}).get("direct")
            if pod["status"] in ("ERROR", "TERMINATED", "EXITED"):
                raise RuntimeError(f"pod went {pod['status']} before SSH came up")
            if pod["status"] == "RUNNING" and direct:
                self.direct = direct
                for _ in range(12):  # sshd can lag RUNNING
                    if self.ssh("true", check=False, quiet=True).returncode == 0:
                        self.log(f"ssh up after {time.time() - t0:.0f}s (cuda {pod.get('cudaVersion')}, "
                                 f"{pod.get('dataCenterId')})")
                        return
                    time.sleep(10)
                raise RuntimeError("ssh endpoint published but never accepted a connection")
            time.sleep(10)
        raise TimeoutError("pod never became reachable over SSH")

    def ssh(self, cmd: str, check=True, quiet=False, stdin=None, stdout=None, timeout=None):
        argv = [
            "ssh", "-i", self.args.ssh_key, "-p", str(self.direct["port"]),
            "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={self.known_hosts}", "-o", "LogLevel=ERROR",
            f"{self.direct['username']}@{self.direct['host']}", cmd,
        ]
        return subprocess.run(
            argv, check=check, input=stdin, timeout=timeout,
            stdout=stdout if stdout is not None else (subprocess.DEVNULL if quiet else None),
            stderr=subprocess.DEVNULL if quiet else None,
        )

    # -- the job

    def provision(self):
        # Dead-man switch: kill the container (pod -> EXITED, no GPU billing)
        # once the hard limit passes, whatever happens to this runner.
        secs = int(self.args.max_minutes * 60)
        self.ssh(f"nohup sh -c 'sleep {secs}; kill -TERM 1' >/dev/null 2>&1 &")
        self.ssh("mkdir -p /work/fa4-jax && tar -xzf - -C /work/fa4-jax", stdin=self.tarball)
        self.ssh(
            "set -e; command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null); "
            "curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; "
            "cd /work/fa4-jax && ~/.local/bin/uv sync -q --extra cu13 && "
            "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader",
            timeout=1800,
        )

    def test(self) -> int:
        pytest_args = " ".join(shlex.quote(a) for a in self.args.pytest_args)
        cmd = (
            "cd /work/fa4-jax && ~/.local/bin/uv run pytest tests -q -p no:cacheprovider "
            f"--junitxml=/work/junit.xml {pytest_args} > /work/pytest.log 2>&1; "
            "echo $? > /work/pytest.rc"
        )
        self.ssh(cmd, check=False, timeout=self.args.max_minutes * 60)
        for name in ("pytest.log", "junit.xml", "pytest.rc"):
            with open(self.out / name, "wb") as f:
                self.ssh(f"cat /work/{name}", check=False, stdout=f)
        rc = (self.out / "pytest.rc").read_text().strip()
        return int(rc) if rc.isdigit() else -1

    def summarize_junit(self):
        try:
            root = ET.parse(self.out / "junit.xml").getroot()
        except (ET.ParseError, FileNotFoundError):
            return
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        if suite is None:
            return
        n = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
        n["passed"] = n["tests"] - n["failures"] - n["errors"] - n["skipped"]
        self.res.tests = n

    def run(self):
        t0 = time.time()
        try:
            self.create()
            self.res.status = "provisioning"
            self.wait_ssh()
            self.provision()
            self.log("running tests")
            self.res.status = "testing"
            rc = self.test()
            self.summarize_junit()
            self.res.status = "PASS" if rc == 0 else f"FAIL (pytest rc={rc})"
            self.log(self.res.status)
        except Exception as e:  # noqa: BLE001 -- report every failure per target
            self.res.status = "ERROR"
            self.res.note = str(e)[:300]
            self.log(f"error: {e}")
        finally:
            if self.pod_id:
                try:
                    terminate(self.pod_id)
                    self.log(f"terminated {self.pod_id}")
                except Exception as e:  # noqa: BLE001
                    self.res.note += f" | TERMINATE FAILED for {self.pod_id}: {e} -- run --cleanup"
                    self.log(f"!!! failed to terminate {self.pod_id}: {e}")
            self.res.minutes = (time.time() - t0) / 60
            self.res.cost = self.rate * self.res.minutes / 60


# ---------------------------------------------------------------- driver


def source_tarball(dirty: bool) -> tuple[bytes, str]:
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    head = subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True)
    if not dirty:
        if status.strip():
            sys.exit("working tree is dirty; commit first (CI tests HEAD) or pass --dirty")
        data = subprocess.check_output(["git", "-C", root, "archive", "--format=tar.gz", "HEAD"])
        return data, head
    files = subprocess.check_output(["git", "-C", root, "ls-files", "-z"]).split(b"\0")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in filter(None, files):
            p = root / f.decode()
            if p.exists():
                tar.add(p, arcname=f.decode())
    return buf.getvalue(), head + ("-dirty" if status.strip() else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", choices=sorted(TARGETS), help="run just these targets")
    ap.add_argument("--dirty", action="store_true", help="ship the working tree instead of HEAD")
    ap.add_argument("--cleanup", action="store_true", help="terminate stray fa4-ci-* pods and exit")
    ap.add_argument("--max-minutes", type=float, default=150, help="per-pod hard limit (dead-man timer)")
    ap.add_argument("--placement-rounds", type=int, default=5, help="capacity retry rounds (60s apart)")
    ap.add_argument("--ssh-key", default=os.path.expanduser("~/.ssh/id_claude"))
    ap.add_argument("--out", default="ci-results")
    ap.add_argument("pytest_args", nargs="*", help="extra pytest args (after --)")
    args = ap.parse_args()
    if "RUNPOD_API_KEY" not in os.environ:
        sys.exit("RUNPOD_API_KEY not set; run under `kw run -e RUNPOD_API_KEY=RUNPOD_API_KEY --`")

    stray = list_ci_pods()
    if args.cleanup:
        for p in stray:
            terminate(p["id"])
            print(f"terminated {p['name']} ({p['id']}, {p['status']})")
        print(f"{len(stray)} CI pod(s) terminated")
        return
    if stray:
        print(f"warning: {len(stray)} stray CI pod(s) exist: "
              + ", ".join(f"{p['name']}/{p['status']}" for p in stray) + " (--cleanup)")

    tarball, rev = source_tarball(args.dirty)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / f"{run_id}-{rev}"
    targets = [TARGETS[t] for t in (args.only or DEFAULT_TARGETS)]
    print(f"run {run_id} @ {rev}: {', '.join(t.name for t in targets)} -> {out}")

    runners = [Runner(t, run_id, tarball, args, out) for t in targets]
    threads = [threading.Thread(target=r.run, name=r.t.name) for r in runners]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    print(f"\n{'target':7} {'status':22} {'passed':>6} {'failed':>6} {'skipped':>7} {'min':>5} {'$':>6}  gpu")
    for r in (x.res for x in runners):
        n = r.tests
        failed = n.get("failures", 0) + n.get("errors", 0) if n else "-"
        print(f"{r.target:7} {r.status:22} {n.get('passed', '-'):>6} {failed:>6} "
              f"{n.get('skipped', '-'):>7} {r.minutes:5.1f} {r.cost:6.2f}  {r.gpu}"
              + (f"\n        {r.note}" if r.note else ""))
    (out / "summary.json").write_text(json.dumps([r.res.__dict__ for r in runners], indent=1))
    sys.exit(0 if all(r.res.status == "PASS" for r in runners) else 1)


if __name__ == "__main__":
    main()

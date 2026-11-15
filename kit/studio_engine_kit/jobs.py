# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Allternit LLC
"""Job state machine, event log and process runner for Engine API v1.

States: queued -> preparing -> running -> exporting -> succeeded, and any
non-terminal state may go to failed or cancelled. One job at a time.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from .adapter import Adapter, Build, Step
from .fit import check_job
from .hardware import device_unavailable, from_env
from .spec import SpecError, Settings, build_settings, contract_errors, load_catalog

TERMINAL = ("succeeded", "failed", "cancelled")
_NEXT = {"queued": "preparing", "preparing": "running", "running": "exporting", "exporting": "succeeded"}

# What the dry-run stub process does instead of the real trainer: print a few
# progress lines (STUB_STEPS, STUB_DELAY seconds apart) and exit 0.
STUB = (
    "import os,sys,time\n"
    "n=int(os.environ.get('STUB_STEPS','3'));d=float(os.environ.get('STUB_DELAY','0'))\n"
    "for i in range(1,n+1):\n"
    "    time.sleep(d);print('step %d/%d loss=%.3f'%(i,n,2.0/(i+1)),flush=True)\n"
    "if os.environ.get('STUB_FAIL')==sys.argv[1]:\n"
    "    print('stub failure',file=sys.stderr,flush=True);sys.exit(3)\n"
)


class JobRunning(Exception):
    pass


class AlreadyFinished(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Job:
    def __init__(self, job_id: str, spec: Dict[str, Any], folder: Path) -> None:
        self.id = job_id
        self.spec = spec
        self.folder = folder
        self.state = "queued"
        self.created_at = _now()
        self.updated_at: Optional[str] = None
        self.error: Optional[str] = None
        self.last_seq = 0
        self.outputs: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        self.proc: Optional[subprocess.Popen] = None
        self.cancel_requested = False
        self.tail: Deque[str] = deque(maxlen=8)
        self.snapshot: Dict[str, Any] = {}

    def push(self, kind: str, message: str, progress: Optional[float] = None) -> None:
        with self.lock:
            self.last_seq += 1
            self.updated_at = _now()
            ev: Dict[str, Any] = {"seq": self.last_seq, "at": self.updated_at, "kind": kind, "message": message}
            if progress is not None:
                ev["progress"] = progress
            self.events.append(ev)
            try:
                with (self.folder / "events.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(ev) + "\n")
            except OSError:
                pass

    def advance(self, to: str, progress: Optional[float] = None) -> bool:
        """Move forward (or to failed/cancelled). False when the job already finished."""
        with self.lock:
            if self.state in TERMINAL:
                return False
            if to not in (_NEXT.get(self.state), "failed", "cancelled"):
                return False
            self.state = to
            self.push("state", to, progress)
            return True

    def fail(self, message: str) -> None:
        with self.lock:
            if self.state in TERMINAL:
                return
            self.error = message
            self.push("error", message)
            self.advance("failed")

    def to_json(self) -> Dict[str, Any]:
        with self.lock:
            return {"id": self.id, "state": self.state, "spec": self.spec, "created_at": self.created_at,
                    "updated_at": self.updated_at, "error": self.error, "last_seq": self.last_seq,
                    "outputs": list(self.outputs)}

    def events_after(self, after: int) -> Dict[str, Any]:
        with self.lock:
            return {"state": self.state, "events": [e for e in self.events if e["seq"] > after]}


# The trainer and its children form one group, so cancel stops all of them.
_GROUP: Dict[str, Any] = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                          else {"start_new_session": True})


class HardwareUnavailable(Exception):
    """The training device cannot be used right now. Becomes HTTP 503 hardware_unavailable."""


_FROM_ENV: Any = object()


class JobManager:
    """One engine's jobs.

    `hardware` is the id the host detected (default: ALLTERNIT_STUDIO_HARDWARE).
    `data_dir` is where job folders go (the host's ALLTERNIT_STUDIO_OUTPUT_DIR);
    it is made absolute, so every output path a job reports is absolute.
    `device_check(hardware)` returns why the device is unusable right now, or None.
    """

    def __init__(self, adapter: Adapter, data_dir: Path, dry_run: bool = False,
                 hardware: Optional[str] = _FROM_ENV, kill_grace: float = 5.0,
                 device_check: Callable[[Optional[str]], Optional[str]] = device_unavailable) -> None:
        self.adapter = adapter
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.dry_run = dry_run
        self.hardware: Optional[str] = from_env() if hardware is _FROM_ENV else hardware
        self.device_check = device_check
        self.kill_grace = kill_grace
        self.jobs: Dict[str, Job] = {}
        self._n = 0
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def submit(self, spec: Any) -> Job:
        """Check the spec and start the job. Raises SpecError (422),
        HardwareUnavailable (503) or JobRunning (409)."""
        errs = contract_errors(spec)
        if errs:
            raise SpecError(errs[0].code, "; ".join(str(e) for e in errs))
        if spec["engine"] != self.adapter.id:
            raise SpecError("unknown_engine", f"this engine is {self.adapter.id}, not '{spec['engine']}'")
        if self.hardware is None:
            raise HardwareUnavailable(str(self.device_check(None)))
        backend, refused = check_job(spec, load_catalog()[self.adapter.id], self.hardware)
        if refused:
            raise SpecError(refused[0].code, "; ".join(e.message for e in refused))
        if not self.dry_run:
            why = self.device_check(self.hardware)
            if why:
                raise HardwareUnavailable(why)
        settings = build_settings(spec, dry_run=self.dry_run, hardware=self.hardware, backend=backend)
        self.adapter.check_spec(settings)
        with self._lock:
            if any(j.state not in TERMINAL for j in self.jobs.values()):
                raise JobRunning("a job is already running on this engine")
            self._n += 1
            job_id = f"job_{self._n:04d}"
            folder = self.data_dir / "jobs" / job_id
            folder.mkdir(parents=True, exist_ok=True)
            job = Job(job_id, spec, folder)
            self.jobs[job_id] = job
        job.push("state", "queued", 0.0)
        (folder / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        job.snapshot = job.to_json()          # what POST /v1/jobs answers: state queued
        threading.Thread(target=self._run, args=(job, settings), daemon=True, name=job_id).start()
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        with job.lock:
            if job.state in TERMINAL:
                raise AlreadyFinished(f"job {job_id} is already {job.state}")
            job.cancel_requested = True
            job.advance("cancelled")
            proc = job.proc
        if proc is not None:
            self._kill(proc)
        return job

    def _kill(self, proc: subprocess.Popen) -> None:
        if os.name == "nt":       # no process groups: taskkill /T ends the whole tree
            if proc.poll() is None:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + self.kill_grace
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _run(self, job: Job, settings: Settings) -> None:
        try:
            self._run_inner(job, settings)
        except SpecError as exc:
            job.fail(f"{exc.code}: {exc.message}")
        except Exception as exc:  # noqa: BLE001 - a crashed worker must not hang the job
            job.fail(f"engine adapter crashed: {exc!r}")

    def _run_inner(self, job: Job, settings: Settings) -> None:
        if not job.advance("preparing"):
            return
        build = self.adapter.build(settings, job.folder)
        for rel, text in build.files.items():
            target = job.folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        (job.folder / "command.json").write_text(
            json.dumps([{"name": s.name, "argv": s.argv, "env": sorted(s.env)} for s in build.steps], indent=2),
            encoding="utf-8")
        for note in settings.notes + build.notes:
            job.push("log", note)
        if self.dry_run:
            job.push("log", "dry-run: config and command written; a stub process runs instead of the trainer")
        if not job.advance("running", 0.0):
            return
        deadline = time.time() + settings.max_hours * 3600 if settings.max_hours else None
        train, rest = build.steps[0], build.steps[1:]
        code = self._exec(job, train, deadline, report_progress=True)
        if code is None:
            return
        if code != 0:
            job.fail(f"{train.name} exited with code {code}: {self._tail(job)}")
            return
        if not job.advance("exporting"):
            return
        for step in rest:
            code = self._exec(job, step, deadline, report_progress=False)
            if code is None:
                return
            if code != 0:
                job.fail(f"{step.name} exited with code {code}: {self._tail(job)}")
                return
        self._register_outputs(job, build)
        job.advance("succeeded", 1.0)

    def _exec(self, job: Job, step: Step, deadline: Optional[float], report_progress: bool) -> Optional[int]:
        argv = [sys.executable, "-c", STUB, step.name] if self.dry_run else step.argv
        env = dict(os.environ)
        env.update(step.env)
        cwd = str(job.folder / step.cwd) if step.cwd else str(job.folder)
        job.push("log", f"{step.name}: {' '.join(step.argv)}")
        with job.lock:
            if job.state in TERMINAL:
                return None
            try:
                proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, **_GROUP)
            except OSError as exc:
                job.fail(f"cannot start {step.name}: {exc}. Is the engine installed in its virtualenv?")
                return None
            job.proc = proc
        timer: Optional[threading.Timer] = None
        timed_out = threading.Event()
        if deadline is not None:
            def _expire() -> None:
                timed_out.set()
                self._kill(proc)
            timer = threading.Timer(max(deadline - time.time(), 0.0), _expire)
            timer.daemon = True
            timer.start()
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            job.tail.append(line)
            p = self.adapter.parse_progress(line) if report_progress else None
            job.push("progress" if p is not None else "log", line[:2000], p)
        proc.wait()
        if timer:
            timer.cancel()
        with job.lock:
            job.proc = None
        if job.cancel_requested or job.state in TERMINAL:
            return None
        if timed_out.is_set():
            job.fail("stopped: resources.max_hours was reached")
            return None
        return proc.returncode

    @staticmethod
    def _tail(job: Job) -> str:
        return " | ".join(job.tail)[-600:] or "no output"

    def _register_outputs(self, job: Job, build: Build) -> None:
        for out in build.outputs:
            pattern = out.path
            if self.dry_run:  # the stub trained nothing, so write placeholder results
                target = job.folder / pattern.replace("*", "0")
                if target.suffix:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"dry-run placeholder\n")
                else:
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "dry-run.txt").write_text("dry-run placeholder\n")
            matches = sorted(job.folder.glob(pattern))
            if not matches:
                job.fail(f"expected {out.kind} output not found at {pattern}")
                return
            path = matches[-1]
            size = path.stat().st_size if path.is_file() else sum(
                f.stat().st_size for f in path.rglob("*") if f.is_file())
            record = {"kind": out.kind, "path": str(path.resolve()), "bytes": size}
            with job.lock:
                job.outputs.append(record)
            job.push("output", f"{out.kind} written: {record['path']}")

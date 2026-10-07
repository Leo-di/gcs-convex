"""Serve the contact-validation page and launch fresh Drake reruns.

The static validation page remains available under
``/contact_dynamics_validation/index.html``.  Two same-origin JSON endpoints
are added:

* ``POST /api/rerun`` with ``{"mode": "full"}`` or
  ``{"mode": "dynamics"}``;
* ``GET /api/status?id=<job id>``.

Only one rerun is allowed at a time.  Every run writes to a unique directory
under ``outputs/results/live_reruns`` so existing evidence is never
overwritten.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "results"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
DEFAULT_REFERENCE = RESULTS_ROOT / "report_ready_gcs_v3"
RERUN_ROOT = RESULTS_ROOT / "live_reruns"


@dataclass
class Job:
    id: str
    mode: str
    created_at: str
    output_root: Path
    status: str = "queued"
    stage: str = "queued"
    started_monotonic: float | None = None
    finished_monotonic: float | None = None
    message: str = "Waiting to start"
    scene_url: str | None = None
    planning_url: str | None = None
    result: dict[str, Any] | None = None
    return_code: int | None = None
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=24))

    def public(self) -> dict[str, Any]:
        now = self.finished_monotonic or time.monotonic()
        elapsed = (
            None
            if self.started_monotonic is None
            else round(now - self.started_monotonic, 2)
        )
        return {
            "id": self.id,
            "mode": self.mode,
            "created_at": self.created_at,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "elapsed_seconds": elapsed,
            "scene_url": self.scene_url,
            "planning_url": self.planning_url,
            "result": self.result,
            "return_code": self.return_code,
            "log_tail": list(self.log_tail),
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._active_id: str | None = None
        self._latest_id: str | None = None
        self._lock = threading.Lock()

    def start(self, mode: str) -> tuple[Job | None, Job | None]:
        if mode not in {"full", "dynamics"}:
            raise ValueError("mode must be 'full' or 'dynamics'")
        with self._lock:
            active = self._jobs.get(self._active_id or "")
            if active and active.status in {"queued", "running"}:
                return None, active
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            identifier = f"{stamp}-{uuid.uuid4().hex[:8]}"
            job = Job(
                id=identifier,
                mode=mode,
                created_at=datetime.now(timezone.utc).isoformat(),
                output_root=RERUN_ROOT / identifier,
            )
            self._jobs[job.id] = job
            self._active_id = job.id
            self._latest_id = job.id
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job, None

    def get(self, identifier: str | None) -> Job | None:
        with self._lock:
            key = identifier or self._active_id or self._latest_id
            return self._jobs.get(key or "")

    def _update(self, job: Job, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(job, key, value)

    def _command(self, job: Job, stage: str, command: list[str]) -> None:
        self._update(job, stage=stage, message=stage)
        job.output_root.mkdir(parents=True, exist_ok=True)
        log_path = job.output_root / "job.log"
        environment = os.environ.copy()
        environment.update(
            {
                "LCM_DEFAULT_URL": "memq://",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n$ " + " ".join(command) + "\n")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                clean = line.rstrip()
                log.write(line)
                log.flush()
                if clean:
                    with self._lock:
                        job.log_tail.append(clean)
                        job.message = clean[-300:]
            code = process.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, command)

    def _run(self, job: Job) -> None:
        self._update(
            job,
            status="running",
            stage="starting",
            started_monotonic=time.monotonic(),
            message="Preparing a fresh run",
        )
        try:
            reference = DEFAULT_REFERENCE
            if job.mode == "full":
                reference = job.output_root / "reference"
                self._command(
                    job,
                    "Planning semantic route, GCS and IRIS regions",
                    [
                        str(PYTHON),
                        "-m",
                        "src.animate_delivery_demo",
                        "--no-wait",
                        "--output-dir",
                        str(reference),
                    ],
                )
                job.planning_url = (
                    f"/live_reruns/{job.id}/reference/review_scene.html"
                )

            dynamics = job.output_root / "dynamics"
            self._command(
                job,
                "Running fresh Drake SAP contact dynamics",
                [
                    str(PYTHON),
                    "-m",
                    "src.execute_delivery_dynamics",
                    str(reference),
                    "--output-dir",
                    str(dynamics),
                    "--time-step",
                    "0.001",
                    "--time-scale",
                    "3.0",
                    "--html",
                ],
            )
            report = json.loads((dynamics / "result.json").read_text())
            compact = {
                "satisfied": bool(report["satisfied"]),
                "duration_s": report["duration_s"],
                "maximum_carry_slip_mm": round(
                    report["maximum_relative_translation_change_during_carry_m"]
                    * 1000,
                    3,
                ),
                "unexpected_contact_count": report["unexpected_contact_count"],
                "held_bilateral_contact_fraction": report[
                    "held_bilateral_contact_fraction"
                ],
                "cup_attachment": report["cup_attachment"],
                "cup_pose_resets_after_initialization": report[
                    "cup_pose_resets_after_initialization"
                ],
            }
            self._update(
                job,
                status="succeeded" if compact["satisfied"] else "failed",
                stage="complete",
                finished_monotonic=time.monotonic(),
                message=(
                    "Fresh run passed all gates"
                    if compact["satisfied"]
                    else "Fresh run completed but failed an acceptance gate"
                ),
                result=compact,
                scene_url=f"/live_reruns/{job.id}/dynamics/scene.html",
                return_code=0 if compact["satisfied"] else 1,
            )
        except Exception as error:  # Surface the subprocess failure to the UI.
            code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
            self._update(
                job,
                status="failed",
                stage="failed",
                finished_monotonic=time.monotonic(),
                message=str(error),
                return_code=code,
            )
        finally:
            with self._lock:
                if self._active_id == job.id:
                    self._active_id = None


MANAGER = JobManager()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(RESULTS_ROOT), **kwargs)

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            identifier = parse_qs(parsed.query).get("id", [None])[0]
            job = MANAGER.get(identifier)
            if job is None:
                self._json(HTTPStatus.OK, {"job": None})
            else:
                self._json(HTTPStatus.OK, {"job": job.public()})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path != "/api/rerun":
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            payload = json.loads(self.rfile.read(length) or b"{}")
            job, active = MANAGER.start(str(payload.get("mode", "")))
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        if active is not None:
            self._json(
                HTTPStatus.CONFLICT,
                {"error": "another rerun is active", "job": active.public()},
            )
            return
        assert job is not None
        self._json(HTTPStatus.ACCEPTED, {"job": job.public()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7002)
    args = parser.parse_args()
    RERUN_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Contact validation server: http://{args.host}:{args.port}/"
        "contact_dynamics_validation/index.html",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

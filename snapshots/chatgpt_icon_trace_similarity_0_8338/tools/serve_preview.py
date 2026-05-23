#!/usr/bin/env python3
"""Serve a browser-based 3D preview for the ChatGPT icon model.

The server intentionally uses only the Python standard library.  On startup it
runs ``models/chatgpt_icon.py`` to refresh exports, then watches that model
file's mtime and rebuilds whenever it changes.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


DEFAULT_PORT = 18763
PROJECT_ROOT = Path(__file__).absolute().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "chatgpt_icon.py"
VIEWER_DIR = PROJECT_ROOT / "viewer"
EXPORTS_DIR = PROJECT_ROOT / "exports"
SINGLE_STL = EXPORTS_DIR / "chatgpt_icon_single.stl"
BASE_STL = EXPORTS_DIR / "chatgpt_icon_base.stl"
LOGO_STL = EXPORTS_DIR / "chatgpt_icon_logo.stl"
STEP_FILE = EXPORTS_DIR / "chatgpt_icon.step"
PREVIEW_PNG = EXPORTS_DIR / "preview.png"
SIMILARITY_REPORT = EXPORTS_DIR / "similarity_report.json"
REFERENCE_IMAGE = PROJECT_ROOT / "reference" / "chatgpt_reference.png"
POLL_INTERVAL_SECONDS = 1.0


@dataclass
class BuildStatus:
    ok: bool = False
    version: int = 0
    last_build_time: str | None = None
    command: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    exports: dict[str, bool] = field(default_factory=dict)
    similarity: dict[str, Any] = field(default_factory=dict)


class PreviewState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = BuildStatus(command=self._build_command())
        self._last_seen_mtime: float | None = None
        self._building = False

    def _build_command(self) -> list[str]:
        return [sys.executable, str(MODEL_PATH)]

    def _run_similarity_check(self) -> dict[str, Any]:
        if not REFERENCE_IMAGE.exists():
            return {
                "available": False,
                "ok": False,
                "reason": f"reference image missing: {REFERENCE_IMAGE}",
                "reference": str(REFERENCE_IMAGE),
            }

        command = [sys.executable, str(PROJECT_ROOT / "tools" / "compare_top_view.py")]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            report = json.loads(SIMILARITY_REPORT.read_text(encoding="utf-8"))
        except Exception:
            report = {}
        report.update(
            {
                "available": True,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        return report

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            status = asdict(self._status)
            status["building"] = self._building
            return status

    def run_build(self) -> None:
        command = self._build_command()
        with self._lock:
            if self._building:
                return
            self._building = True

        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            ok = completed.returncode == 0
            exports = self._exports_snapshot()
            similarity = self._run_similarity_check() if ok else {}
            with self._lock:
                next_version = self._status.version + 1 if ok else self._status.version
                self._status = BuildStatus(
                    ok=ok,
                    version=next_version,
                    last_build_time=datetime.now(timezone.utc).isoformat(),
                    command=command,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    returncode=completed.returncode,
                    exports=exports,
                    similarity=similarity,
                )
        except Exception as exc:  # pragma: no cover - defensive server boundary
            with self._lock:
                self._status = BuildStatus(
                    ok=False,
                    version=self._status.version,
                    last_build_time=datetime.now(timezone.utc).isoformat(),
                    command=command,
                    stdout="",
                    stderr=f"ERROR: preview build failed: {exc}",
                    returncode=None,
                    exports=self._exports_snapshot(),
                    similarity={},
                )
        finally:
            try:
                self._last_seen_mtime = MODEL_PATH.stat().st_mtime
            except FileNotFoundError:
                self._last_seen_mtime = None
            with self._lock:
                self._building = False

    def watch_model(self) -> None:
        while True:
            try:
                current_mtime = MODEL_PATH.stat().st_mtime
            except FileNotFoundError:
                current_mtime = None

            if self._last_seen_mtime is None:
                self._last_seen_mtime = current_mtime
            elif current_mtime is not None and current_mtime != self._last_seen_mtime:
                self.run_build()

            time.sleep(POLL_INTERVAL_SECONDS)

    def _exports_snapshot(self) -> dict[str, bool]:
        return {
            "directory": EXPORTS_DIR.exists(),
            "chatgpt_icon_single.stl": SINGLE_STL.exists(),
            "chatgpt_icon_base.stl": BASE_STL.exists(),
            "chatgpt_icon_logo.stl": LOGO_STL.exists(),
            "chatgpt_icon.step": STEP_FILE.exists(),
            "preview.png": PREVIEW_PNG.exists(),
            "similarity_report.json": SIMILARITY_REPORT.exists(),
        }


def _safe_static_path(url_path: str) -> Path | None:
    if url_path in {"", "/"}:
        relative = Path("viewer/index.html")
    else:
        clean_path = unquote(url_path.lstrip("/"))
        relative = Path(clean_path)

    candidate = (PROJECT_ROOT / relative).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        return None

    if candidate.is_dir():
        candidate = candidate / "index.html"
    return candidate


def make_handler(state: PreviewState) -> type[BaseHTTPRequestHandler]:
    class PreviewHandler(BaseHTTPRequestHandler):
        server_version = "ChatGPTIconPreview/1.0"

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                self._send_json(state.snapshot())
                return

            static_path = _safe_static_path(parsed.path)
            if static_path is None or not static_path.is_file():
                self.send_error(404, "Not found")
                return

            content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
            if static_path.suffix.lower() == ".stl":
                content_type = "model/stl"

            data = static_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}")

        def _send_json(self, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return PreviewHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the ChatGPT icon 3D browser preview.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = PreviewState()

    print("Running initial export build...")
    state.run_build()
    initial = state.snapshot()
    if not initial["ok"]:
        print("Initial build failed; the viewer will show the captured stderr/stdout.", file=sys.stderr)

    watcher = threading.Thread(target=state.watch_model, name="chatgpt-icon-watch", daemon=True)
    watcher.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving preview at {url}")
    print(f"Watching {MODEL_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping preview server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

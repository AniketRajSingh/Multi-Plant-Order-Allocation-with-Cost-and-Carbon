#!/usr/bin/env python3
"""Unified execution entrypoint for Multi-Plant Order Allocation platform.

Usage:
------
  # Run both FastAPI backend (port 8000) and Streamlit UI (port 8501)
  python run.py

  # Run only the FastAPI backend
  python run.py --api

  # Run only the Streamlit frontend UI
  python run.py --ui

  # Run full verification test suite
  python run.py --test

  # Run Section 4.5 dataset validation checks
  python run.py --validate
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import (  # noqa: E402
    get_config,
    get_db_path,
    get_llm_model,
    get_llm_provider,
    get_mock_llm,
    get_ollama_base_url,
    get_planning_start,
)


def print_banner() -> None:
    cfg = get_config()
    model = get_llm_model() or "(not set in config)"
    provider = get_llm_provider()
    url = get_ollama_base_url()
    mock = get_mock_llm()
    db = get_db_path()

    print("=" * 72)
    print("  🚀 Multi-Plant Order Allocation with Cost & Carbon")
    print("  HCLTech Future Ready AI Engineer Hackathon Platform")
    print("=" * 72)
    print(f"  • Database:        {db}")
    print(f"  • LLM Provider:    {provider}")
    print(f"  • LLM Model:       {model}")
    print(f"  • LLM Endpoint:    {url}")
    print(f"  • Mock LLM Mode:   {mock}")
    print("=" * 72)


def run_api(host: str = "0.0.0.0", port: int = 8000) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"[*] Starting FastAPI backend on http://localhost:{port} (Swagger: http://localhost:{port}/docs)")
    return subprocess.Popen(cmd, cwd=str(ROOT))


def run_ui(port: int = 8501) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/app.py",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    print(f"[*] Starting Streamlit UI on http://localhost:{port}")
    return subprocess.Popen(cmd, cwd=str(ROOT))


def run_tests() -> int:
    print("[*] Running comprehensive test suite (pytest)...")
    code = subprocess.call([sys.executable, "-m", "pytest", "tests/"], cwd=str(ROOT))
    if code != 0:
        return code
    print("\n[*] Running dataset validation checks...")
    return subprocess.call(
        [sys.executable, "scripts/load_data.py", "--check-only", "--planning-start", get_planning_start()],
        cwd=str(ROOT),
    )


def run_validate() -> int:
    print("[*] Running dataset validation...")
    code = subprocess.call(
        [sys.executable, "scripts/validate_data.py", "--dir", "data/sample"],
        cwd=str(ROOT),
    )
    if code != 0:
        return code
    return subprocess.call(
        [sys.executable, "scripts/load_data.py", "--check-only", "--planning-start", get_planning_start()],
        cwd=str(ROOT),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified runner for the Multi-Plant Order Allocation platform"
    )
    parser.add_argument("--api", action="store_true", help="Run only the FastAPI backend")
    parser.add_argument("--ui", action="store_true", help="Run only the Streamlit UI")
    parser.add_argument("--test", action="store_true", help="Run full test suite")
    parser.add_argument("--validate", action="store_true", help="Run data validation checks")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--ui-port", type=int, default=8501, help="UI port (default: 8501)")
    parser.add_argument("--host", default="0.0.0.0", help="API host (default: 0.0.0.0)")

    args = parser.parse_args()
    print_banner()

    if args.test:
        return run_tests()

    if args.validate:
        return run_validate()

    procs: list[subprocess.Popen] = []

    def cleanup(signum=None, frame=None):
        print("\n[*] Shutting down servers gracefully...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        if args.api:
            p_api = run_api(host=args.host, port=args.port)
            procs.append(p_api)
        elif args.ui:
            p_ui = run_ui(port=args.ui_port)
            procs.append(p_ui)
        else:
            # Default: run both API and UI
            p_api = run_api(host=args.host, port=args.port)
            procs.append(p_api)
            time.sleep(1)
            p_ui = run_ui(port=args.ui_port)
            procs.append(p_ui)

        print("\n✨ Ready! Access points:")
        if not args.ui:
            print(f"  • API & Docs:  http://localhost:{args.port}/docs")
            print(f"  • API Health:  http://localhost:{args.port}/health")
        if not args.api:
            print(f"  • Frontend UI: http://localhost:{args.ui_port}")
        print("\nPress Ctrl+C to stop all servers.\n")

        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"[!] A server process exited with code {p.returncode}")
                    cleanup()
            time.sleep(0.5)

    except KeyboardInterrupt:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())

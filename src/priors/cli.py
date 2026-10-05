"""Command-line entry point.

    priors app                    start the Streamlit app
    priors refresh                download stock data and write a snapshot (run daily, off-hours)
    priors sharadar-download      download Sharadar with your own key, to this computer only
    priors proxy                  start the classroom proxy (needs ANTHROPIC_API_KEY and a classroom.yaml)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="priors", description="Priors: evidence-based factor investing, for teaching.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("app", help="start the Streamlit app")
    a.add_argument("--port", type=int, default=8501)

    r = sub.add_parser("refresh", help="download stock data and write a snapshot")
    r.add_argument("--universe", choices=["russell3000", "russell1000"], default="russell3000")
    r.add_argument("--start", default="1995-01-01")
    r.add_argument("--no-fundamentals", action="store_true", help="prices only (minutes instead of half an hour)")

    d = sub.add_parser("sharadar-download", help="download Sharadar with your own Nasdaq Data Link key, to this computer")
    d.add_argument("--key", help="Nasdaq Data Link API key (or set NASDAQ_DATA_LINK_API_KEY)")

    x = sub.add_parser("proxy", help="start the classroom proxy")
    x.add_argument("--host", default="127.0.0.1")
    x.add_argument("--port", type=int, default=8787)

    args = p.parse_args(argv)
    if args.cmd == "app":
        script = Path(__file__).parent / "app" / "streamlit_app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(args.port)])
    if args.cmd == "refresh":
        from .stockdata import refresh_snapshot

        refresh_snapshot(args.universe, args.start, fundamentals=not args.no_fundamentals)
        return 0
    if args.cmd == "sharadar-download":
        import os

        from dotenv import load_dotenv

        from .data.sharadar_download import download_sharadar

        load_dotenv(Path.cwd() / ".env")   # NASDAQ_DATA_LINK_API_KEY may live in .env (never committed)

        paths = download_sharadar(args.key or os.environ.get("NASDAQ_DATA_LINK_API_KEY", ""))
        print("Sharadar is now available as a data source in the app on this computer:")
        for k, v in paths.items():
            print(f"  {k} = {v}")
        return 0
    if args.cmd == "proxy":
        import uvicorn
        from dotenv import load_dotenv

        from .proxy import app_from_env

        load_dotenv()
        uvicorn.run(app_from_env(), host=args.host, port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Agent entrypoint with no-op loop."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path

from .devux import start_http_server
from .llm import create_llm
from .problems import monitor


def main() -> None:
    """Start the agent and run an idle loop."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    buffer_size = os.environ.get("BUFFER_SIZE", "100")
    rate = float(os.environ.get("ANALYSIS_RATE_SECONDS", "0"))
    max_lines = int(os.environ.get("ANALYSIS_MAX_LINES", "2000"))
    token = os.environ.get("SUPERVISOR_TOKEN", "")
    problem_dir = Path("/data/problems")
    ws_url = "ws://supervisor/core/websocket"
    logging.info(
        "Agent starting (log level: %s, buffer size: %s, ws: %s)",
        log_level,
        buffer_size,
        ws_url,
    )
    Path("/tmp/healthy").touch()
    llm = create_llm()
    server = start_http_server(problem_dir, llm=llm)

    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        logging.info("Received shutdown signal, stopping agent...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal)

    async def _run() -> None:
        await monitor(
            ws_url,
            token,
            problem_dir,
            llm=llm,
            analysis_rate_seconds=rate,
            analysis_max_lines=max_lines,
            stop_event=stop_event,
        )
        server.shutdown()

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover - manual entry
    main()

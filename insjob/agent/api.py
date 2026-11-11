"""Dashboard + JSON API, served by every agent on 127.0.0.1:<user port> (reach it via SSH)."""

from __future__ import annotations

import asyncio
import hmac
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from .scheduler import Agent, tail
from .views import overview

STATIC = Path(__file__).parent / "static"


def make_app(agent: Agent, token: str) -> FastAPI:
    app = FastAPI(title="insjob", docs_url=None, redoc_url=None)

    def auth(request: Request) -> None:
        got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        got = got or request.query_params.get("token", "")
        if not hmac.compare_digest(got, token):
            raise HTTPException(401, "bad token")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/overview", dependencies=[Depends(auth)])
    def api_overview():
        d = overview(agent.state, agent.lab_status, agent.ident, agent.policy, agent.lab)
        d["served_by"] = agent.host
        return d

    @app.get("/api/events", dependencies=[Depends(auth)])
    def api_events(since: float = 0):
        return agent.state.events(since)

    @app.get("/api/jobs/{jid}/log", dependencies=[Depends(auth)],
             response_class=PlainTextResponse)
    def api_log(jid: int, lines: int = 300):
        return "\n".join(tail(agent.state.run_dir(jid) / "log", lines))

    @app.get("/api/jobs/{jid}/stream", dependencies=[Depends(auth)])
    async def api_stream(jid: int):
        path = agent.state.run_dir(jid) / "log"

        async def gen():
            pos = max(0, path.stat().st_size - 32 * 1024) if path.exists() else 0
            idle = 0
            while idle < 3600:
                if path.exists() and path.stat().st_size > pos:
                    with open(path, "rb") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    for line in chunk.decode(errors="replace").splitlines():
                        yield f"data: {line}\n\n"
                    idle = 0
                else:
                    idle += 1
                    yield ": ping\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/jobs/{jid}/stop", dependencies=[Depends(auth)])
    def api_stop(jid: int):
        return {"result": agent.state.request_cancel(jid)}

    return app

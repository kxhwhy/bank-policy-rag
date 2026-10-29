# -*- coding: utf-8 -*-
"""FastAPI 服务。

接口：
    POST /api/ask          非流式问答，返回答案 + 依据条款
    POST /api/ask/stream   SSE 流式问答
    GET  /api/overview     知识库概况
    GET  /api/documents    制度清单（含效力状态，便于核对旧制度是否已废止）
    POST /api/reset        清空会话上下文

前端页面挂在 /，直接访问 http://127.0.0.1:8000 即可。
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.config import config
from core.logger import logger
from core.qa_system import qa_system
from core.store.policy_db import db

app = FastAPI(
    title="徽商银行制度智能问答 API",
    description="基于混合检索与条款溯源的制度问答服务",
    version="1.0.0",
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    business_line: str | None = Field(None, description="业务条线，可选")
    session_id: str | None = Field(None, description="会话 ID，用于多轮指代补全")


class ResetRequest(BaseModel):
    session_id: str


@app.get("/")
async def index():
    page = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(page):
        raise HTTPException(status_code=404, detail="页面未部署")
    return FileResponse(page)


@app.get("/api/overview")
async def overview():
    return qa_system.overview()


@app.get("/api/documents")
async def documents(status: str | None = None):
    docs = db.list_documents(status=status)
    return {
        "total": len(docs),
        "items": [
            {
                "doc_no": d["doc_no"],
                "title": d["title"],
                "status": d["status"],
                "business_line": d["business_line"],
                "effective_date": d["effective_date"].isoformat() if d["effective_date"] else None,
                "repeal_date": d["repeal_date"].isoformat() if d["repeal_date"] else None,
                "supersedes": d["supersedes"],
            }
            for d in docs
        ],
    }


@app.post("/api/ask")
async def ask(req: AskRequest):
    if req.business_line and req.business_line not in config.BUSINESS_LINES:
        raise HTTPException(status_code=400, detail=f"未知业务条线：{req.business_line}")
    try:
        return qa_system.ask(req.question, business_line=req.business_line, session_id=req.session_id)
    except Exception as exc:
        logger.exception("问答处理异常")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    if req.business_line and req.business_line not in config.BUSINESS_LINES:
        raise HTTPException(status_code=400, detail=f"未知业务条线：{req.business_line}")

    def event_source():
        try:
            for event in qa_system.ask_stream(
                req.question, business_line=req.business_line, session_id=req.session_id
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("流式问答异常")
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/api/reset")
async def reset(req: ResetRequest):
    qa_system.reset_session(req.session_id)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

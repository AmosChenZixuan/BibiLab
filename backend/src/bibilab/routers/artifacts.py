import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from bibilab.config import BibilabConfig, bibilab_home, get_config
from bibilab.db.artifacts import (
    delete_artifact,
    get_artifact,
    get_artifacts_for_list,
    update_artifact_name,
)
from bibilab.db.jobs import create_job
from bibilab.db.lists import get_list
from bibilab.models.artifacts import (
    ArtifactCreateRequest,
    ArtifactPatchRequest,
    ArtifactResponse,
)
from bibilab.models.jobs import JobResponse, JobStatus
from bibilab.routers._model_gate import require_models_present

router = APIRouter()


@router.get("/lists/{list_id}/artifacts")
async def list_artifacts(list_id: str) -> list[ArtifactResponse]:
    row = await get_list(list_id)
    if row is None:
        raise HTTPException(status_code=404, detail="List not found")
    rows = await get_artifacts_for_list(list_id)
    return [ArtifactResponse.from_row(dict(r)) for r in rows]


@router.post("/lists/{list_id}/artifacts", status_code=201)
async def create_artifact_endpoint(
    list_id: str,
    req: ArtifactCreateRequest,
    request: Request,
    cfg: BibilabConfig = Depends(get_config),
) -> JobResponse:
    row = await get_list(list_id)
    if row is None:
        raise HTTPException(status_code=404, detail="List not found")

    require_models_present(cfg)

    artifact_id = str(uuid.uuid4())

    ui_lang_header = request.headers.get("X-UI-Lang", "en")
    output_lang = cfg.ai.output_language
    resolved_lang = ui_lang_header if output_lang == "ui" else output_lang

    # Queue a job for artifact generation (worker will create artifact record on success)
    job_id = await create_job(
        type="artifact",
        meta={
            "artifact_id": artifact_id,
            "list_id": list_id,
            "type": req.type,
            "prompt": req.prompt,
            "source_ids": req.source_ids,
            "ui_lang": resolved_lang,
        },
    )

    now = datetime.now(timezone.utc).isoformat()
    return JobResponse(
        id=job_id,
        type="artifact",
        status=JobStatus.QUEUED,
        progress=0,
        error=None,
        created_at=now,
        updated_at=now,
        meta={
            "artifact_id": artifact_id,
            "list_id": list_id,
            "type": req.type,
            "prompt": req.prompt,
            "source_ids": req.source_ids,
        },
    )


@router.get("/artifacts/{artifact_id}")
async def get_artifact_metadata(artifact_id: str) -> ArtifactResponse:
    """Single-artifact metadata. Unused by the web SPA (the list response carries
    all fields); kept for REST consistency with the by-id PATCH/DELETE/content routes."""
    row = await get_artifact(artifact_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return ArtifactResponse.from_row(dict(row))


@router.get("/artifacts/{artifact_id}/content")
async def get_artifact_content(artifact_id: str) -> FileResponse:
    row = await get_artifact(artifact_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    content_path_str = row["content_path"]
    if content_path_str is None:
        raise HTTPException(status_code=404, detail="Content not available yet")

    content_path = bibilab_home() / content_path_str
    if not content_path.exists():
        raise HTTPException(status_code=404, detail="Content file not found")

    return FileResponse(content_path, media_type="text/markdown")


@router.patch("/artifacts/{artifact_id}")
async def update_artifact(
    artifact_id: str,
    req: ArtifactPatchRequest,
) -> ArtifactResponse:
    row = await get_artifact(artifact_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if not req.model_fields_set:
        raise HTTPException(status_code=422, detail="No fields to update")

    if "name" in req.model_fields_set:
        if req.name is None or not req.name.strip():
            raise HTTPException(status_code=422, detail="Name cannot be empty")
        await update_artifact_name(artifact_id, req.name.strip())

    next_row = await get_artifact(artifact_id)
    if next_row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return ArtifactResponse.from_row(dict(next_row))


@router.delete("/artifacts/{artifact_id}", status_code=204)
async def delete_artifact_endpoint(artifact_id: str) -> None:
    row = await get_artifact(artifact_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    content_path_str = row["content_path"]
    if content_path_str:
        content_path = bibilab_home() / content_path_str
        content_path.unlink(missing_ok=True)

    await delete_artifact(artifact_id)

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from bibilab.config import bibilab_home
from bibilab.db import (
    create_artifact,
    create_job,
    delete_artifact,
    get_artifact,
    get_artifacts_for_list,
    get_list,
    update_artifact_name,
)
from bibilab.models.artifacts import (
    ArtifactCreateRequest,
    ArtifactPatchRequest,
    ArtifactResponse,
)

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
) -> ArtifactResponse:
    row = await get_list(list_id)
    if row is None:
        raise HTTPException(status_code=404, detail="List not found")

    artifact_id = str(uuid.uuid4())

    # Queue a job for artifact generation
    await create_job(
        type="artifact",
        meta={
            "artifact_id": artifact_id,
            "list_id": list_id,
            "type": req.type,
            "prompt": req.prompt,
            "source_ids": req.source_ids,
        },
    )

    # Create artifact record immediately (worker will update it later)
    await create_artifact(
        artifact_id=artifact_id,
        list_id=list_id,
        name=None,
        type=req.type,
        prompt=req.prompt,
        source_ids=req.source_ids,
        status="generating",
        content_path=None,
    )

    return ArtifactResponse(
        id=artifact_id,
        list_id=list_id,
        name=None,
        type=req.type,
        prompt=req.prompt,
        source_ids=req.source_ids,
        status="generating",
        content_path=None,
        error=None,
        created_at=datetime.now(timezone.utc),
    )


@router.get("/artifacts/{artifact_id}")
async def get_artifact_metadata(artifact_id: str) -> ArtifactResponse:
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

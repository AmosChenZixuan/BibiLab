import asyncio
from typing import Literal

from bibilab.db import get_jobs_for_video_ids, get_source_video_ids
from bibilab.models.jobs import JobStatus


def derive_video_statuses(
    video_ids: list[str],
    jobs: list[dict[str, str]],
    processed_ids: set[str],
) -> dict[str, Literal["new", "processed", "in_progress", "needs_auth"]]:
    needs_auth_videos: set[str] = set()
    in_progress_videos: set[str] = set()

    for row in jobs:
        vid = row["video_id"]
        st = row["status"]
        if st == JobStatus.NEEDS_AUTH.value:
            needs_auth_videos.add(vid)
        elif st in (
            JobStatus.QUEUED.value,
            JobStatus.DOWNLOADING.value,
            JobStatus.TRANSCRIBING.value,
            JobStatus.PROCESSING.value,
        ):
            in_progress_videos.add(vid)

    statuses: dict[str, Literal["new", "processed", "in_progress", "needs_auth"]] = {}
    for vid in video_ids:
        if vid in needs_auth_videos:
            statuses[vid] = "needs_auth"
        elif vid in in_progress_videos:
            statuses[vid] = "in_progress"
        elif vid in processed_ids:
            statuses[vid] = "processed"
        else:
            statuses[vid] = "new"

    return statuses


async def get_video_statuses(
    video_ids: list[str], list_id: str
) -> dict[str, Literal["new", "processed", "in_progress", "needs_auth"]]:
    jobs, processed_ids = await asyncio.gather(
        get_jobs_for_video_ids(video_ids, list_id),
        get_source_video_ids(video_ids, list_id),
    )
    return derive_video_statuses(video_ids, jobs, processed_ids)

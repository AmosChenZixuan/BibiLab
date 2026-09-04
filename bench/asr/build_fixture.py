"""Build the frozen ASR fixture: audio + reference transcripts, as a tarball.

Runs once, on the machine that holds the library. The pipeline deletes its
audio after transcription, so the wavs have to be re-fetched; this reuses the
production adapter and extract_audio so the result is byte-identical to what
the pipeline would have handed the model.

The tarball is the portable unit -- copy it to any other machine and bench.py
runs there against identical input. Re-downloading on the far machine would
not give the same bytes.

    cd backend && uv run python ../bench/asr/build_fixture.py
"""

import hashlib
import json
import shutil
import sqlite3
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "src"))

from bibilab.adapters import get_adapter_for_platform
from bibilab.config import load_config
from bibilab.pipeline.audio import extract_audio

HERE = Path(__file__).parent
OUT = HERE / "fixture"
DB = Path.home() / ".bibilab" / "bibilab.db"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    picks = json.loads((HERE / "fixture.json").read_text())["sources"]
    OUT.mkdir(exist_ok=True)
    cfg = load_config()

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    manifest = []

    for pick in picks:
        prefix, video_id = pick["source_id"], pick["video_id"]
        row = con.execute(
            "SELECT id, platform, source_url, duration_seconds, title, whisper_model, language "
            "FROM sources WHERE id LIKE ?",
            (prefix + "%",),
        ).fetchone()
        if row is None:
            raise SystemExit(f"source {prefix} not in {DB}")

        wav = OUT / f"{prefix}.wav"
        if not wav.exists():
            print(f"[{prefix}] downloading {video_id} ...", flush=True)
            adapter = get_adapter_for_platform(row["platform"], cfg)
            media = adapter.download(video_id, row["source_url"], connections=4)
            shutil.move(extract_audio(media, row["duration_seconds"]), wav)

        segments = [
            {"seq": s["seq"], "start_s": s["start_s"], "end_s": s["end_s"],
             "speaker": s["speaker"], "text": s["text"]}
            for s in con.execute(
                "SELECT seq, start_s, end_s, speaker, text FROM transcript_segments "
                "WHERE source_id = ? ORDER BY seq",
                (row["id"],),
            )
        ]
        (OUT / f"{prefix}.reference.json").write_text(
            json.dumps(
                {"source_id": row["id"], "video_id": video_id, "title": row["title"],
                 "engine": row["whisper_model"], "language": row["language"],
                 "duration_s": row["duration_seconds"], "segments": segments},
                ensure_ascii=False, indent=1,
            )
        )

        manifest.append({
            "source_id": prefix, "video_id": video_id, "title": row["title"],
            "duration_s": row["duration_seconds"], "wav_sha256": sha256(wav),
            "wav_bytes": wav.stat().st_size, "segments": len(segments),
            "speakers": len({s["speaker"] for s in segments}),
            "why": pick["why"],
        })
        print(f"[{prefix}] {row['duration_seconds']}s {len(segments)} segs "
              f"-> {wav.name}", flush=True)

    (OUT / "manifest.json").write_text(
        json.dumps({"reference_engine": "funasr sensevoice-small + fsmn-vad + cam++ + ct-punc",
                    "sources": manifest}, ensure_ascii=False, indent=1)
    )

    tar = HERE / "fixture.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(OUT, arcname="fixture")
    total = sum(m["duration_s"] for m in manifest)
    print(f"\n{len(manifest)} sources, {total / 60:.1f} min audio")
    print(f"{tar} {tar.stat().st_size / 2**20:.0f} MB  sha256={sha256(tar)}")


if __name__ == "__main__":
    main()

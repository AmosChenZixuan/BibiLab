"""Frozen ASR benchmark for #679: sherpa-onnx vs the current FunASR path.

One command, same on Linux and macOS, JSON out. Runs the frozen fixture through
one engine/provider/concurrency configuration, then scores the transcript
against the reference shipped in the fixture.

    python bench.py --fetch-models
    python bench.py --engine sherpa --provider cpu --threads 4 --concurrency 1
    python bench.py --engine funasr --device cpu --concurrency 2

The reference is the library's existing transcript, produced by FunASR +
FSMN-VAD + CAM++ + ct-punc. It is a comparison baseline, NOT ground truth:
`cer` measures how far sherpa-onnx drifts from what we ship today, and says
nothing about which one is right. `--dump-divergent` writes the worst windows
side by side so a human can adjudicate the disagreements.

Requires (sherpa): sherpa-onnx==1.13.7, numpy, soundfile.
Requires (funasr): the backend env -- run it with `uv run` from backend/.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).parent
FIXTURE = HERE / "fixture"
MODELS = HERE / "models"

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
ASSETS = {
    "silero_vad.onnx": f"{RELEASE}/asr-models/silero_vad.onnx",
    "campplus_zh.onnx": f"{RELEASE}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2": f"{RELEASE}/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
    "sherpa-onnx-whisper-large-v3.tar.bz2": f"{RELEASE}/asr-models/sherpa-onnx-whisper-large-v3.tar.bz2",
    "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2": f"{RELEASE}/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2",
}

# Mirrors pipeline/punctuate.py's _PUNC. Duplicated so this script runs without
# the backend installed; both sides get stripped before scoring, so a drift here
# costs nothing as long as the set stays a superset of what either path emits.
PUNC = frozenset("。，、！？；：．,.!?;:…　 \n\t")

# Matches transcribe.py's vad_kwargs: max_single_segment_time 15000ms,
# max_end_silence_time 500ms.
VAD_MAX_SPEECH = 15.0
VAD_MIN_SILENCE = 0.5


# --------------------------------------------------------------------------- io


def peak_rss_gb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KiB on Linux, bytes on macOS.
    return kb / 2**20 if sys.platform == "linux" else kb / 2**30


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def fetch_models() -> None:
    MODELS.mkdir(exist_ok=True)
    digests = {}
    for name, url in ASSETS.items():
        dest = MODELS / name
        if not dest.exists():
            print(f"fetching {name} ...", flush=True)
            urllib.request.urlretrieve(url, dest)
        digests[name] = sha256(dest.read_bytes()).hexdigest()
        if name.endswith(".tar.bz2") and not (MODELS / name[: -len(".tar.bz2")]).exists():
            with tarfile.open(dest) as tf:
                tf.extractall(MODELS)
        print(f"  {name}  {digests[name][:16]}")

    lock = MODELS / "models.lock.json"
    if lock.exists():
        old = json.loads(lock.read_text())
        drift = {k for k, v in digests.items() if old.get(k) not in (None, v)}
        if drift:
            raise SystemExit(f"model digest drift, results not comparable: {sorted(drift)}")
    lock.write_text(json.dumps(digests, indent=1))
    print(f"\npinned in {lock}")


def read_wav(path: Path):
    import soundfile as sf

    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    assert rate == 16000, f"{path}: expected 16 kHz, got {rate}"
    return samples, rate


# ---------------------------------------------------------------------- engines


class Sherpa:
    """VAD -> per-segment ASR -> per-segment speaker embedding -> cluster -> punct.

    Same stage order as the FunASR path, so the transcripts are comparable.
    Providers are per stage: VAD runs a tiny model on 512-sample windows and
    speaker embedding on short segments, both of which lose to kernel-launch
    overhead on a GPU, while ASR wins on it.
    """

    def __init__(self, asr: str, providers: dict, threads: int, vad_threshold: float,
                 vad_min_silence: float, int8: bool) -> None:
        import sherpa_onnx as so

        self.so = so
        self.threads = threads
        self.providers = providers

        self.vad_cfg = so.VadModelConfig()
        self.vad_cfg.silero_vad.model = str(MODELS / "silero_vad.onnx")
        self.vad_cfg.silero_vad.threshold = vad_threshold
        self.vad_cfg.silero_vad.min_silence_duration = vad_min_silence
        self.vad_cfg.silero_vad.max_speech_duration = VAD_MAX_SPEECH
        self.vad_cfg.sample_rate = 16000
        self.vad_cfg.provider = providers["vad"]
        self.vad_cfg.num_threads = threads

        if asr == "sensevoice":
            d = MODELS / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
            self.rec = so.OfflineRecognizer.from_sense_voice(
                model=str(d / ("model.int8.onnx" if int8 else "model.onnx")),
                tokens=str(d / "tokens.txt"), num_threads=threads,
                provider=providers["asr"], language="zh", use_itn=True,
            )
        else:
            d = MODELS / "sherpa-onnx-whisper-large-v3"
            self.rec = so.OfflineRecognizer.from_whisper(
                encoder=str(d / "large-v3-encoder.int8.onnx"),
                decoder=str(d / "large-v3-decoder.int8.onnx"),
                tokens=str(d / "large-v3-tokens.txt"),
                num_threads=threads, provider=providers["asr"], language="zh",
                task="transcribe",
            )

        self.spk = so.SpeakerEmbeddingExtractor(
            so.SpeakerEmbeddingExtractorConfig(
                model=str(MODELS / "campplus_zh.onnx"),
                provider=providers["spk"], num_threads=threads,
            )
        )
        pd = MODELS / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"
        self.punct = so.OfflinePunctuation(
            so.OfflinePunctuationConfig(
                model=so.OfflinePunctuationModelConfig(
                    ct_transformer=str(pd / "model.onnx"),
                    num_threads=threads, provider=providers["punct"],
                )
            )
        )

    def _vad(self, samples, rate) -> list[tuple[float, float]]:
        vad = self.so.VoiceActivityDetector(self.vad_cfg, buffer_size_in_seconds=100)
        win = self.vad_cfg.silero_vad.window_size
        spans: list[tuple[float, float]] = []

        def drain() -> None:
            while not vad.empty():
                seg = vad.front
                spans.append((seg.start / rate, (seg.start + len(seg.samples)) / rate))
                vad.pop()

        for i in range(0, len(samples) - win, win):
            vad.accept_waveform(samples[i : i + win])
            drain()
        vad.flush()
        drain()
        return spans

    def transcribe(self, wav: Path) -> tuple[list[dict], dict]:
        import numpy as np

        samples, rate = read_wav(wav)
        stage: dict[str, float] = {}

        t = time.perf_counter()
        spans = self._vad(samples, rate)
        stage["vad_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        texts = []
        for start, end in spans:
            st = self.rec.create_stream()
            st.accept_waveform(rate, samples[int(start * rate) : int(end * rate)])
            self.rec.decode_stream(st)
            texts.append(st.result.text.strip())
        stage["asr_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        embs = []
        for start, end in spans:
            st = self.spk.create_stream()
            st.accept_waveform(rate, samples[int(start * rate) : int(end * rate)])
            st.input_finished()
            embs.append(self.spk.compute(st))
        labels = cluster(np.asarray(embs, dtype="float32")) if embs else []
        stage["spk_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        kept = [(s, e, normalize(txt), f"SPK_{lbl}")
                for (s, e), txt, lbl in zip(spans, texts, labels, strict=True) if txt]
        raw = "".join(k[2] for k in kept)
        segments = align_punctuation(kept, self.punct.add_punctuation(raw)) if raw else []
        stage["punct_s"] = round(time.perf_counter() - t, 2)
        return segments, stage


class Funasr:
    """The current path, unmodified, as the baseline."""

    def __init__(self, device: str) -> None:
        from funasr import AutoModel

        base = Path.home() / ".bibilab" / "models" / "asr"
        self.model = AutoModel(
            model=str(base / "sensevoice-small"), device=device,
            vad_model=str(base / "fsmn-vad"),
            vad_kwargs={"max_single_segment_time": 15000, "max_end_silence_time": 500,
                        "speech_2_noise_ratio": 0.7},
            spk_model=str(base / "cam++"),
            disable_update=True, disable_pbar=True,
        )

    def transcribe(self, wav: Path) -> tuple[list[dict], dict]:
        t = time.perf_counter()
        res = self.model.generate(input=str(wav), use_itn=True, merge_vad=False, language="zh")
        elapsed = round(time.perf_counter() - t, 2)
        raw = (res[0].get("sentence_info") or []) if res else []
        segments = [
            {"start_s": float(s.get("start", 0)) / 1000.0,
             "end_s": float(s.get("end", 0)) / 1000.0,
             "speaker": f"SPK_{s.get('spk')}" if s.get("spk") is not None else None,
             "text": str(s.get("text") or "").strip()}
            for s in raw
            if str(s.get("text") or "").strip()
        ]
        # generate() is monolithic; it cannot report per-stage time.
        return segments, {"generate_s": elapsed}


# Mirrors pipeline/chunk.py's _SENT_END, which punctuate.py splits on.
SENT_END = ("。", "！", "？", "．", "…", "!", "?")


def align_punctuation(spans: list[tuple], punctuated: str) -> list[dict]:
    """Map one whole-transcript punctuation pass back onto the VAD spans.

    Mirrors pipeline/punctuate.py: the reference in the fixture is post-ct-punc
    *sentence* segments cut from a single punctuation call over the whole
    transcript. Punctuating each VAD span separately instead gives a very
    different sentence density, so the comparison only means something if this
    side is shaped the same way. Splits on sentence-final punctuation and on
    speaker change; each segment's time comes from the spans its characters
    came from.
    """
    out: list[dict] = []
    idx = 0            # which span the next raw char belongs to
    consumed = 0       # raw chars already taken from that span
    buf = ""
    first, last = idx, idx

    def flush() -> None:
        nonlocal buf, first
        if buf.strip():
            out.append({"start_s": spans[first][0], "end_s": spans[last][1],
                        "speaker": spans[first][3], "text": buf})
        buf = ""
        first = idx

    for ch in punctuated:
        if ch in PUNC:
            buf += ch
            if ch in SENT_END:
                flush()
            continue
        while idx < len(spans) and consumed >= len(spans[idx][2]):
            idx += 1
            consumed = 0
        if idx >= len(spans):
            raise ValueError("punctuation model emitted more content than it was given")
        if buf and spans[idx][3] != spans[first][3]:
            flush()
        buf += ch
        last = idx
        consumed += 1
    flush()

    taken = sum(len(s[2]) for s in spans[:idx]) + consumed
    total = sum(len(s[2]) for s in spans)
    if taken != total:
        raise ValueError(f"punctuation dropped content: consumed {taken}/{total} chars")
    return out


def cluster(embs, threshold: float = 0.5) -> list[int]:
    """Greedy cosine agglomeration -- the speaker count is unknown, as it is
    for CAM++ in the current path."""
    import numpy as np

    if len(embs) == 0:
        return []
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    centroids: list = []
    counts: list[int] = []
    labels: list[int] = []
    for e in embs:
        if centroids:
            sims = np.asarray(centroids) @ e
            best = int(sims.argmax())
            if sims[best] >= threshold:
                labels.append(best)
                centroids[best] = (centroids[best] * counts[best] + e) / (counts[best] + 1)
                centroids[best] /= np.linalg.norm(centroids[best]) + 1e-9
                counts[best] += 1
                continue
        centroids.append(e.copy())
        counts.append(1)
        labels.append(len(centroids) - 1)
    return labels


# ---------------------------------------------------------------------- scoring


def normalize(text: str) -> str:
    return "".join(c for c in text.lower() if c not in PUNC)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def flatten(segments: list[dict]) -> tuple[str, list[float]]:
    """Whole transcript as one string, plus the source time of every character.

    Scoring cannot bucket by time: the two paths use different VAD, so a
    sentence straddling any bucket edge lands on opposite sides and scores as a
    total mismatch when the text is in fact identical. The character-time map
    is what puts a divergence back on the clock afterwards.
    """
    chars: list[str] = []
    times: list[float] = []
    for s in segments:
        norm = normalize(s["text"])
        chars.append(norm)
        times.extend([(s["start_s"] + s["end_s"]) / 2] * len(norm))
    return "".join(chars), times


def score_text(ref: list[dict], hyp: list[dict]) -> dict:
    """Levenshtein over the whole transcript, split at difflib's common blocks.

    A full DP over two ~10k-char transcripts is minutes of Python. Near-identical
    text has long shared runs, so anchoring on them and running the exact DP only
    inside the disagreements costs milliseconds and gives the same distance.
    """
    from difflib import SequenceMatcher

    rt, r_times = flatten(ref)
    ht, _ = flatten(hyp)
    dist = 0
    diffs = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, rt, ht, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        d = edit_distance(rt[i1:i2], ht[j1:j2]) if tag == "replace" else max(i2 - i1, j2 - j1)
        dist += d
        diffs.append({"t": round(r_times[i1], 1) if i1 < len(r_times) else None,
                      "dist": d, "ref": rt[i1:i2], "hyp": ht[j1:j2],
                      "ref_ctx": rt[max(0, i1 - 15) : i2 + 15]})
    return {
        "cer": round(dist / len(rt), 4) if rt else None,
        "ref_chars": len(rt), "hyp_chars": len(ht),
        "diffs": diffs,
    }


def label_at(segments: list[dict], t: float) -> str | None:
    for s in segments:
        if s["start_s"] <= t < s["end_s"]:
            return s["speaker"]
    return None


def score_speakers(ref: list[dict], hyp: list[dict], duration: float, step: float = 5.0) -> dict:
    """Permutation-invariant: compare every pair of sampled instants on
    'same speaker or not', which needs no mapping between the two label sets."""
    pairs = [(label_at(ref, i * step), label_at(hyp, i * step))
             for i in range(int(duration / step))]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    agree = total = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            total += 1
            agree += (pairs[i][0] == pairs[j][0]) == (pairs[i][1] == pairs[j][1])
    return {
        "pairwise_agreement": round(agree / total, 4) if total else None,
        "sampled_points": len(pairs),
        "ref_speakers": len({s["speaker"] for s in ref}),
        "hyp_speakers": len({s["speaker"] for s in hyp}),
    }


# ----------------------------------------------------------------------- driver


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fetch-models", action="store_true")
    p.add_argument("--engine", choices=["sherpa", "funasr"], default="sherpa")
    p.add_argument("--asr", choices=["sensevoice", "whisper"], default="sensevoice")
    p.add_argument("--provider", default="cpu", help="sherpa: cpu|cuda|coreml")
    p.add_argument("--provider-vad", help="override, e.g. cpu while --provider=cuda")
    p.add_argument("--provider-spk")
    p.add_argument("--provider-punct")
    p.add_argument("--device", default="cpu", help="funasr: cpu|cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--share-model", action="store_true",
                   help="sherpa only: one engine across all workers (FunASR's "
                        "generate() mutates shared state, so it always gets one each)")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--vad-threshold", type=float, default=0.5)
    p.add_argument("--vad-min-silence", type=float, default=VAD_MIN_SILENCE)
    p.add_argument("--sources", help="comma-separated source_id prefixes; default all")
    p.add_argument("--dump-divergent", type=int, default=0,
                   help="write the N worst windows per source for human adjudication")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    if args.fetch_models:
        fetch_models()
        return

    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    picks = manifest["sources"]
    if args.sources:
        want = set(args.sources.split(","))
        picks = [s for s in picks if s["source_id"] in want]
    if not picks:
        raise SystemExit("no sources selected")

    for s in picks:
        wav = FIXTURE / f"{s['source_id']}.wav"
        got = sha256(wav.read_bytes()).hexdigest()
        if got != s["wav_sha256"]:
            raise SystemExit(f"{wav.name} does not match the manifest -- fixture corrupted")

    providers = {
        "vad": args.provider_vad or args.provider,
        "asr": args.provider,
        "spk": args.provider_spk or args.provider,
        "punct": args.provider_punct or args.provider,
    }

    def build():
        if args.engine == "funasr":
            return Funasr(args.device)
        return Sherpa(args.asr, providers, args.threads, args.vad_threshold,
                      args.vad_min_silence, args.int8)

    t = time.perf_counter()
    shared = build() if (args.share_model and args.engine == "sherpa") else None
    local = None if shared else __import__("threading").local()
    load_s = round(time.perf_counter() - t, 2)

    def run(src: dict) -> dict:
        engine = shared
        if engine is None:
            if not hasattr(local, "engine"):
                local.engine = build()
            engine = local.engine
        wav = FIXTURE / f"{src['source_id']}.wav"
        t0 = time.perf_counter()
        segments, stage = engine.transcribe(wav)
        wall = round(time.perf_counter() - t0, 2)

        ref = json.loads((FIXTURE / f"{src['source_id']}.reference.json").read_text())
        text = score_text(ref["segments"], segments)
        spk = score_speakers(ref["segments"], segments, src["duration_s"])
        diffs = sorted(text.pop("diffs"), key=lambda d: -d["dist"])
        out = {
            "source_id": src["source_id"], "title": src["title"],
            "duration_s": src["duration_s"], "wall_s": wall,
            "rtf": round(wall / src["duration_s"], 4),
            "segments": len(segments), "ref_segments": src["segments"],
            "n_diffs": len(diffs), "stage_s": stage, **text, **spk,
        }
        if args.dump_divergent:
            out["divergent"] = diffs[: args.dump_divergent]
        print(f"  {src['source_id']} {wall:6.1f}s rtf={out['rtf']:.3f} "
              f"cer={out['cer']} spk={out['pairwise_agreement']}", flush=True)
        return out

    t0, c0 = time.perf_counter(), cpu_seconds()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, picks))
    wall, cpu = time.perf_counter() - t0, cpu_seconds() - c0

    audio_s = sum(s["duration_s"] for s in picks)
    total_ref = sum(r["ref_chars"] for r in results)
    report = {
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": platform.python_version()},
        "config": {"engine": args.engine, "asr": args.asr, "providers": providers,
                   "device": args.device, "threads": args.threads,
                   "concurrency": args.concurrency, "share_model": bool(shared),
                   "int8": args.int8, "vad_threshold": args.vad_threshold,
                   "vad_min_silence": args.vad_min_silence},
        "totals": {
            "sources": len(picks), "audio_s": audio_s,
            "model_load_s": load_s, "wall_s": round(wall, 1),
            "cpu_s": round(cpu, 1), "effective_cores": round(cpu / wall, 2),
            "throughput_x_realtime": round(audio_s / wall, 2),
            "peak_rss_gb": round(peak_rss_gb(), 2),
            "cer": round(sum(r["cer"] * r["ref_chars"] for r in results) / total_ref, 4)
            if total_ref else None,
        },
        "sources": results,
    }
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\n-> {args.out}")
    print(json.dumps(report["totals"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

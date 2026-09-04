# 0001 — ASR runtime

**Status:** Reversed 2026-09-04. FunASR + PyTorch → **sherpa-onnx, CPU-only on every
platform.** Migration tracked in #679.
**Originally decided:** 2026-05-28 (PR #374). **Re-measured:** 2026-09-03 (Linux),
2026-09-04 (macOS).

This document keeps the original decision and the argument that overturned it. The
reversal is only legible next to what it replaced.

## The original decision (2026-05-28)

`pipeline/transcribe.py` runs every ASR stage through one `funasr.AutoModel`:
SenseVoice or Whisper for recognition, FSMN-VAD for segmentation, CAM++ for speaker
labelling, ct-punc for punctuation. FunASR is a PyTorch library, so `torch` +
`torchaudio` are production dependencies, split across `cpu` / `cuda` / `rocm`
conflicting dependency groups in `pyproject.toml`.

Two reasons were carried for keeping it:

1. Whisper depends on torch, so torch cannot be removed.
2. CUDA is worth its install cost because the GPU is much faster at transcription.

## Why it was reopened — both reasons measured false

Test bed: 43.1 min 16 kHz mono, three alternating speakers. Host: i9-14900F (16P/32T),
RTX 5070 Ti (sm_120), WSL2. Same audio, same pipeline stages, every row.

| Runtime | wall | peak RSS | VRAM | install |
|---|---:|---:|---:|---:|
| FunASR CPU (torch, ncpu=4) | 145.2 s | 3.20 GB | — | ~6 GB |
| FunASR GPU (torch cu130) | 79.0 s | 3.2 GB | 6.6 GB | ~6 GB |
| sherpa-onnx CPU, 4 threads | 77.9 s | 1.38 GB | — | **43 MB** |
| sherpa-onnx CPU, 16 threads | 73.6 s | 1.41 GB | — | **43 MB** |
| sherpa-onnx all-CUDA, 4 threads | 132.3 s | 2.70 GB | small | 374 MB |
| sherpa-onnx mixed provider, 16 threads | **35.9 s** | 1.61 GB | small | 374 MB |

**Reason 1 is false.** `faster-whisper` and `ctranslate2` were dropped from
`pyproject.toml` in #426; Whisper now rides FunASR's `WhisperWarp`. Torch is pulled in
by FunASR, not by Whisper. `sherpa_onnx.OfflineRecognizer.from_whisper` exists, as do
`from_sense_voice`, `OfflineSpeakerDiarization`, `SileroVadModelConfig` and
`OfflinePunctuation` — the whole pipeline surface, in a 43 MB runtime.

**Reason 2 is false.** The 1.84× GPU win compares FunASR's GPU branch against its own
CPU branch, which upstream deliberately hobbles:

```python
# funasr/auto/auto_model.py:935-936
if kwargs["device"] == "cpu":
    batch_size = 0          # CPU only; the GPU branch keeps batch_size_s=300
```

A runtime without that asymmetry reaches GPU-class throughput on CPU alone. The GPU
never was the bottleneck: utilisation stayed at ~21% and raising `batch_size_s` bought
utilisation but not throughput (300→1200 moved wall time 79.0 s → 102.6 s, VRAM
6.6 GB → 13.4 GB; 2400 exhausted VRAM and hung).

## Why the workload does not want a GPU

Per-stage timing, only visible once the stages are split apart — FunASR's monolithic
`generate()` never exposed it:

| stage (43.1 min) | CPU 16t | all-CUDA 4t | mixed 16t |
|---|---:|---:|---:|
| VAD (Silero) | 8.5 s | 95.5 s | 9.7 s (cpu) |
| ASR (SenseVoice) | 51.8 s | 11.0 s | 12.2 s (cuda) |
| Speaker (CAM++ per segment) | 11.5 s | 23.8 s | 11.6 s (cpu) |

VAD runs a tiny model over 512-sample windows: one kernel launch per window, 11× slower
on GPU than CPU. Speaker embedding is small per-segment work, 2× slower. Only ASR has
tensors large enough to pay for the device, at 4.7×. The right configuration is
per-stage provider selection — which `AutoModel` cannot express, having a single global
`device`.

The same shape explains two older observations. `83fc1b35` measured "GPU only ~17%
busy" and serialised transcription behind a lock; that reading was correct. And CPU
utilisation sits near 10% because `AutoModel` overrides `torch.set_num_threads` with a
hardcoded `ncpu=4` (`auto_model.py:569-572`) while `batch_size = 0` leaves the threads
nothing to batch — raising `ncpu` to 32 makes it **0.42×**, not faster. sherpa-onnx has
neither defect and reaches 16.1 effective cores at 16 threads.

Those numbers killed the two stated reasons but not the decision: a 43-minute single
sample on one host says nothing about transcript quality or about the platforms this
runs on. Everything above is a *runtime* comparison. #679 was opened to answer whether
the output survives the swap, and whether it survives it everywhere.

## The re-evaluation (#679)

### The measuring instrument came first

9 sources sampled from the local library, 82.7 min, stratified over speaker count
(1–10), duration (40 s – 20 min) and content type. All `zh` — the library holds no
other language, so none of this speaks to the non-`zh` path.

The pipeline deletes its audio after transcription, so the wavs were re-fetched through
the production adapter and `extract_audio` and sha256-pinned, making them byte-identical
to what the pipeline originally fed the model. Model weights are pinned the same way, so
a silently-redeployed release is a hard stop rather than a drifting number.

**The reference transcripts are the library's own** — produced by the current FunASR +
FSMN-VAD + CAM++ + ct-punc path. `cer` therefore measures **divergence from what we
ship today, not accuracy.** There are confirmed diffs where sherpa is the correct side:
the reference reads `搅拌均均匀`, sherpa emits `搅拌均匀`.

Before trusting a single comparison, the harness was pointed at the incumbent: FunASR,
run through the production functions themselves, scored **CER 0.0000** against its own
reference. That one number validates audio identity, reference extraction and the
scoring function simultaneously, and it is the reason the rest of the table is worth
reading.

### Throughput, whole fixture, busy pipeline

| config | wall | throughput | peak RSS |
|---|---:|---:|---:|
| FunASR c=1 (today) | 300.9 s | 16.5× realtime | 6.08 GB |
| FunASR c=2 | 289.8 s | 17.1× | 6.27 GB |
| sherpa CPU c=1 | 151.8 s | 32.7× | 2.22 GB |
| sherpa CPU c=4, shared model | 96.9 s | **51.2×** | **2.38 GB** |

FunASR gains 3.8% from concurrency because `_transcribe_lock` serialises it. That is
today's real behaviour, lock included, so it is the honest baseline.

sherpa runs four workers off **one shared model for +0.16 GB**, and CER is
bit-identical across all four concurrency levels — which is the evidence that sharing
is not corrupting state. FunASR cannot do this: each instance is +3.2 GB, and c=4 would
OOM a 16 GB host, so the baseline was capped at c=2 rather than risking the machine.

**sherpa at c=4 is 3.1× today's throughput at 0.39× the memory.**

### Quality

Weighted CER **0.048** over 82.7 min. Clean speech lands at 1.7–5.0%. The only
systematic weakness is two cooking videos with music under the speech, at 12.8% and
14.7%.

Segment density mostly lands at 0.82–1.06× the reference, with two outliers — 2.11×
over-split and 0.41× under-split. Both reproduce identically on macOS, so they are the
untuned Silero VAD, not a platform artifact.

Speaker labelling is unresolved **on both sides**: sherpa over-splits (11 vs 10, 4 vs
2), but the reference itself claims 10 speakers on a single anime recap. Neither side
is a credible target. This needs ground truth, not a comparison.

### Silero VAD tuning — measured, and it is not the lever

Two 3×3 sweeps, `threshold` ∈ {0.3, 0.5, 0.7} × `min_silence_duration` ∈ {0.10,
0.25, 0.50}. Default is `0.5 / 0.50`.

On the three multi-speaker sources, **threshold 0.3 dominates 0.5 on CER at every
min_silence** (0.0444 / 0.0428 / 0.0423 against 0.0519 / 0.0515 / 0.0498), and 0.7 is
worst everywhere (0.0656 / 0.0640 / 0.0606). A higher threshold gates out real speech.
`0.3 / 0.25` beats the default on CER *and* speaker agreement at once — 0.0428 vs
0.0498 and 0.795 vs 0.745 — which is the only non-trade in the grid.

**The density outliers do not move.** Across all nine settings `310eeb03` stays at
2.11–2.27× the reference and `bf36b4f0` at 0.40–0.43×. Segment count is set by ct-punc
sentence splitting, not by VAD: at `0.3 / 0.50` the tech talk is cut into **23 VAD spans
with a p95 width of 37 s**, which punctuation then splits into 101 sentences. The
earlier guess on file — that `threshold=0.70, min_silence_duration=0.10` fixes segment
density — was inferred from segment counts alone and is wrong in both direction and
mechanism.

**Speaker labelling is not fixable here either.** Count and pairwise agreement point
opposite ways: `0.3 / 0.50` gets two sources exactly right (7:7, 2:2) with the *worst*
agreement (0.682), while `0.7 / 0.10` has the best agreement (0.840) at 29 speakers
against a reference 10. The mechanism is visible in the span columns — `min_silence
0.50` collapses 569 spans to 222 and pushes p95 span width to 17.3 s, so it reaches the
right count by merging, including across speaker changes. That is the conflation #384
fixed, reintroduced through a different knob.

One setting, `0.7 / 0.10` on the density pair, aborted the run: ct-punc consumed
3174 of 3178 characters, tripping the harness's content-preservation check. Production
already handles this — `_align` raises and `punctuate` catches it, degrading to
unpunctuated segments with a warning — but it means that setting would silently cost a
whole source its punctuation. Another mark against high thresholds.

**Conclusion: adopt `threshold=0.3, min_silence_duration=0.25`** for the ~0.007 absolute
CER gain and the speaker-agreement gain, and treat segment density and speaker
labelling as open problems that live downstream of VAD.

### GPU — measured, then declined

| config | wall | throughput | VRAM | CER |
|---|---:|---:|---:|---:|
| CPU c=4 | 96.9 s | 51.2× | — | 0.0482 |
| **CUDA mixed c=2** | **60.7 s** | **81.7×** | 4.3 GB | 0.0482 |
| CUDA mixed c=4 | 60.8 s | 81.6× | 4.4 GB | 0.0482 |
| **all-CUDA c=1** | **284.7 s** | **17.4×** | 5.4 GB | 0.0483 |

Mixed = ASR on CUDA, VAD + speaker + punctuation on CPU. All-CUDA is not merely
suboptimal, it is **4.7× slower than mixed and slower than CPU-only** — the same
kernel-launch story as above, now at fixture scale. Mixed saturates at c=2.

Install cost, measured rather than assumed: sherpa CPU-only is a **107 MB** venv. The
CUDA path adds a **1.1 GB** transitive closure (cublas, cublasLt, cudart, cufft, curand,
nvrtc, cudnn) plus ~900 MB of cuDNN engine libraries loaded at runtime.

So the GPU trade is **~2 GB of install and a per-platform dependency matrix for 1.60×**,
on top of a CPU path already running at 3.1× today's throughput.

### macOS — Apple M1 (4P+4E), 16 GB, fanless MacBook Air

CoreML initialises cleanly and all four stages run. **CER came back 0.0482, identical to
Linux**, at every CPU concurrency level and in the CoreML-mixed config. Same weights →
same transcript across OS, architecture and provider. The lone shift is all-CoreML on
one source (0.0410 → 0.0404), i.e. ANE half-precision numerics on one stage.

CoreML buys nothing: 12.73× vs 15.85× for CPU c=1, and mixed c=2 at 21.23× against CPU
c=4 at 21.88×. Peak RSS stayed 1.8–3.0 GB, matching Linux.

**The macOS throughput column is thermally confounded and unusable as a scaling curve** —
the same configuration measured 15.85× cold and 8.15× on the next run, recovering to
21.88× later. That is chassis temperature interacting with run order. Only three facts
from that host survive: CER stability, flat RSS, and CoreML working.

### Whisper — runs, fails on quality

All four stages produce output, so the branch is wired correctly. The transcript is not:
**CER 0.40**, characters dropped throughout, junk tokens, at rtf 0.81 — **26× slower
than SenseVoice**. Cause: the k2-fsa release ships large-v3 **int8 only**, and int8
large-v3 degrades badly on Chinese. Two ways out: export a non-quantized large-v3, or
substitute a smaller Whisper that ships fp32. Blocks nothing today — all 55 library
sources are SenseVoice.

## The decision

**Adopt sherpa-onnx. Ship the CPU build on every platform. Do not ship an accelerator
path.**

The accelerator question is settled by two hosts agreeing from opposite directions:
CUDA buys 1.60× for ~2 GB, CoreML buys nothing at all. Declining both collapses the
`cpu` / `cuda` / `rocm` conflicting dependency groups in `pyproject.toml` into a single
107 MB wheel, and removes `torch` + `torchaudio` from production entirely.

Consequences that follow, and must not be re-litigated against the old numbers:

- **`_transcribe_lock` and `max_concurrent_jobs`** were both sized against a 3.2 GB
  resident model that mutates shared state on `generate()`. Neither premise survives:
  the model is 1.4 GB and is safely shared across four workers.
- **The cu130 bump** — `uv.lock` pins `torch 2.6.0+cu124` while sm_120 needs CUDA 12.8,
  so the `cuda` group is broken on RTX 50-series. It was a real bug in a path that
  "may not survive". It did not survive; do not fix it.

## Traps, so they are not stepped in twice

1. **Never benchmark a runtime's GPU branch against its own CPU branch.** FunASR forces
   `batch_size = 0` on CPU only (`auto_model.py:935-936`) and hardcodes `ncpu=4`. The
   "1.84× GPU win" was measuring a deliberate handicap, and it survived on file for
   three months.
2. **All-accelerator is a trap in a multi-stage pipeline.** VAD and speaker embedding
   are tiny models on short windows; kernel launch dominates and they run *slower* on
   the device. Only ASR pays for it. A runtime with one global `device` cannot express
   this, which is itself an argument against such a runtime.
3. **Windowed CER is a lie when two paths segment differently.** Bucketing by time
   scored 0.5542 where the true divergence was 0.0542 — a sentence straddling a bucket
   edge lands on opposite sides and reads as a total mismatch when the text is
   identical. Score whole-transcript with anchored alignment, or not at all.
4. **A reference produced by your own current pipeline measures divergence, never
   accuracy.** Label it that way in the harness, or someone will quote it as a quality
   number.
5. **Segment counts are post-punctuation sentences, not VAD spans.** ct-punc runs once
   over the whole concatenated transcript, outside the ASR model; punctuating per-span
   instead gives a completely different sentence density and an invented regression.
6. **Throughput from a fanless laptop under sustained load is thermal state, not
   scaling.** Cross-machine comparisons need either cooldown gaps or a chassis with a
   fan.
7. **Two "install size" numbers can both be right.** The 43 MB / 374 MB figures above
   are wheel sizes; 107 MB / 1.1 GB are resolved venvs. Say which one is being quoted.
8. **Do not attribute an effect to the nearest plausible knob without turning it.**
   Segment density was blamed on VAD parameters and written into this document as a
   tuning suggestion. Nine settings later, density had not moved at all — the cause is
   downstream, in sentence splitting. The guess cost nothing to write and would have
   cost a migration to believe.

## Still not measured

- **Speaker-labelling accuracy** against real ground truth. Both sides are suspect; the
  comparison cannot resolve it, and the VAD sweep showed the two available metrics
  disagreeing about which direction is better.
- **Segment density.** Now known *not* to be a VAD problem. Where ct-punc sentence
  splitting diverges from the incumbent, and whether that matters downstream of
  chunking, is unexamined.
- **Hand adjudication** of the divergent windows — the 4.8% is known to contain cases
  where sherpa is correct, but the split has not been counted.
- **Any non-`zh` audio.** The library has none.
- **A FunASR baseline on macOS.** The 3.1× and 0.39× figures are Linux-only. Measuring
  it on the Air would need interleaved runs with forced cooldowns, reporting the ratio
  and never the absolutes — judged not worth a machine, since ONNX-CPU beating
  PyTorch-CPU is not in doubt and the memory delta is architectural.

# 0001 — ASR runtime: FunASR + PyTorch

**Status:** Contested — the decision stands, both of its stated reasons measured false.
**Decided:** 2026-05-28 (PR #374). **Measured:** 2026-09-03.

## The decision as it stands

`pipeline/transcribe.py` runs every ASR stage through one `funasr.AutoModel`:
SenseVoice or Whisper for recognition, FSMN-VAD for segmentation, CAM++ for speaker
labelling, ct-punc for punctuation. FunASR is a PyTorch library, so `torch` +
`torchaudio` are production dependencies, split across `cpu` / `cuda` / `rocm`
conflicting dependency groups in `pyproject.toml`.

Two reasons were carried for keeping it:

1. Whisper depends on torch, so torch cannot be removed.
2. CUDA is worth its install cost because the GPU is much faster at transcription.

## What the measurements say

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

## Consequences held open

- **cu130 bump.** `uv.lock` pins `torch 2.6.0+cu124`; sm_120 entered the toolchain in
  CUDA 12.8, so no cu124 wheel can drive Blackwell — the `cuda` group is simply broken
  on RTX 50-series. Verified working under `torch 2.14.0+cu130`. Real bug, but it
  repairs a path that may not survive.
- **Transcribe lock and worker concurrency.** Both were sized against a 3.2 GB
  resident model. At 1.4 GB the arithmetic changes; do not re-litigate concurrency
  against the old number.

## Not measured

WER, speaker-labelling accuracy, and the Whisper branch (SenseVoice only). Silero's
default VAD parameters produce coarser segments than FSMN-VAD tuned with
`speech_2_noise_ratio=0.7` — 76 vs 138 segments on a 10.8 min sample, with an observed
9 s segment spanning a speaker change, the exact conflation #384 fixed. A sweep lands
`threshold=0.70, min_silence_duration=0.10` at 142 segments, so it is a tuning knob
rather than a wall, but it is untuned and unvalidated.

Nothing here decides the migration. It records that the reasons on file no longer hold.
Re-evaluation tracked in #679.

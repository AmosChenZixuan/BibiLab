# ASR benchmark (#679)

Throwaway harness for the sherpa-onnx re-evaluation. Lives on `onnx-migration`
and is deleted before that branch merges to `master` — the deliverable is the
numbers in `docs/decisions/`, not this code.

## What is frozen, and why

`fixture.tar` (151 MB, 9 sources, 82.7 min) is the unit of portability. The
pipeline deletes its audio after transcription, so the wavs were re-fetched with
the production adapter and `extract_audio`, making them byte-identical to what
the pipeline feeds the model. Copy the tarball to the other machine rather than
re-downloading there: a second download is not guaranteed to give the same
bytes, and then the two machines are not running the same benchmark.

`models/models.lock.json` pins the sha256 of every model file. `--fetch-models`
refuses to proceed if a digest drifts.

The fixture ships a reference transcript per source, taken from the library —
i.e. produced by the current FunASR + FSMN-VAD + CAM++ + ct-punc path. **It is a
baseline, not ground truth.** `cer` measures divergence from what we ship today
and says nothing about which side is correct. Use `--dump-divergent N` to write
the worst windows side by side, then adjudicate by hand.

Corpus shape: 9 sources spanning 1–10 speakers, 40 s – 20 min, four content
types (anime recap, cooking, tech talk, lore narration). All Chinese — the
library holds no other language, so nothing here says anything about the
non-`zh` path.

## Running it

```bash
# once per machine
python bench.py --fetch-models

# quality: one file at a time
python bench.py --engine sherpa --provider cpu --threads 4 --concurrency 1 \
                --dump-divergent 3 --out results/$(uname -s)-sherpa-cpu.json

# throughput: the whole fixture, N at a time
for n in 1 2 3 4; do
  python bench.py --engine sherpa --provider cpu --threads 4 --concurrency $n \
                  --share-model --out results/sherpa-cpu-c$n.json
done

# baseline (needs the backend env)
cd ../../backend && uv run python ../bench/asr/bench.py --engine funasr --device cpu
```

Providers are per stage. All-GPU is a trap — VAD and speaker embedding lose to
kernel-launch overhead — so the mixed configuration is what a GPU run should
measure:

```bash
python bench.py --provider cuda --provider-vad cpu --provider-spk cpu --provider-punct cpu
```

On macOS substitute `--provider coreml`.

## Whisper

`--asr whisper` runs all four stages, but the k2-fsa release ships large-v3 as
int8 only, and int8 large-v3 drops characters throughout on Chinese: CER 0.40
against the SenseVoice reference, at 26× the cost (rtf 0.81 vs 0.031). Treat the
Whisper branch as unmeasured until either a non-quantized large-v3 is exported or
a smaller Whisper that ships fp32 is substituted. Nothing in the library uses
Whisper today — all 55 sources are SenseVoice — so this blocks nothing yet.

## Requirements

- sherpa path: `sherpa-onnx==1.13.7`, `numpy`, `soundfile`
- funasr baseline: the backend env, so run it under `uv run` from `backend/`

Note `--share-model` is sherpa-only. FunASR's `generate()` mutates shared state
(this is what `_transcribe_lock` exists to serialize), so every worker gets its
own instance and memory scales with concurrency.

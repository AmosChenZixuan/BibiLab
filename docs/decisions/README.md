# Decision Log

Load-bearing system decisions and the evidence behind them. One file per decision,
numbered, never rewritten in place — a decision that gets overturned gets a new
entry that supersedes the old one, so the reasoning trail survives.

What belongs here: a choice that would be expensive to reverse and whose rationale
is not recoverable from the code (runtime/engine selection, storage model, protocol
shape). What does not: implementation notes (`docs/specs/`, untracked), anything the
code or git history already states plainly.

Each entry carries a **Status**: `Accepted` · `Superseded by NNNN` · `Contested`
(the decision stands but its stated rationale has been measured false).

| # | Decision | Status |
|---|---|---|
| [0001](0001-asr-runtime.md) | ASR runtime: FunASR + PyTorch | Contested |

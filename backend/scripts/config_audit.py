"""Audit persisted config for fields that drifted from Pydantic defaults.

Compares ~/.bibilab/config.json against BibilabConfig().model_dump() and
reports any field where the persisted value overrides a newer code default.

Usage:
    uv run python scripts/config_audit.py
"""

import json

from bibilab.config import BibilabConfig, bibilab_home


def walk(prefix: str, persisted: dict, default: dict) -> list[str]:
    diffs: list[str] = []
    for key in default:
        full = f"{prefix}.{key}"
        p_val = persisted.get(key)
        d_val = default[key]

        if isinstance(d_val, dict) and isinstance(p_val, dict):
            diffs.extend(walk(full, p_val, d_val))
        elif p_val != d_val:
            if isinstance(d_val, float) and isinstance(p_val, (int, float)):
                if abs(p_val - d_val) < 1e-6:
                    continue
            diffs.append(f"  {full}: persisted={p_val!r}, default={d_val!r}")
    return diffs


def main() -> None:
    config_path = bibilab_home() / "config.json"
    if not config_path.exists():
        print("No config file found. Nothing to audit.")
        return

    persisted = json.loads(config_path.read_text())
    default = BibilabConfig().model_dump()

    diffs = walk("config", persisted, default)

    # Filter out obviously-intentional user settings
    intentional_prefixes = {
        "config.accounts",
        "config.ai",
        "config.backend.cors_origins",
    }
    real_diffs = [d for d in diffs if not any(d.strip().startswith(p) for p in intentional_prefixes)]

    if real_diffs:
        print("Fields where persisted value overrides a changed default:\n")
        for d in real_diffs:
            print(d)
        print(f"\n{len(real_diffs)} drifted fields found.")
    else:
        print("No drifted fields. Config matches current defaults.")

    # Also report intentional diffs for transparency
    if diffs != real_diffs:
        print(f"\n({len(diffs) - len(real_diffs)} intentional diffs omitted: accounts, ai, cors_origins)")


if __name__ == "__main__":
    main()

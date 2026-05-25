import json

import pytest

from eval.config import (
    DEFAULT_LANGUAGE,
    PROFILE_NAMES,
    _eval_config_path,
    get_language,
    load_eval_config,
    resolve_profile,
    save_eval_config,
)


SAMPLE_CONFIG = {
    "profiles": {
        "generate": None,
        "test": {
            "protocol": "openai",
            "model": "glm-4.7-flash",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        },
        "grade": None,
    },
    "language": "en",
}


def test_eval_config_path(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    assert _eval_config_path() == tmp_path / "eval_config.json"


def test_load_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    cfg = load_eval_config()
    assert cfg["profiles"]["generate"] is None
    assert cfg["profiles"]["grade"] is None
    assert cfg["profiles"]["test"]["model"] == "glm-4.7-flash"
    assert cfg["language"] == DEFAULT_LANGUAGE


def test_load_from_file(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    tmp_path.joinpath("eval_config.json").write_text(json.dumps(SAMPLE_CONFIG))
    cfg = load_eval_config()
    assert cfg["profiles"]["test"]["model"] == "glm-4.7-flash"
    assert cfg["language"] == "en"


def test_invalid_language_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    bad = dict(SAMPLE_CONFIG)
    bad["language"] = "fr"
    tmp_path.joinpath("eval_config.json").write_text(json.dumps(bad))
    assert load_eval_config()["language"] == DEFAULT_LANGUAGE


def test_resolve_profile_null_uses_backend(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)
    monkeypatch.setattr("bibilab.config._config_cache", None)
    backend = {
        "ai": {
            "protocol": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test",
            "base_url": "https://api.openai.com/v1",
            "output_language": "ui",
            "transcript_char_limit": 400000,
        }
    }
    tmp_path.joinpath("config.json").write_text(json.dumps(backend))
    tmp_path.joinpath("eval_config.json").write_text(
        json.dumps({"profiles": {"generate": None}, "language": "zh"})
    )
    ai = resolve_profile("generate")
    assert ai.model == "gpt-4o"
    assert ai.api_key == "sk-test"


def test_resolve_profile_custom(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    tmp_path.joinpath("eval_config.json").write_text(json.dumps(SAMPLE_CONFIG))
    ai = resolve_profile("test")
    assert ai.model == "glm-4.7-flash"
    assert ai.api_key == "ollama"
    assert ai.base_url == "http://localhost:11434/v1"


def test_resolve_unknown_profile_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    with pytest.raises(KeyError):
        resolve_profile("bogus")


def test_save_eval_config(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    save_eval_config(SAMPLE_CONFIG)
    loaded = json.loads((tmp_path / "eval_config.json").read_text())
    assert loaded["profiles"]["test"]["model"] == "glm-4.7-flash"
    assert loaded["language"] == "en"


def test_get_language(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    tmp_path.joinpath("eval_config.json").write_text(json.dumps(SAMPLE_CONFIG))
    assert get_language() == "en"


def test_profile_names_constant():
    assert PROFILE_NAMES == ("generate", "test", "grade")

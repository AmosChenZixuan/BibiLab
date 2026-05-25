# eval/tests/test_config.py
import json
from pathlib import Path
import pytest
from eval.config import load_eval_config, resolve_profile, save_eval_config, _eval_config_path

SAMPLE_CONFIG = {
    "generate": {"source": "inherit"},
    "test": {
        "source": "custom",
        "protocol": "openai",
        "model": "glm-4.7-flash",
        "base_url": "http://localhost:11434/v1",
    },
    "grade": {"source": "inherit"},
}


def test_eval_config_path(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    path = _eval_config_path()
    assert path == tmp_path / "eval_config.json"


def test_load_eval_config_not_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    cfg = load_eval_config()
    assert cfg["generate"]["source"] == "inherit"
    assert cfg["test"]["source"] == "custom"
    assert cfg["test"]["model"] == "glm-4.7-flash"


def test_load_eval_config_from_file(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    tmp_path.joinpath("eval_config.json").write_text(json.dumps(SAMPLE_CONFIG))
    cfg = load_eval_config()
    assert cfg["generate"]["source"] == "inherit"
    assert cfg["test"]["model"] == "glm-4.7-flash"


def test_resolve_profile_inherit(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    monkeypatch.setattr("bibilab.config.bibilab_home", lambda: tmp_path)
    monkeypatch.setattr("bibilab.config._config_cache", None)  # reset cache
    backend_config = {
        "ai": {
            "protocol": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test",
            "base_url": "https://api.openai.com/v1",
            "output_language": "ui",
            "transcript_char_limit": 400000,
        }
    }
    tmp_path.joinpath("config.json").write_text(json.dumps(backend_config))
    tmp_path.joinpath("eval_config.json").write_text(
        json.dumps({"generate": {"source": "inherit"}})
    )
    from bibilab.config import AIConfig

    ai_cfg = resolve_profile("generate")
    assert isinstance(ai_cfg, AIConfig)
    assert ai_cfg.model == "gpt-4o"
    assert ai_cfg.api_key == "sk-test"


def test_resolve_profile_custom(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    tmp_path.joinpath("eval_config.json").write_text(json.dumps(SAMPLE_CONFIG))
    from bibilab.config import AIConfig

    ai_cfg = resolve_profile("test")
    assert isinstance(ai_cfg, AIConfig)
    assert ai_cfg.model == "glm-4.7-flash"
    assert ai_cfg.api_key == "ollama"
    assert ai_cfg.base_url == "http://localhost:11434/v1"


def test_save_eval_config(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    cfg = SAMPLE_CONFIG
    save_eval_config(cfg)
    path = tmp_path / "eval_config.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["test"]["model"] == "glm-4.7-flash"

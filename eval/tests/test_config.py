import json

import pytest

from eval.config import (
    DEFAULT_LANGUAGE,
    PROFILE_NAMES,
    EvalConfig,
    Language,
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
    assert cfg.profiles["generate"] is None
    assert cfg.profiles["grade"] is None
    assert cfg.profiles["test"].model == "glm-4.7-flash"
    assert cfg.language == DEFAULT_LANGUAGE


def test_load_from_file(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    tmp_path.joinpath("eval_config.json").write_text(json.dumps(SAMPLE_CONFIG))
    cfg = load_eval_config()
    assert cfg.profiles["test"].model == "glm-4.7-flash"
    assert cfg.language == Language.EN


def test_invalid_language_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    bad = dict(SAMPLE_CONFIG)
    bad["language"] = "fr"
    tmp_path.joinpath("eval_config.json").write_text(json.dumps(bad))
    assert load_eval_config().language == DEFAULT_LANGUAGE


def test_unknown_profile_names_filtered(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    bad = {"profiles": {"bogus": {"model": "x"}, "test": None}, "language": "zh"}
    tmp_path.joinpath("eval_config.json").write_text(json.dumps(bad))
    cfg = load_eval_config()
    assert "bogus" not in cfg.profiles
    assert cfg.profiles["test"] is None


def test_resolve_profile_null_returns_none(tmp_path, monkeypatch):
    """None = no override: requests omit the `llm` field and the backend
    serves the call with its own configured LLM."""
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    tmp_path.joinpath("eval_config.json").write_text(
        json.dumps({"profiles": {"generate": None}, "language": "zh"})
    )
    assert resolve_profile("generate") is None


def test_resolve_profile_custom(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    tmp_path.joinpath("eval_config.json").write_text(json.dumps(SAMPLE_CONFIG))
    profile = resolve_profile("test")
    assert profile.model == "glm-4.7-flash"
    assert profile.api_key == "ollama"
    assert profile.base_url == "http://localhost:11434/v1"


def test_resolve_unknown_profile_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    with pytest.raises(KeyError):
        resolve_profile("bogus")


def test_save_eval_config(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    save_eval_config(EvalConfig.model_validate(SAMPLE_CONFIG))
    loaded = json.loads((tmp_path / "eval_config.json").read_text())
    assert loaded["profiles"]["test"]["model"] == "glm-4.7-flash"
    assert loaded["language"] == "en"


def test_get_language(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    tmp_path.joinpath("eval_config.json").write_text(json.dumps(SAMPLE_CONFIG))
    assert get_language() == Language.EN


def test_profile_names_constant():
    assert PROFILE_NAMES == ("generate", "test", "grade")


def test_get_response_language_returns_code(tmp_path, monkeypatch):
    """Returns the language code, not the display name — build_grounding_prompt
    looks the result up in _LANG_NATIVE_NAME which is keyed on codes. A
    display-name string would fall through to the English default."""
    from eval.config import get_response_language

    # eval.config imports bibilab_home at module level, so the local binding
    # is what _eval_config_path() resolves; patch it there.
    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    cfg_path = tmp_path / "eval_config.json"

    cfg_path.write_text('{"language": "zh", "profiles": {}}')
    assert get_response_language() == "zh"

    cfg_path.write_text('{"language": "en", "profiles": {}}')
    assert get_response_language() == "en"


def test_backend_url_default_and_roundtrip(tmp_path, monkeypatch):
    from eval.config import DEFAULT_BACKEND_URL, get_backend_url

    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    assert get_backend_url() == DEFAULT_BACKEND_URL == "http://127.0.0.1:8765"

    cfg = load_eval_config()
    cfg.backend_url = "http://10.0.0.2:8765"
    save_eval_config(cfg)
    assert get_backend_url() == "http://10.0.0.2:8765"


def test_backend_url_blank_falls_back_to_default(tmp_path, monkeypatch):
    """A blank URL (TUI edit cleared, stale file) must never persist — httpx
    raises on an empty base_url before the TUI can render a fix screen."""
    from eval.config import DEFAULT_BACKEND_URL, get_backend_url

    monkeypatch.setattr("eval.config.bibilab_home", lambda: tmp_path)
    tmp_path.joinpath("eval_config.json").write_text(
        json.dumps({"profiles": {}, "language": "zh", "backend_url": "  "})
    )
    assert get_backend_url() == DEFAULT_BACKEND_URL

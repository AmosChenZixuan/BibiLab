"""Tests for config thread safety, stale cache, and testability."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


class TestConfigThreadSafety:
    """Test config module for thread safety issues."""

    def test_save_config_invalidates_cache_internally_consistent(self, tmp_path: Path) -> None:
        """
        Verify that save_config() + load_config() roundtrip works correctly
        even when called from multiple threads.

        This tests the stale cache issue: if load_config() reads _config_cache
        while save_config() is concurrently setting it to None, we get stale data.
        """
        from bibilab import config as config_module

        # Patch bibilab_home to use temp dir
        with patch.object(config_module, "bibilab_home", return_value=tmp_path):
            config_module._config_cache = None  # Reset cache

            # Load initial config
            cfg1 = config_module.load_config()
            cfg1.ai.model = "gpt-4o"
            config_module.save_config(cfg1)

            # Now stress test: interleaved save/load from multiple threads
            errors: list[str] = []
            results: list[dict] = []

            def load_and_validate() -> None:
                try:
                    cfg = config_module.load_config()
                    # The loaded config should have our saved value
                    if cfg.ai.model != "gpt-4o":
                        errors.append(f"Stale cache: expected gpt-4o, got {cfg.ai.model}")
                    results.append({"model": cfg.ai.model})
                except Exception as e:
                    errors.append(str(e))

            def save_new_model(model: str) -> None:
                try:
                    cfg = config_module.load_config()
                    cfg.ai.model = model
                    config_module.save_config(cfg)
                except Exception as e:
                    errors.append(str(e))

            # Run concurrent loads and saves
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Start multiple savers
                for model in ["gpt-4o-mini", "claude-3", "gpt-4o"]:
                    executor.submit(save_new_model, model)

                # Start multiple loaders that check for stale data
                for _ in range(10):
                    executor.submit(load_and_validate)

            assert len(errors) == 0, f"Thread safety issues: {errors}"
            # Verify we got some results
            assert len(results) > 0

    def test_config_cache_reset_for_testing(self, tmp_path: Path) -> None:
        """
        Verify that _config_cache can be reset for test isolation.
        Without a reset mechanism, tests can pollute each other.
        """
        from bibilab import config as config_module

        with patch.object(config_module, "bibilab_home", return_value=tmp_path):
            # Set a known value
            cfg1 = config_module.load_config()
            cfg1.ai.model = "test-model-1"
            config_module.save_config(cfg1)

            # Another test or code path might want to reset the cache
            # Currently there's no public API to do this
            # After fix, there should be a way to reset
            assert hasattr(config_module, "_config_cache")

            # The cache should be usable and consistent
            cfg2 = config_module.load_config()
            assert cfg2.ai.model == "test-model-1"

            # If we manually reset, we should still be consistent
            config_module._config_cache = None
            cfg3 = config_module.load_config()
            assert cfg3.ai.model == "test-model-1"

    def test_concurrent_save_no_data_loss(self, tmp_path: Path) -> None:
        """
        Test that concurrent saves don't corrupt the config file.
        Uses file-based verification to check no data loss.
        """
        from bibilab import config as config_module

        with patch.object(config_module, "bibilab_home", return_value=tmp_path):
            config_module._config_cache = None

            # Pre-populate config
            cfg = config_module.load_config()
            config_module.save_config(cfg)

            errors: list[str] = []

            def save_different_models(model_suffix: str) -> None:
                try:
                    cfg = config_module.load_config()
                    cfg.ai.model = model_suffix
                    config_module.save_config(cfg)

                    # Verify file was written correctly
                    path = config_module._config_path()
                    if path.exists():
                        with path.open() as f:
                            data = json.load(f)
                        if data.get("ai", {}).get("model") != model_suffix:
                            errors.append(
                                f"File corruption: expected {model_suffix}, got {data.get('ai', {}).get('model')}"
                            )
                except Exception as e:
                    errors.append(str(e))

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(save_different_models, f"model-{i}") for i in range(8)]
                for f in futures:
                    f.result()

            assert len(errors) == 0, f"Data corruption: {errors}"

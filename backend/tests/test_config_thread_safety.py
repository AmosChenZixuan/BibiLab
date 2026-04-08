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
        while save_config() is concurrently setting it, we could get stale data.
        This test verifies no exceptions occur and final state is valid.
        """
        from bibilab import config as config_module

        # Patch bibilab_home to use temp dir
        with patch.object(config_module, "bibilab_home", return_value=tmp_path):
            config_module._config_cache = None  # Reset cache

            # Pre-populate config
            cfg = config_module.load_config()
            cfg.ai.model = "gpt-4o"
            config_module.save_config(cfg)

            # Concurrent save/load stress test
            errors: list[str] = []
            valid_models = {"gpt-4o", "gpt-4o-mini", "claude-3"}

            def load_and_validate() -> None:
                try:
                    cfg = config_module.load_config()
                    # Model should be one of the valid values
                    if cfg.ai.model not in valid_models:
                        errors.append(f"Invalid model: {cfg.ai.model}")
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
                futures = []
                # Submit savers
                for model in valid_models:
                    futures.append(executor.submit(save_new_model, model))
                # Submit loaders
                for _ in range(10):
                    futures.append(executor.submit(load_and_validate))
                # Wait for all
                for f in futures:
                    f.result()

            assert len(errors) == 0, f"Thread safety issues: {errors}"

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

        This test verifies:
        - All saves complete without exception
        - Final file is valid JSON with a valid model name from the set
        - No silent data loss (truncated files or corrupted JSON)

        NOTE: We do NOT verify that each thread's specific write persists
        immediately after its save (that's impossible with concurrent
        atomic os.replace() operations - thread A's write can be overwritten
        by thread B before thread A reads it back).
        """
        from bibilab import config as config_module

        with patch.object(config_module, "bibilab_home", return_value=tmp_path):
            config_module._config_cache = None

            # Pre-populate config
            cfg = config_module.load_config()
            config_module.save_config(cfg)

            errors: list[str] = []
            model_names = {f"model-{i}" for i in range(8)}

            def save_different_models(model_suffix: str) -> None:
                try:
                    cfg = config_module.load_config()
                    cfg.ai.model = model_suffix
                    config_module.save_config(cfg)
                except Exception as e:
                    errors.append(f"Save exception: {e}")

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(save_different_models, f"model-{i}") for i in range(8)]
                for f in futures:
                    f.result()

            # Verify no exceptions during saves
            assert len(errors) == 0, f"Save errors: {errors}"

            # Verify final file is valid JSON with a valid model name
            path = config_module._config_path()
            assert path.exists(), "Config file does not exist"

            with path.open() as f:
                content = f.read()

            # Check for truncated file (empty or incomplete JSON)
            assert content.strip(), "Config file is empty"
            assert content.strip().endswith("}"), "Config file appears truncated"

            # Check file is valid JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise AssertionError(f"Config file is not valid JSON: {e}")

            # Verify model is one of the valid saved values
            final_model = data.get("ai", {}).get("model")
            assert final_model in model_names, f"Final model {final_model!r} not in expected set {model_names}"

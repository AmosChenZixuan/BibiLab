"""Tests for search_mode_to_params (replaces params_for_type)."""


def test_search_mode_to_params_factual():
    from bibilab.pipeline.chat_tools import search_mode_to_params

    params = search_mode_to_params("factual", sources_total=10)
    assert params.depth_per_source == 1
    assert params.top_k == 4


def test_search_mode_to_params_breadth():
    from bibilab.pipeline.chat_tools import search_mode_to_params

    params = search_mode_to_params("breadth", sources_total=10)
    assert params.depth_per_source == 1
    assert params.top_k == 10  # capped at sources_total


def test_search_mode_to_params_breadth_small_list():
    from bibilab.pipeline.chat_tools import search_mode_to_params

    params = search_mode_to_params("breadth", sources_total=2)
    assert params.depth_per_source == 1
    assert params.top_k == 4  # degrades to factual


def test_search_mode_to_params_analytical():
    from bibilab.pipeline.chat_tools import search_mode_to_params

    params = search_mode_to_params("analytical", sources_total=20)
    assert params.depth_per_source == 4
    assert params.top_k == 20  # min(sources_total, 12*3=36) = 20


def test_search_mode_to_params_unknown_falls_back_to_factual():
    from bibilab.pipeline.chat_tools import search_mode_to_params

    params = search_mode_to_params("garbage", sources_total=10)
    assert params.depth_per_source == 1
    assert params.top_k == 4

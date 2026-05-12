"""Verify apply_source_filter and _format_filter_miss_message are deleted after #287."""


def test_apply_source_filter_is_deleted():
    """Verify apply_source_filter no longer exists after #287 deletion."""
    import bibilab.pipeline.embed as embed_module

    assert not hasattr(embed_module, "apply_source_filter"), "apply_source_filter should be deleted in #287"


def test_format_filter_miss_message_is_deleted():
    """Verify _format_filter_miss_message no longer exists after #287 deletion."""
    import bibilab.pipeline.chat_tools as chat_tools_module

    assert not hasattr(chat_tools_module, "_format_filter_miss_message"), (
        "_format_filter_miss_message should be deleted in #287"
    )

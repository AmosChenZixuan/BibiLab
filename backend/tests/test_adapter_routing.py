import pytest

from bibilab.adapters import get_adapter_for_platform, get_adapter_for_url
from bibilab.adapters.base import UnsupportedPlatformError
from bibilab.adapters.bilibili import BilibiliAdapter
from bibilab.adapters.tiktok import TikTokAdapter
from bibilab.adapters.youtube import YouTubeAdapter
from bibilab.config import BibilabConfig


@pytest.fixture
def cfg() -> BibilabConfig:
    return BibilabConfig()


@pytest.mark.parametrize(
    "url",
    [
        "https://www.bilibili.com/video/BV1abc123",
        "https://bilibili.com/video/BV1abc123",
        "https://space.bilibili.com/123/favlist?fid=456",
        "https://b23.tv/abcdef",
    ],
)
def test_url_routes_bilibili_domains(url: str, cfg: BibilabConfig):
    assert isinstance(get_adapter_for_url(url, cfg), BilibiliAdapter)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/playlist?list=PLabc",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=x",
    ],
)
def test_url_routes_youtube_domains(url: str, cfg: BibilabConfig):
    assert isinstance(get_adapter_for_url(url, cfg), YouTubeAdapter)


def test_platform_youtube(cfg: BibilabConfig):
    assert isinstance(get_adapter_for_platform("youtube", cfg), YouTubeAdapter)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.tiktok.com/@someuser/video/7371330159376370462",
        "https://www.tiktok.com/@someuser/collection/title-7111887189571160875",
        "https://vm.tiktok.com/ZMabcdef/",
        "https://vt.tiktok.com/ZSabcdef/",
        "https://www.tiktok.com/t/ZTabcdef/",
    ],
)
def test_url_routes_tiktok_domains(url: str, cfg: BibilabConfig):
    assert isinstance(get_adapter_for_url(url, cfg), TikTokAdapter)


def test_platform_tiktok(cfg: BibilabConfig):
    assert isinstance(get_adapter_for_platform("tiktok", cfg), TikTokAdapter)


def test_url_unknown_host_raises(cfg: BibilabConfig):
    with pytest.raises(UnsupportedPlatformError) as exc_info:
        get_adapter_for_url("https://example.com/watch", cfg)
    assert "example.com" in str(exc_info.value)


def test_url_no_suffix_spoofing(cfg: BibilabConfig):
    with pytest.raises(UnsupportedPlatformError):
        get_adapter_for_url("https://evilbilibili.com/video/BV1", cfg)


def test_platform_bilibili(cfg: BibilabConfig):
    assert isinstance(get_adapter_for_platform("bilibili", cfg), BilibiliAdapter)


def test_platform_unknown_raises(cfg: BibilabConfig):
    with pytest.raises(UnsupportedPlatformError) as exc_info:
        get_adapter_for_platform("nosuch", cfg)
    assert "nosuch" in str(exc_info.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://evil-youtube.com.attacker.org/watch?v=x",
        "https://youtube.com.evil.com/watch?v=x",
        "https://xn--youtube.com/watch?v=x",
        "https://byoutu.be/x",
        "https://tiktok.com.phish.io/@u/video/1",
    ],
)
def test_url_rejects_lookalike_hosts(url: str, cfg: BibilabConfig):
    with pytest.raises(UnsupportedPlatformError):
        get_adapter_for_url(url, cfg)

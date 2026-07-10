"""Platform adapter registry: URL/platform → PlatformAdapter dispatch."""

from urllib.parse import urlparse

from bibilab.adapters.base import PlatformAdapter, UnsupportedPlatformError
from bibilab.adapters.bilibili import BilibiliAdapter
from bibilab.adapters.tiktok import TikTokAdapter
from bibilab.adapters.youtube import YouTubeAdapter
from bibilab.config import BibilabConfig

# platform key → (registered domains, adapter factory); subdomains match implicitly
_REGISTRY = {
    "bilibili": (("bilibili.com", "b23.tv"), lambda cfg: BilibiliAdapter(cookie=cfg.accounts.bilibili.cookie)),
    "youtube": (("youtube.com", "youtu.be"), lambda cfg: YouTubeAdapter()),
    "tiktok": (("tiktok.com",), lambda cfg: TikTokAdapter()),
}


def get_adapter_for_platform(platform: str, cfg: BibilabConfig) -> PlatformAdapter:
    if platform not in _REGISTRY:
        raise UnsupportedPlatformError(platform)
    return _REGISTRY[platform][1](cfg)


def get_adapter_for_url(url: str, cfg: BibilabConfig) -> PlatformAdapter:
    host = (urlparse(url).hostname or "").lower()
    for domains, factory in _REGISTRY.values():
        if any(host == d or host.endswith(f".{d}") for d in domains):
            return factory(cfg)
    raise UnsupportedPlatformError(host or url)

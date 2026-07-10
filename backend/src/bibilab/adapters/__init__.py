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

# Cover-image CDN hosts per platform → optional Referer the CDN demands.
# routers/proxy.py derives its allowlist and Referer policy from this table,
# so a new platform can't land with working ingest but broken cover previews.
CDN_DOMAINS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "bilibili": (("hdslb.com",), "https://www.bilibili.com/"),
    "youtube": (("ytimg.com",), None),
    "tiktok": (("tiktokcdn.com", "tiktokcdn-us.com"), None),
}
assert CDN_DOMAINS.keys() == _REGISTRY.keys(), "every registered platform needs a CDN entry"


def host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def get_adapter_for_platform(platform: str, cfg: BibilabConfig) -> PlatformAdapter:
    if platform not in _REGISTRY:
        raise UnsupportedPlatformError(platform)
    return _REGISTRY[platform][1](cfg)


def get_adapter_for_url(url: str, cfg: BibilabConfig) -> PlatformAdapter:
    host = (urlparse(url).hostname or "").lower()
    for domains, factory in _REGISTRY.values():
        if any(host_matches(host, d) for d in domains):
            return factory(cfg)
    raise UnsupportedPlatformError(host or url)

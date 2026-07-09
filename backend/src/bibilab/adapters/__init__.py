"""Platform adapter registry: URL/platform → PlatformAdapter dispatch."""

from collections.abc import Callable
from urllib.parse import urlparse

from bibilab.adapters.base import PlatformAdapter, UnsupportedPlatformError
from bibilab.adapters.bilibili import BilibiliAdapter
from bibilab.config import BibilabConfig


def _bilibili(cfg: BibilabConfig) -> PlatformAdapter:
    return BilibiliAdapter(cookie=cfg.accounts.bilibili.cookie)


# platform key → (registered domains, adapter factory); subdomains match implicitly
_REGISTRY: dict[str, tuple[tuple[str, ...], Callable[[BibilabConfig], PlatformAdapter]]] = {
    "bilibili": (("bilibili.com", "b23.tv"), _bilibili),
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

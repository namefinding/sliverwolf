from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


_BLOCKED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
}


def validate_public_url(raw_url: str) -> None:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Unsupported URL: {raw_url}")

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ValueError(f"URL is missing hostname: {raw_url}")
    if hostname in _BLOCKED_HOSTS or hostname.endswith(".local"):
        raise ValueError(f"Blocked local URL: {raw_url}")

    try:
        ip_value = ipaddress.ip_address(hostname)
    except ValueError:
        return

    if ip_value.is_private or ip_value.is_loopback or ip_value.is_link_local or ip_value.is_reserved:
        raise ValueError(f"Blocked private URL: {raw_url}")

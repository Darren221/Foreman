"""SSRF protection for the HTTP fetch tool.

The fetch URL is LLM-authored (and influenced by untrusted web content the researcher
reads), so a fetch could be steered at the internal network or the cloud metadata
endpoint. Every real fetch validates that the URL's host resolves only to public
addresses, before connecting and on every redirect hop.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urlparse

Resolver = Callable[[str], list[str]]


def _default_resolve(host: str) -> list[str]:
    # info[4][0] is the address; the sockaddr tuple is typed loosely, so coerce to str.
    return [str(info[4][0]) for info in socket.getaddrinfo(host, None)]


def assert_public_url(url: str, resolve: Resolver = _default_resolve) -> None:
    """Raise `ValueError` unless `url` is an http(s) URL whose host resolves only to
    public addresses. Blocks loopback, private (RFC1918), link-local (including the
    cloud metadata IP `169.254.169.254`), and other reserved/multicast ranges.

    `resolve` is injectable so tests don't touch real DNS.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("url must be an http(s) URL")
    host = parsed.hostname
    if not host:
        raise ValueError("url has no host")
    for address in _addresses_for(host, resolve):
        ip = ipaddress.ip_address(address)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(
                f"url host {host!r} resolves to a non-public address ({address})"
            )


def _addresses_for(host: str, resolve: Resolver) -> list[str]:
    try:
        ipaddress.ip_address(host)  # already an IP literal: use it directly, no DNS
    except ValueError:
        return resolve(host)  # a hostname: resolve it
    return [host]

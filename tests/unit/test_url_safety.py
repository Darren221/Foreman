"""SSRF guard: the HTTP fetch tool must refuse URLs that resolve to internal or cloud
metadata addresses, so an LLM-authored URL can't be steered at the internal network."""

from __future__ import annotations

import pytest

from foreman.tools.net import assert_public_url


def test_blocks_internal_ip_literals() -> None:
    for url in [
        "http://127.0.0.1/x",  # loopback
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/",  # private
        "http://192.168.1.1/",  # private
        "http://[::1]/",  # IPv6 loopback
    ]:
        with pytest.raises(ValueError):
            assert_public_url(url)


def test_allows_public_addresses() -> None:
    assert_public_url("http://8.8.8.8/")  # public IP literal, no DNS
    # a hostname resolving to a public IP (stubbed resolver, no real DNS)
    assert_public_url("https://example.com/data", resolve=lambda host: ["93.184.216.34"])


def test_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError):
        assert_public_url("file:///etc/passwd")


def test_blocks_dns_rebinding_to_a_private_ip() -> None:
    # a public-looking host that resolves to a private address is still blocked
    with pytest.raises(ValueError):
        assert_public_url("http://trusted.example.com/", resolve=lambda host: ["10.1.2.3"])

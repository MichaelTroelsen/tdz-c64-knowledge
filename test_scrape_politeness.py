"""Politeness tests for outbound scraping (R11 in CODE-REVIEW.md).

The sources this tool targets are small volunteer-run retro-computing wikis
and archives. Before this, scrape_url sent no identifying User-Agent, ignored
robots.txt entirely, and treated a 429 "slow down" exactly like a 404 (drop
the page, immediately hammer the next one) - a realistic way to get
rate-limited or IP-banned.

Everything here runs against a local http.server, so no external site is
contacted by the test suite itself.
"""
import http.server
import threading
import time

import pytest

import server as server_module


class _PoliteHandler(http.server.BaseHTTPRequestHandler):
    """Serves a robots.txt and records the User-Agent of every request."""

    robots_body = b"User-agent: *\nDisallow: /private/\n"
    requests_seen = []
    fail_times = 0  # how many times /flaky should 503 before succeeding

    def log_message(self, *args):
        pass  # keep pytest output clean

    def do_GET(self):
        type(self).requests_seen.append((self.path, self.headers.get('User-Agent')))

        if self.path == '/robots.txt':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(self.robots_body)))
            self.end_headers()
            self.wfile.write(self.robots_body)
            return

        if self.path == '/flaky':
            prior = len([p for p, _ in self.requests_seen if p == '/flaky'])
            if prior <= type(self).fail_times:
                self.send_response(503)
                self.send_header('Retry-After', '0')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return

        body = b'<html><body>ok</body></html>'
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def local_site(monkeypatch):
    _PoliteHandler.requests_seen = []
    _PoliteHandler.fail_times = 0

    httpd = http.server.HTTPServer(('127.0.0.1', 0), _PoliteHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    # robots.txt results are cached per origin for the process lifetime.
    monkeypatch.setattr(server_module, '_robots_cache', {})
    try:
        yield base, _PoliteHandler
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_requests_send_an_identifying_user_agent(local_site):
    base, handler = local_site
    resp = server_module.http_get_polite(f"{base}/page")
    assert resp.status_code == 200

    agents = [ua for path, ua in handler.requests_seen if path == '/page']
    assert agents, "no request reached the server"
    ua = agents[0]
    assert 'tdz-c64-knowledge' in ua, f"unidentifying User-Agent: {ua!r}"
    assert 'http' in ua, f"User-Agent should carry a contact URL: {ua!r}"


def test_robots_disallow_is_respected(local_site):
    base, _ = local_site
    assert server_module.robots_allows(f"{base}/public/page") is True
    assert server_module.robots_allows(f"{base}/private/secret") is False


def test_robots_result_is_cached_per_origin(local_site):
    base, handler = local_site
    for _ in range(5):
        server_module.robots_allows(f"{base}/page")

    fetches = [p for p, _ in handler.requests_seen if p == '/robots.txt']
    assert len(fetches) == 1, (
        f"robots.txt fetched {len(fetches)} times - a crawl of N pages must not "
        "cost N robots.txt requests"
    )


def test_missing_robots_txt_fails_open(monkeypatch):
    """An unreachable robots.txt must not block a scrape the user asked for."""
    monkeypatch.setattr(server_module, '_robots_cache', {})
    # Port 9 (discard) refuses/blackholes - stands in for "no robots.txt".
    assert server_module.robots_allows("http://127.0.0.1:9/page") is True


def test_robots_can_be_disabled(local_site, monkeypatch):
    base, _ = local_site
    monkeypatch.setattr(server_module, 'RESPECT_ROBOTS', False)
    assert server_module.robots_allows(f"{base}/private/secret") is True


def test_transient_5xx_is_retried_with_backoff(local_site):
    base, handler = local_site
    handler.fail_times = 2  # two 503s, then success

    started = time.time()
    resp = server_module.http_get_polite(f"{base}/flaky", base_delay=0.05, max_attempts=4)
    elapsed = time.time() - started

    assert resp.status_code == 200, "gave up on a retryable 503"
    attempts = len([p for p, _ in handler.requests_seen if p == '/flaky'])
    assert attempts == 3, f"expected 3 attempts (2 failures + success), got {attempts}"
    assert elapsed >= 0.05, "retried with no delay at all - that is not backoff"


def test_gives_up_after_max_attempts(local_site):
    base, handler = local_site
    handler.fail_times = 99  # never succeeds

    resp = server_module.http_get_polite(f"{base}/flaky", base_delay=0.01, max_attempts=3)
    assert resp.status_code == 503, "should surface the final failure, not raise"
    attempts = len([p for p, _ in handler.requests_seen if p == '/flaky'])
    assert attempts == 3, f"expected exactly max_attempts=3 tries, got {attempts}"


def test_scrape_url_refuses_a_robots_disallowed_url(local_site, tmp_path, monkeypatch):
    """The check must gate scrape_url itself, not just the helpers."""
    base, _ = local_site
    monkeypatch.setenv("ALLOWED_DOCS_DIRS", str(tmp_path))
    monkeypatch.setenv("AUTO_EXTRACT_ENTITIES", "0")

    kb = server_module.KnowledgeBase(str(tmp_path / "data"))
    try:
        result = kb.scrape_url(f"{base}/private/secret", follow_links=False)
    finally:
        kb.close()

    assert result['status'] == 'failed'
    assert 'robots.txt' in result['error']
    assert result['docs_added'] == 0


def test_crawl_defaults_are_gentle():
    """Defaults matter more than the knobs - most callers never pass these."""
    import inspect

    sig = inspect.signature(server_module.KnowledgeBase.scrape_url)
    threads = sig.parameters['threads'].default
    delay = sig.parameters['delay'].default

    assert threads <= 4, f"default threads={threads} is too aggressive for small sites"
    assert delay >= 300, f"default delay={delay}ms is too short between requests"

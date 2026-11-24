"""Groww client wrapper: rate limiting, retries with backoff, automatic re-login. Thread-safe."""
import os
import threading
import time


def _real_factory():
    from growwapi import GrowwAPI
    key = os.getenv("GROWW_API_KEY", "")
    totp = os.getenv("GROWW_TOTP_SECRET", "")
    if totp:
        import pyotp
        token = GrowwAPI.get_access_token(api_key=key, totp=pyotp.TOTP(totp).now())
    else:
        token = GrowwAPI.get_access_token(api_key=key, secret=os.getenv("GROWW_API_SECRET", ""))
    return GrowwAPI(token)


class Client:
    def __init__(self, factory=None, rps=5.0):
        self._factory = factory or _real_factory
        self.rps = float(rps)
        self._lock = threading.Lock()
        self._next = 0.0
        self._g, self._login_at = None, 0.0
        self.n_requests = 0
        self.n_retries = 0

    def _get(self):
        with self._lock:
            if self._g is None or time.time() - self._login_at > 3 * 3600:   # tokens expire daily at 6 AM
                self._g, self._login_at = self._factory(), time.time()
            return self._g

    def _relogin(self):
        with self._lock:
            self._g, self._login_at = self._factory(), time.time()

    def _throttle(self):
        with self._lock:
            now = time.time()
            wait = self._next - now
            self._next = max(now, self._next) + 1.0 / self.rps
        if wait > 0:
            time.sleep(wait)

    def const(self, name):
        return getattr(self._get(), name)

    def call(self, method, **kw):
        delay = 1.0
        for attempt in range(6):
            self._throttle()
            try:
                self.n_requests += 1
                return getattr(self._get(), method)(**kw)
            except Exception as e:  # noqa
                name, msg = type(e).__name__.lower(), str(e).lower()
                code = str(getattr(e, "code", "") or "")
                if not code:
                    import re
                    m = re.search(r"error (\d{3})", msg)
                    code = m.group(1) if m else ""
                if code in ("401", "403") or "unauthor" in msg or "expired" in msg or "auth" in name:
                    if attempt >= 2:
                        raise
                    self._relogin()
                    continue
                if code.startswith("4") and code not in ("408", "429"):
                    raise                      # bad request / not found: retrying can't help
                if code == "429" or "ratelimit" in name or ("rate" in msg and "limit" in msg):
                    self.rps = max(1.0, self.rps / 2)
                if attempt == 5:
                    raise
                self.n_retries += 1
                time.sleep(delay)
                delay = min(delay * 2, 30)

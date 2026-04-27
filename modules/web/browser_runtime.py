from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable, TypeVar

T = TypeVar("T")


class SharedPlaywrightBrowser:
    def __init__(
        self,
        *,
        browser_channel: str = "msedge",
        headless: bool = False,
    ) -> None:
        self.browser_channel = browser_channel
        self.headless = headless
        self._ensure_lock = Lock()
        self._playwright = None
        self._browser = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright-browser")
        atexit.register(self.close)

    def run_with_page(self, callback: Callable[[object], T]) -> T:
        future: Future[T] = self._executor.submit(self._run_with_page, callback)
        return future.result()

    def close(self) -> None:
        executor = self._executor
        if executor is None:
            return
        self._executor = None
        try:
            future = executor.submit(self._close_worker)
            future.result(timeout=5)
        except Exception:
            pass
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _ensure_browser(self):
        if self._browser is not None:
            return self._browser
        with self._ensure_lock:
            if self._browser is not None:
                return self._browser
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("playwright is not installed") from exc

            playwright = sync_playwright().start()
            launch_kwargs = {"headless": self.headless}
            channel = self.browser_channel.strip().lower()
            if channel and channel != "chromium":
                launch_kwargs["channel"] = channel
            self._playwright = playwright
            self._browser = playwright.chromium.launch(**launch_kwargs)
            return self._browser

    def _run_with_page(self, callback: Callable[[object], T]) -> T:
        browser = self._ensure_browser()
        page = browser.new_page()
        try:
            return callback(page)
        finally:
            try:
                page.close()
            except Exception:
                pass

    def _close_worker(self) -> None:
        browser = self._browser
        playwright = self._playwright
        self._browser = None
        self._playwright = None
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass

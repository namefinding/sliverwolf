from __future__ import annotations

import os
import webbrowser

from local_agent.modules.web.safety import validate_public_url


class SystemBrowserOpener:
    name = "system"

    def open(self, url: str) -> dict:
        validate_public_url(url)
        normalized_url = url.strip()

        if os.name == "nt":
            os.startfile(normalized_url)
            opened = True
        else:  # pragma: no cover - Windows is the primary runtime here.
            opened = bool(webbrowser.open_new_tab(normalized_url))

        return {
            "path": normalized_url,
            "url": normalized_url,
            "opened": opened,
            "opener": self.name,
        }

_need_bypass = self.bypass_proxy or self._is_ollama() or any(
    h in (self.base_url or "").lower()
    for h in ("localhost", "127.0.0.1")
)
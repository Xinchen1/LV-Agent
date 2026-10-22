"""Lazy `requests` proxy.

requests+urllib3 import costs ~50ms per process(itself+ssl+urllib3 chain),
but many tool paths list tools/never touch network. Import-site compatible:
`from .lazy_http import requests` drops into the `requests.<attr>` call sites.
"""

import importlib.util as _ils

if _ils.find_spec("requests") is not None:

    class _LazyRequests:
        """Attribute访问时才真正 import requests;后续调用直通真实模块。"""

        _mod = None

        def __getattr__(self, name):
            import requests as mod
            _LazyRequests._mod = mod
            return getattr(mod, name)

    requests = _LazyRequests()

else:  # 未安装: 保持调用点行为一致(报 ImportError 而不是 AttributeError)

    class _LazyRequests:
        def __getattr__(self, name):
            raise ImportError("requests is not installed; `pip install requests`")

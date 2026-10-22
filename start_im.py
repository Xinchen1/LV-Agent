#!/usr/bin/env python3
"""Start an IM bridge for the desktop Messaging view.

Supported platforms: feishu, dingtalk, wechat.
Structured events are printed as lines prefixed with @@IM@@ (JSON).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent_project.im_bridge.base import AgentWorker, emit, log  # noqa: E402

PLATFORMS = {
    "feishu": ("lark_oapi", "lark-oapi", ("FEISHU_APP_ID", "FEISHU_APP_SECRET")),
    "dingtalk": ("dingtalk_stream", "dingtalk-stream", ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET")),
    "wechat": ("itchat", "itchat", ()),
}


def check(platform: str) -> int:
    module, package, keys = PLATFORMS[platform]
    missing = [p for p in (module,) if importlib.util.find_spec(p) is None]
    missing_env = [k for k in keys if not (os.getenv(k) or "").strip()]
    result = {
        "platform": platform,
        "module": module,
        "installed": not missing,
        "missing_env": missing_env,
    }
    import json

    print(json.dumps(result, ensure_ascii=False))
    return 0 if not missing and not missing_env else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--register", action="store_true", help="Feishu: create an app by QR scan")
    args = parser.parse_args()

    if args.check:
        return check(args.platform)

    platform = args.platform
    try:
        if platform == "feishu":
            from agent_project.im_bridge.feishu import FeishuBridge, register_app

            bridge_cls = FeishuBridge
        elif platform == "dingtalk":
            from agent_project.im_bridge.dingtalk import DingTalkBridge, register_app as register_dingtalk

            bridge_cls = DingTalkBridge
        else:
            from agent_project.im_bridge.wechat import WeChatBridge

            bridge_cls = WeChatBridge

        emit("status", platform=platform, state="starting")
        worker = AgentWorker()
        if platform == "feishu" and args.register:
            app_id, app_secret = register_app()
            os.environ["FEISHU_APP_ID"] = app_id
            os.environ["FEISHU_APP_SECRET"] = app_secret
            bridge = bridge_cls(worker, app_id=app_id, app_secret=app_secret)
        elif platform == "dingtalk" and args.register:
            client_id, client_secret = register_dingtalk()
            os.environ["DINGTALK_CLIENT_ID"] = client_id
            os.environ["DINGTALK_CLIENT_SECRET"] = client_secret
            bridge = bridge_cls(worker, client_id=client_id, client_secret=client_secret)
        else:
            bridge = bridge_cls(worker)
        bridge.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001
        log(f"{platform} failed: {exc}")
        traceback.print_exc()
        emit("error", platform=platform, message=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

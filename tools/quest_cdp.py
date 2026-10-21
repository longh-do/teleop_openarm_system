#!/usr/bin/env python3
"""Debug the teleop page inside the Quest browser from the laptop (Quest on USB, developer mode, USB debugging allowed).
  adb forward tcp:9222 localabstract:chrome_devtools_remote
  python3 tools/quest_cdp.py eval "navigator.userAgent"
  python3 tools/quest_cdp.py watch 60        # print the page's console + errors for 60 s
"""
import asyncio
import json
import sys

import aiohttp


async def page_ws(s):
    async with s.get("http://localhost:9222/json") as r:
        pages = await r.json()
    for p in pages:
        if p["type"] == "page" and ":8443" in p["url"]:
            return p["webSocketDebuggerUrl"]
    sys.exit(f"teleop page not open in the Quest browser: {[p['url'] for p in pages]}")


async def main():
    mode = sys.argv[1]
    async with aiohttp.ClientSession() as s:
        url = await page_ws(s)
        async with s.ws_connect(url, max_msg_size=1 << 24) as ws:
            n = 0

            async def call(method, **params):
                nonlocal n
                n += 1
                await ws.send_json({"id": n, "method": method, "params": params})
                my = n
                async for m in ws:
                    d = json.loads(m.data)
                    if d.get("id") == my:
                        return d
                    show(d)

            def show(d):
                meth = d.get("method", "")
                if meth == "Runtime.consoleAPICalled":
                    p = d["params"]
                    print(f"[console.{p['type']}]", " ".join(str(a.get("value", a.get("description", ""))) for a in p["args"]), flush=True)
                elif meth == "Runtime.exceptionThrown":
                    print("[exception]", d["params"]["exceptionDetails"].get("exception", {}).get("description"), flush=True)
                elif meth == "Log.entryAdded":
                    e = d["params"]["entry"]
                    print(f"[log.{e['level']}]", e["text"], flush=True)

            if mode == "eval":
                r = await call("Runtime.evaluate", expression=sys.argv[2], returnByValue=True, awaitPromise=True, userGesture=True)
                res = r.get("result", {})
                print(json.dumps(res.get("result", {}).get("value", res), indent=1))
            elif mode == "watch":
                await call("Runtime.enable")
                await call("Log.enable")
                try:
                    async with asyncio.timeout(float(sys.argv[2])):
                        async for m in ws:
                            show(json.loads(m.data))
                except TimeoutError:
                    pass


asyncio.run(main())

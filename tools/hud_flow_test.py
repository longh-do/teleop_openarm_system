#!/usr/bin/env python3
"""Walk the whole operator flow against a running server and print what the headset panel would say after each step.
  .venv/bin/python tools/hud_flow_test.py [--url wss://127.0.0.1:8444/ws]
Steps: AR starts -> squeeze before aligning (refused) -> B align -> drive -> release -> Y stop -> thumbstick-hold power on.
The flow switches the arms on and drives the right arm, so point it at a simulator. The default URL expects a server
started with --port 8444 (teleop_server.py's own default is 8443).
"""
import argparse
import asyncio
import json
import ssl
import time

import aiohttp


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="wss://127.0.0.1:8444/ws")
    args = ap.parse_args()
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    last = {}
    async with aiohttp.ClientSession() as s, s.ws_connect(args.url, ssl=ctx) as ws:
        async def reader():
            async for m in ws:
                d = json.loads(m.data)
                if d.get("t") == "status":
                    last["hud"] = d.get("hud")
        rt = asyncio.ensure_future(reader())
        t0, seq = time.monotonic(), 0

        def hand(sq=0.0, b=(0, 0, 0), p=(0.2, 1.0, -0.3)):
            return {"ok": True, "p": list(p), "q": [0, 0, 0, 1], "aim": [0, 0, -1], "sq": sq, "tr": 0.0, "b": list(b), "stick": [0, 0]}

        async def stream(seconds, right=None, left=None, label=""):
            nonlocal seq
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                seq += 1
                r = right() if callable(right) else (right or hand())
                lft = left() if callable(left) else (left or hand())
                await ws.send_str(json.dumps({"t": "xr", "seq": seq, "ts": (time.monotonic() - t0) * 1000, "fps": 90,
                                              "head": None, "hands": {"left": lft, "right": r}}))
                await asyncio.sleep(1 / 90)
            h = last.get("hud") or {}
            print(f"{label:38s} -> {h.get('headline')!r:30s} L:{h.get('left'):9s} R:{h.get('right'):9s} {h.get('detail')}")

        await stream(1.0, label="1 page open (no AR yet)")
        await ws.send_str(json.dumps({"t": "cmd", "cmd": "xr_start"}))
        await stream(0.6, label="2 AR started")
        await stream(0.6, right=hand(sq=1.0), label="3 squeeze before aligning")
        await stream(0.3, right=hand(sq=0.0), label="   release")
        await stream(0.3, right=hand(b=(0, 1, 0)), label="4 press B (align)")
        await stream(0.3, right=hand(b=(0, 0, 0)), label="   B released")
        k = {"i": 0}

        def moving():
            k["i"] += 1
            return hand(sq=1.0, p=(0.2, 1.0 + 0.0005 * k["i"], -0.3 - 0.0005 * k["i"]))
        await stream(1.5, right=moving, label="5 squeeze + move right hand")
        await stream(0.6, right=hand(sq=0.0, p=(0.2, 1.07, -0.37)), label="6 release")
        await stream(0.3, left=hand(b=(0, 1, 0)), label="7 press Y (STOP)")
        await stream(4.0, left=hand(b=(0, 0, 0)), label="   ...4 s later")
        await stream(1.5, right=hand(b=(0, 0, 1)), label="8 hold right thumbstick 1.5 s")
        await stream(1.0, right=hand(b=(0, 0, 0)), label="   released")
        await stream(8.0, label="   ...8 s later")
        rt.cancel()


asyncio.run(main())
